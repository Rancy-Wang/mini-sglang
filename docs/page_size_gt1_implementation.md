# page_size > 1 实现记录

## 目标

这次实现的目标是让 Mini-SGLang 在 `--page-size` 大于 1 时支持核心 KV cache、普通 attention backend metadata、CUDA graph capture/replay 以及 sparse prefix-cache 路径。

原有实现里虽然部分 allocator 已经按页分配 KV cache，但很多上层路径仍默认 `page_size == 1`：

- 全局 `page_table` 以 token slot 形式保存，每个 token 位置对应一个 KV slot。
- FlashAttention、TensorRT-LLM、FlashInfer 的 backend metadata 需要的是 page id 或 page-level indptr。
- FlashInfer 之前把 KV cache 展平成 `page_size = 1` 的形状传入 wrapper。
- context-mask prefill 和 sparse cache 路径直接拒绝 `page_size > 1`。
- prefill admission 估算容量时按 token 数估计，会在 `page_size > 1` 时低估实际页占用。

本次实现的核心原则是：内部调度和全局表继续保留 token-addressed 表达，只有在交给 backend 或 prefix cache 做 page-level 操作时转换成 page-aware 形式。

## 核心实现逻辑

### 1. 明确 token 和 page 两套单位

全局 `page_table` 仍然保持“按 token 位置索引”的语义。例如 `page_size = 4` 时，一个物理页会在表里写入 4 个连续 token slot：

```text
token-addressed page_table row:
[0, 1, 2, 3, 8, 9, 10, 11, ...]

backend page id:
[0, 2, ...]
```

因此 backend metadata 不能直接使用 `page_table[... : seq_len]`，而要按 `page_size` stride 取每页第一个 token slot，并在 `page_size > 1` 时除以 `page_size` 转成 page id。

### 2. Backend metadata 使用 page-level 表达

FlashAttention 和 TensorRT-LLM 使用固定二维 page table：

- 每个 request 一行。
- 行宽是 `ceil(max_seqlen_k / page_size)`。
- 表项是 page id，不是 token slot。

FlashInfer 使用 ragged paged KV metadata：

- `paged_kv_indptr` 使用 page count 的前缀和。
- `paged_kv_indices` 是每个 request 的 page id 列表。
- `paged_kv_last_page_len` 记录最后一页有效 token 数。
- KV cache 不再 flatten 成 `page_size = 1`，而是按真实 paged cache shape 传给 FlashInfer。

### 3. CUDA graph buffer 按 page count 分配

原来 capture graph 用 `max_seq_len // page_size`，在非整除时会少分配一页。现在统一使用 `ceil(max_seq_len / page_size)`。

FlashInfer 的 graph replay 还需要把真实 request metadata 复制进 capture wrapper 的固定 buffer：

- page-level indptr。
- ragged page indices。
- last page length。

否则 `page_size > 1` 时 replay 仍可能沿用 capture 阶段的 dummy metadata。

### 4. Context-mask prefill 的当前边界

FlashAttention 的 FA4 context-mask path 需要 dense K/V 输入。原实现只支持 `page_size == 1`，直接把 page table 当 token table 使用。

当前已处理的部分是 FA4 dense-gather path：

- 根据 `cache_seqlens` 计算真实 key length。
- 根据 `page_size` 计算需要 gather 的页数。
- 从 paged KV cache 中 `index_select` 对应 page。
- flatten page 维度后只保留真实 `key_len`，丢掉 partial final page 的 padding token。

这样 FA4 mask kernel 仍然看到 dense logical sequence，但底层 cache 可以是多 token 页。

尚未完成的是 FA3/FlashInfer 的 segmented context-mask 编译路径。该路径目前仍通过
`compile_context_page_tables(...)` 按绝对 token position 读取 `page_table`，返回 token
slot 而不是后端需要的 page id；如果直接用于 `page_size > 1`，metadata 语义不完整。
因此这部分需要后续单独扩展 `python/minisgl/attention/base.py`，把 segment key
positions 编译成 page-level indptr/page id/last-page-len。

### 5. Sparse cache 在 mixed page 时做 KV compaction

Drop Message / sparse prefix filtering 的语义是：后续 kept token 的 KV 仍然复用 full stream 中已经计算好的 KV，因此这些 KV 仍然 condition 在当时可见的 dropped message 上。`page_size == 1` 时可以直接按 token 粒度复用任意 kept KV slot。

`page_size > 1` 时，backend metadata 只能表达普通 paged KV，不能直接表达页内 hole。如果某个 active cached prefix 由非连续 token slot 组成，例如一个物理页中部分 kept、部分 dropped，就不能直接把这些 slot 作为 active page table 交给 attention backend。

新逻辑是：

- sparse match 仍然按 token 粒度选择所有 kept KV，和 `page_size == 1` 保持同样的 conditioning 语义。
- 如果选出的 active KV slot 已经能表示成完整 paged layout，并且 active cached prefix 长度页对齐，例如整页 kept、整页 dropped 后接整页 kept，则直接复用原 slot。
- 如果选出的 active KV slot 不能表示成 paged layout，例如页内 mixed kept/dropped，则标记 `requires_compaction`。
- 调度分配页时，为 cached prefix 分配新的 active pages，把 full-cache 中的 kept KV slot 复制/压实到这些新页。
- 压实后，attention backend 看到的是普通连续 active paged KV，但 KV 值仍然来自 full stream cache，不会重新 prefill 成去掉 dropped message 后的 KV。

### 6. Sparse/full cache commit 按页对齐

Sparse finished cache commit 会先重建 full-position KV slot 列表。对于 `page_size > 1`：

- 如果中间出现 hole，只 cache 到第一个 hole 之前。
- cacheable length 再向下对齐到完整页。
- 新分配但没有被 prefix cache 接管的 active slot 会释放。

Full-stream context cache commit 移除了 `page_size == 1` 限制，并补齐了 partial final page 的释放逻辑：prefix cache 只会接管完整页，未插入的尾部 token slot 会被释放。

### 7. Prefill admission 按整页预留容量

原来的 prefill admission 使用：

```text
input_len - cached_len + output_len
```

这在 `page_size > 1` 时会低估容量。例如只剩 1 个物理页时，两个很短请求分别看起来只需要 2 个 token，但每个请求实际上都至少占用一整页。

现在预估为：

```text
(ceil((input_len + output_len) / page_size) - ceil(cached_len / page_size)) * page_size
```

当 `page_size == 1` 时，该公式退化为原来的 token 级估算。

如果 sparse cached prefix 需要 KV compaction，cached prefix 会被复制到新页，不能继续当作节省下来的容量。因此 admission 在 `compact_cached_prefix = True` 时不会扣减 cached pages。

## 逐文件变化

### `python/minisgl/utils/misc.py`

新增 page helper：

- `page_count(num_tokens, page_size)`：返回 token 长度需要的页数，使用 ceil division。
- `last_page_len(num_tokens, page_size)`：返回最后一页真实有效 token 数；整除时返回 `page_size`，长度为 0 时返回 0。

### `python/minisgl/utils/__init__.py`

导出新增 helper：

- `page_count`
- `last_page_len`

让 attention 和 scheduler 代码可以通过 `minisgl.utils` 统一引用。

### `python/minisgl/attention/utils.py`

新增 backend metadata 转换 helper：

- `make_backend_page_table(...)`
  - 从 token-addressed global page table 中按 `page_size` stride 抽取每页代表 slot。
  - `page_size > 1` 时将 token slot 转成 page id。
  - 给 FlashAttention 和 TensorRT-LLM 使用。

- `make_paged_kv_indices(...)`
  - 为 FlashInfer 生成 ragged page id 列表。
  - 每个 request 只取到 `req.device_len` 覆盖的页。

- `make_page_indptr_cpu(...)`
  - 根据 sequence token length 生成 page-count indptr。
  - 用于 FlashInfer `paged_kv_indptr`。

- `make_last_page_len_cpu(...)`
  - 为 FlashInfer 生成每个 request 的 last-page effective length。

### `python/minisgl/attention/fa.py`

主要变化：

- 使用 `make_backend_page_table` 生成 backend page table。
- CUDA graph capture buffer 宽度改为 `page_count(max_seq_len, page_size)`。
- 移除 FlashAttention context-mask prefill 的 `page_size == 1` 限制。
- FA4 context-mask path 从 paged KV cache 中 gather dense K/V：
  - 通过 `cache_seqlens` 得到真实 key length。
  - 按 page count 选择物理页。
  - flatten 后裁掉 partial final page padding。

### `python/minisgl/attention/trtllm.py`

主要变化：

- 使用 `make_backend_page_table` 生成 TensorRT-LLM backend page table。
- CUDA graph capture buffer 宽度改为 `page_count(max_seq_len, page_size)`，避免非整除长度少分配一页。

### `python/minisgl/attention/fi.py`

主要变化：

- `FIMetadata.page_size` 从 `Literal[1]` 改为普通 `int`，移除 `page_size == 1` assertion。
- backend 初始化时记录真实 `self.page_size`。
- `forward()` 不再将 KV cache flatten 成 `page_size = 1`，直接传真实 paged KV cache。
- `prepare_metadata()` 改为生成 FlashInfer 需要的 paged metadata：
  - KV indptr 使用 page-count prefix sum。
  - Q indptr 继续保持 token-level query prefix sum。
  - indices 使用 page id ragged list。
  - last_page_len 使用真实最后一页长度。
  - metadata 的 `page_size` 使用真实配置。
- CUDA graph capture buffer 宽度改为 page count。
- CUDA graph replay 时复制真实 paged metadata 到 capture wrapper 的固定 buffer：
  - `cu_seqlens_k`
  - `indices`
  - `last_page_len`

### `python/minisgl/core.py`

新增内部字段：

- `Req.compact_cached_prefix`：标记该 request 的 cached prefix 在 forward 前需要从 full-cache slot 压实复制到新的 active pages。

该字段只用于 scheduler/cache 内部，不改变 tokenizer 或用户请求协议。

### `python/minisgl/kvcache/base.py`

在 `BaseKVCachePool` 增加接口：

- `copy_slots(src, dst)`：跨所有 KV cache layer 将一组 token slot 的 K/V 值复制到另一组 token slot。

### `python/minisgl/kvcache/mha_pool.py`

实现 `copy_slots(src, dst)`：

- 对每一层的 K cache 和 V cache 做 `index_select` + `index_copy_`。
- 用于 sparse mixed-page cached prefix compaction。

### `python/minisgl/scheduler/scheduler.py`

移除 contextual prefill 在 `page_size != 1` 时直接报错的 guard。

具体影响：

- `flashinfer-mask`
- `flashattention-mask`
- 其他非 `staged` contextual prefill mode

这些路径不再因为 page size 被 scheduler 层拒绝。

### `python/minisgl/scheduler/cache.py`

主要变化：

- 新增 `_is_page_representable_active_prefix(...)`：
  - 判断 token-level active KV slot 列表是否可以直接表示为普通 paged KV layout。
  - `page_size == 1` 总是可表示。

- `match_req(...)`：
  - 返回所有 kept cached slot，保留 page_size=1 的 drop-message conditioning 语义。
  - 将 compaction 需求通过 `ContextMatchResult.requires_compaction` 传给 prefill。

- `allocate_paged(...)`：
  - 在普通 extend page 分配前调用 `_compact_cached_prefixes(...)`。

- 新增 `_compact_cached_prefixes(...)`：
  - 为需要 compaction 的 cached prefix 分配新的 active pages。
  - 调用 `get_global_ctx().kv_cache.copy_slots(...)` 把 full-cache kept KV slot 复制到新 slot。
  - 复制后解锁原 full-prefix handle，并把 request 的 cache handle 重置为空 handle。
  - 将 `initial_active_cached_len` 重置为 0，使后续 commit/free 逻辑把复制出来的 slot 当作 request 自己分配的 slot 管理。

- `_cache_finished_sparse_req(...)`：
  - 移除 `page_size == 1` 限制。
  - cacheable length 向下对齐到完整页。
  - 保持对 hole、overlap 和 missing kept positions 的校验。
  - 释放未被 prefix cache 接管的新分配 active slot。

- `_cache_finished_full_req(...)`：
  - 移除 `page_size == 1` 限制。
  - commit 后释放未插入 prefix cache 的 partial final page tail。

## 已知限制

- FA3/FlashInfer 的 segmented context-mask path 仍需 page-aware compiler 支持；当前方案
  不应声明该路径已经完整支持 `page_size > 1`。
- Drop-aware eviction 仍保持 `page_size == 1` 限制；方案1只处理非 drop-aware 的
  delta-marker/sparse reuse。
- mixed-page sparse prefix 采用 request-local copy，长前缀会产生明显 KV copy 成本和临时
  额外页占用。

### `python/minisgl/scheduler/prefill.py`

主要变化：

- 新增 `_estimate_tokens_to_reserve(...)`。
- admission check 和 `reserved_size` 增量都使用整页容量估算。
- `page_size == 1` 下行为与原公式等价。
- 将 `requires_compaction` 从 match result 传入 `Req.compact_cached_prefix`。
- compaction 请求按未命中 cached prefix 的容量保守预留，因为 cached KV 需要复制到新页。

### `tests/core/test_page_size_helpers.py`

新增回归测试覆盖：

- attention helper 是否能把 token-addressed table 正确转成 page-level metadata。
- FlashInfer page indptr / ragged indices / last_page_len 是否正确。
- sparse match 是否保留 mixed page 后续 kept cached token，并标记 compaction。
- sparse cached prefix compaction 是否复制 kept KV slot 到新的 active pages。
- sparse finished cache 是否只插入完整页，并释放 partial page。
- full-stream context cache 是否接受 partial final page，并保持 cache integrity。
- prefill admission 是否按整页预留容量，避免多请求低估容量。

## 验证

在新 conda 环境 `minisgl-page-size` 中完成验证：

```bash
PYTHONPATH=python conda run -n minisgl-page-size python -m py_compile \
  python/minisgl/core.py \
  python/minisgl/kvcache/base.py \
  python/minisgl/kvcache/mha_pool.py \
  python/minisgl/utils/misc.py \
  python/minisgl/utils/__init__.py \
  python/minisgl/attention/utils.py \
  python/minisgl/attention/fa.py \
  python/minisgl/attention/trtllm.py \
  python/minisgl/attention/fi.py \
  python/minisgl/scheduler/cache.py \
  python/minisgl/scheduler/prefill.py \
  python/minisgl/scheduler/scheduler.py \
  tests/core/test_page_size_helpers.py
```

```bash
PYTHONPATH=python conda run -n minisgl-page-size python -m pytest tests/core -q
```

结果：

- `py_compile` 通过。
- `tests/core` 通过：`12 passed, 1 warning`。
- `git diff --check` 通过。

测试中的 warning 来自可选的 `tvm_ffi` torch C DLPack JIT extension，未影响本次 page-size 相关测试结果。

## 当前边界和后续建议

本次实现覆盖了核心 CPU metadata/cache 测试和 Python 静态编译，但还没有在真实 GPU 模型服务上做端到端验证。建议后续补充：

- 使用 `--page-size 4` 或 backend 支持的更大 page size 跑最小模型 prefill/decode smoke test。
- 分别覆盖 `fa`、`fi`、`trtllm` backend 组合。
- 覆盖 contextual prefill 的 `flashattention-mask` 和 `flashinfer-mask` 真实请求。
- 在 CUDA graph enabled 的 decode 路径上验证 FlashInfer replay metadata。
