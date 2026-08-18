# mini-sglang Context System 项目上下文

本文解释 2026-08-12 当前 working tree 中的 Context System。它是机制导航，不是修改
授权；源码继续变化时，必须重新检查路径、符号和行号。当前默认组合是
`radix_drop_key_mode="delta-marker"` 与 `contextual_prefill_mode="mask"`，而旧会话中
把 message 文本或 `symbol` 当默认 Radix 状态的 pasted-text 说明已经过时
（`python/minisgl/scheduler/config.py:15-21`，`SchedulerConfig`）。

## 一、输入、完整模板与 token provenance

公开 `drop_rule.type` 选择 `message_drop`、`text_drop` 或 `thinking_drop`。三种规则分别按
完整 message owner、严格原始 content 子串、结构化 assistant thinking 选择内容，但都会
编译到同一绝对 token-position delta wire。旧顶层 `drop_message` 只作为
`MessageDropRule` 兼容入口；新旧字段同时出现会拒绝。结构、role 对齐、occurrence、子串和
thinking 来源在 `python/minisgl/tokenizer/drop_rules.py` 校验，HTTP 入口在
`python/minisgl/server/api_server.py` 转成 canonical wire。

消息边界不是把每条消息单独 tokenize 后拼接。`TokenizeManager._round_by_round_no_gen`
对完整 chat template 的逐轮结果比较稳定前缀/后缀，生成 owner 和 query epoch；普通
Hugging Face 模板在有 Drop 时还会渲染带 marker 的完整模板，并依靠 fast tokenizer 的
`offset_mapping` 把完整输出 token 归属回消息
（`python/minisgl/tokenizer/tokenize.py:697-789`，`_merge_owner_track`、
`_round_by_round_no_gen`；`python/minisgl/tokenizer/template_provenance.py:203-239`，
`build_template_token_provenance`）。因此 BOS/EOS、generation prompt、模板分隔符、空消息、
special token 和模板重写都必须从完整模板结果判断，不能用字符查找或重复文本匹配猜边界。

当前未提交 GPT-OSS 原型改走 Harmony token stream，并把 Harmony 注入的 system/developer
前缀标为 owner `-1`，避免 Drop 删除模型元数据；这是 working-tree 原型事实，不等于已
通过端到端验证（`python/minisgl/tokenizer/tokenize.py:62-115`，`_call_chat_template`、
`_get_harmony_encoding`；`python/minisgl/tokenizer/tokenize.py:742-789`，
`_round_by_round_no_gen`）。

## 二、message selector 编译为绝对位置 delta

owner 先被压成每个消息的完整 token 半开区间。每个 Drop event 只收集相对此前新生效的
消息，转换为排序、去重、合并后的绝对 `[start, end)` 区间；event 插入位置来自
`bisect_right(query_epoch, event_n)`。代码拒绝“token 尚未计算就被隐藏”的区间，并同时
生成 `full_token_visible_until`，供可选 full-context mask warmup 使用
（`python/minisgl/tokenizer/tokenize.py:523-625`，`_build_owner_position_ranges`、
`_build_position_drop_plan`）。

最终请求保留两套明确不同的视图：

- `full_input_ids` / `radix_match_ids` 是完整模板 token 轴；
- `input_ids` 是按最终 `keep_mask` 压紧的 active token；
- `true_positions` 保留 active token 在完整轴上的绝对位置；
- `drop_event_positions`、`drop_range_offsets`、`drop_position_ranges` 是一维 wire metadata。

这些字段的组装见 `python/minisgl/tokenizer/tokenize.py:934-965` 和
`python/minisgl/tokenizer/tokenize.py:1020-1044`（`TokenizeManager._chat_tokenize`）。没有
Drop 时 `position_drop_plan` 为空、active 流保持线性，不能改变 mini-sglang 基线行为。

## 三、delta-marker Radix key

Scheduler 收到完整三元 wire metadata 后，`inject_delta_markers` 在每个绝对事件边界插入
一个 virtual key。canonical delta 是绝对 token 半开区间集合；相同 delta 复用同一负数
signed-int64 marker，不同 delta 在该标量处产生不同 Radix 分支。virtual key 没有 KV page，
`virtual_mask=True`、`key_to_token=-1`，真实 token 仍保留双向 key/token 映射
（`python/minisgl/scheduler/radix_delta.py:56-87`，`DeltaMarkerRegistry`；
`python/minisgl/scheduler/radix_delta.py:150-168`，`DeltaRadixLayout`、
`key_prefix_len_for_token_boundary`；`python/minisgl/scheduler/radix_delta.py:205-268`，
`inject_delta_markers`）。这使 token 文本相同但 Drop history 不同的请求不会错误共享状态。

Scheduler 将 layout 写回请求，并在异常、abort 和请求资源回收时释放 request refs
（`python/minisgl/scheduler/scheduler.py:360-413`，`Scheduler._process_one_msg`；
`python/minisgl/scheduler/scheduler.py:415-438`，`Scheduler._process_one_msg`、
`_free_req_resources`）。delta-marker 要求最终 `page_size=1`，且不兼容要求非 unit page 的
TRTLLM（`python/minisgl/scheduler/scheduler.py:54-100`，`Scheduler.__init__`）。

## 四、full match、active KV 与绝对 position

Radix 先在完整 key 轴匹配，再删除 virtual marker 的 `-1` value，得到完整 token 的
`full_match_indices`；普通请求随后按 `prefix_keep_mask` 过滤成
`active_match_indices`。命中率分母是当前可匹配 active prefix，而不是完整 key 数
（`python/minisgl/scheduler/cache.py:103-160`，`CacheManager._radix_query_prefix`、
`matchable_active_prefix_len`、`match_req`；`python/minisgl/scheduler/prefill.py:39-92`，
`PrefillAdder._try_allocate_one`）。调度表只装 active KV，forward 的 position 继续来自
`true_positions`，所以删除历史 token 不会把幸存 token 的位置重新编号。

请求结束提交 delta 分支时，`_cache_finished_delta_req` 用 active KV slot 与绝对 position
重建 full-token prefix；virtual key 的 page value 固定为 `-1`，真实 hole 会截断可提交
前缀，已匹配位置若指向不同 KV slot 会直接报错。未被 Radix 接纳或超出提交边界的 page
随后释放（`python/minisgl/scheduler/cache.py:245-383`，
`CacheManager._cache_finished_delta_req`）。Radix 插入、prune、evict 会增减 tree refs，
`check_integrity` 同时验证 virtual/real page 值、size accounting 和 marker refs，不能通过
放宽 integrity 检查掩盖 page/index 错配
（`python/minisgl/kvcache/radix_cache.py:192-232`，`insert_prefix`；
`python/minisgl/kvcache/radix_cache.py:287-342`，`prune_suffix`、`evict`；
`python/minisgl/kvcache/radix_cache.py:354-386`，`check_integrity`）。

## 五、默认 mask warmup 与可选 staged 回退

有已生效 Drop 的请求默认发送一次 `mask` warmup。前端只设置
`use_context_mask=True`，并携带 tokenizer 生成的完整 token 轴与
`full_token_visible_until`；它不再指定 FlashInfer 或 FlashAttention 的私有格式
（`python/minisgl/server/api_server.py:721-755`，
`FrontendManager.run_contextual_warmup`）。Scheduler 根据实际 Prefill attention backend
调用统一能力校验，不支持的 backend 在启动期失败，不能静默退化为普通 causal attention
（`python/minisgl/scheduler/scheduler.py:104-112`，`Scheduler.__init__`；
`python/minisgl/attention/base.py:296-348`，`BaseAttnBackend`、`HybridBackend`）。

后端从同一可见性语义生成原生执行布局。FlashInfer 将请求编译成 active-KV segments；
FlashAttention 在 SM80/SM90 使用 FA3 segmented adapter，在 SM100/SM110 使用 FA4
`mask_mod` adapter（`python/minisgl/attention/base.py:102-230`，
`build_context_attention_segments`、`build_context_attention_batch`；
`python/minisgl/attention/fi.py:330-409`，`FlashInferBackend.prepare_metadata`；
`python/minisgl/attention/fa.py:56-138`，`validate_fa_context_mask_support`、
`FlashAttentionBackend.validate_context_mask_prefill`；
`python/minisgl/attention/fa.py:195-292`，`FlashAttentionBackend.prepare_metadata`）。
FA4 仍限制每批一个 mask 请求，且尚未启用 GPT-OSS sinks/sliding-window 与 Context mask
的组合。

`staged` 保留为兼容和排障路径。首次 warmup 若 `hit_ratio >= 0.95` 就结束；严格低于
`0.95` 才依次发送 `messages[:end]` 的 prefix warmup
（`python/minisgl/server/api_server.py:753-777`，
`FrontendManager.run_contextual_warmup`）。命中率公式仍是
`cached_active_len / matchable_active_prefix_len`，空分母取 `1.0`
（`python/minisgl/scheduler/prefill.py:39-63`，`PrefillAdder._try_allocate_one`）。每个
staged warmup 的 commit boundary 被映射到包含边界 marker 的 key prefix，避免把未来状态
提前提交（`python/minisgl/tokenizer/tokenize.py:936-945`，`_chat_tokenize`；
`python/minisgl/scheduler/radix_delta.py:159-168`，`key_prefix_len_for_token_boundary`）。

## 六、操作不变量与远端验证

必须保持：无 Drop 基线不变；message id 与模板 owner 一致；absolute position 不压缩；
full/active metadata 不混用；不同 Drop history 不碰撞；request/tree marker refs 与 page 生命周期
闭合；低命中 fallback 的公式、阈值和比较符不漂移。

本地只编辑 `System-test`。完成获批修改并 commit/push 后，才在 `InfiniAI-BUS` 的
`/share/wangruoxi/repo/mini-sglang` 执行 `git pull --ff-only origin System-test`，激活 conda
环境 `rosetta` 并运行 Linux/CUDA 测试；禁止直接编辑远端源码。

## 七、Text/Thinking Drop 与响应命中率

`TextDropRule.drop_messages` 与原始 `messages` 严格等长、同序、同 role；空 selector 是
no-op。非空 `str | list[str]` 使用 UTF-8 Aho-Corasick kernel 找到允许重叠的 occurrence，
再从 raw content span 映射到 canonical rendered char span 和 fast-tokenizer offsets。只删除
完全包含的 token；跨边界 token 保留。selector 并集覆盖完整 content 时提升为完整 owner
ranges，因此与相同目标的 `MessageDropRule` 精确等价。

`ThinkingDropRule` 未开启时不改变 tokenizer。开启后只接受 `reasoning_content` 或唯一 leading
`<think>` block；惰性 probe 判断模型模板是否原生保留历史 thinking。Qwen-like 已知 guard
使用请求级 adapter，未知过滤模板 fail closed；Harmony 直接按 analysis 内容 token 定位。
thinking 的事件在对应 assistant 末尾，保证 final 可见同轮 thinking，而后续 query 不可见。

Scheduler 仍用 `cached_len / matchable_active_prefix_len`（空分母 1.0）计算
`cache_hit_ratio`。该值只随最终生成请求的 terminal reply 传回：非流式位于 OpenAI response
顶层，流式只位于最后一个 finish chunk；内部 warmup ratio 不对用户暴露。
