# Features of Mini-SGLang

## Online Serving

Mini-SGLang supports online serving with an OpenAI-compatible API server. It provides the standard `/v1/chat/completions` endpoint, allowing seamless integration with existing tools and clients. For detailed command-line arguments and configuration options, run `python -m minisgl --help`.

## Interactive Shell Mode

For demonstration and testing purposes, an interactive shell mode is available. In this mode, users can input prompts directly, and the LLM will generate responses in real-time. The shell automatically caches chat history to maintain context. To clear the conversation history and start a new session, use the `/reset` command.

Example:

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

## Distributed Serving

To scale performance across multiple GPUs, Mini-SGLang supports Tensor Parallelism (TP). You can enable distributed serving by specifying the number of GPUs with the `--tp n` argument, where `n` is the degree of parallelism.

## Supported Models

Our framework currently supports the following dense model architectures:

- [`Llama-3`](https://huggingface.co/collections/meta-llama/llama-31) series
- [`Qwen-3`](https://huggingface.co/collections/Qwen/qwen3) series (including MoE)
- [`Qwen-2.5`](https://huggingface.co/collections/Qwen/qwen25) series

## Chunked Prefill

Chunked Prefill, a technique introduced by [Sarathi-Serve](https://arxiv.org/abs/2403.02310), is enabled by default. This feature splits long prompts into smaller chunks during the prefill phase, significantly reducing peak memory usage and preventing Out-Of-Memory (OOM) errors in long-context serving. The chunk size can be configured using `--max-prefill-length n`. Note that setting `n` to a very small value (e.g., 128) is not recommended as it may significantly degrade performance.

## Page Size

You can specify the page size of the system using the `--page-size` argument.

## Attention Backends

Mini-SGLang integrates high-performance attention kernels, including [`FlashAttention`](https://github.com/Dao-AILab/flash-attention) (`fa`), [`FlashInfer`](https://github.com/flashinfer-ai/flashinfer) (`fi`) and [`TensorRT-LLM fmha`](https://github.com/NVIDIA/TensorRT-LLM) (`trtllm`). It supports using different backends for the prefill and decode phases to maximize efficiency. For example, on NVIDIA Hopper GPUs, `FlashAttention 3` is used for prefill and `FlashInfer` for decode by default.

You can specify the backend using the `--attn` argument. If two values are provided (e.g., `--attn fa,fi`), the first specifies the prefill backend and the second the decode backend. Note that some attention backend might override the user-provided page size (e.g. `trtllm` only supports page size 16,32,64).

## CUDA Graph

To minimize CPU launch overhead during decoding, Mini-SGLang supports capturing and replaying CUDA graphs. This feature is enabled by default. The maximum batch size for CUDA graph capture can be set with `--cuda-graph-max-bs n`. Setting `n` to `0` disables this feature.

## Radix Cache

Adopting the original design from [SGLang](https://github.com/sgl-project/sglang.git), Mini-SGLang implements a Radix Cache to manage the Key-Value (KV) cache. This allows the reuse of KV cache for shared prefixes across requests, reducing redundant computation. This feature is enabled by default but can be switched to a naive cache management strategy using `--cache naive`.

![radix](https://lmsys.org/images/blog/sglang/radix_attn.jpg)
*Illustration of Radix Attention from [LMSYS Blog](https://lmsys.org/blog/2024-01-17-sglang/).*

## Keep-text Drop Rule

`keep_text_drop` lets a stateless chat request expose only the ordered text that should remain
visible while supplying the complete history used for Radix matching:

```json
{
  "messages": [
    {"role": "user", "content": "multiply it by 3"}
  ],
  "drop_rule": {
    "type": "keep_text_drop",
    "full_messages": [
      {"role": "user", "content": "What is 15 + 27?"},
      {"role": "assistant", "content": "15 + 27 = 42."},
      {"role": "user", "content": "Then multiply it by 3."}
    ],
    "force": false
  }
}
```

Visible messages are matched in order from right to left, so repeated text selects the latest
compatible messages by default. Role and tool-call protocol metadata must also match. A selected
substring keeps every overlapping token, including tokens cut by either substring boundary, and
keeps that message's chat-template wrapper tokens. Unselected messages are dropped completely.

If projection fails, the default is an HTTP 400 response. Setting `force` to `true` instead runs a
normal inference using the outer `messages` as the complete prompt and does not reuse the supplied
hidden history.

## Contextual Prefill Usage

默认的 `mask` contextual prefill 会先用完整 Radix key 做一次匹配。调度器随后用
tokenizer 产生的稀疏 Drop event/range 元数据判断 compact causal Extend 是否与精确
Context mask 等价：等价时直接执行 mask-free Extend，不等价、元数据异常或模型使用
sliding window 时保守回退到原始 mask Prefill。启动时添加
`--disable-mask-free-context-prefill` 可以强制回退，供算法对照实验使用。

非流式 OpenAI chat-completions 响应使用 SGLang 风格的 usage：

```json
{
  "usage": {
    "prompt_tokens": 82000,
    "completion_tokens": 96,
    "total_tokens": 82096,
    "prompt_tokens_details": {
      "cached_tokens": 50000,
      "drop_skipped_tokens": 31000
    }
  }
}
```

`prompt_tokens` 始终是完整 chat-template prompt 的 token 数；`completion_tokens` 是输出
计数，`total_tokens` 为两者之和。缓存明细必须整体来自同一阶段：普通 Drop 请求有独立
warmup 时采用该 warmup 的报告（legacy staged fallback 也保留首次探测的完整报告）；
没有 warmup 时采用生成请求的报告。Reposition 采用最后一个 scheduler stage 的报告，
不累计前置 stage，也不从累计性能指标中拼接 Drop 数。

Reposition 请求的 `prompt_tokens_details` 包含三项，即使全部为零也返回：

```json
{"cached_tokens": 4, "drop_skipped_tokens": 2, "repos_tokens": 2}
```

- `cached_tokens`：该 stage 从 Radix 取得、实际参与 Prefill/Extend attention、且本
  stage 没有通过 Retry 做 RoPE 转换的不同 token 数。
- `repos_tokens`：该 stage 通过 Retry 成功做 RoPE 转换，并实际参与 attention 复用的
  不同 token 数。不能用 Retry plan 长度或累计转换次数替代，因为转换可能涉及本 stage
  不使用的页。
- `drop_skipped_tokens`：该 stage 从 Radix 匹配到，但因 Drop 完全没有参与该 stage
  attention 的不同 token 数。只包含已匹配的物理缓存 token，不包含未命中输入或虚拟标记。

设 R 为该阶段匹配的物理缓存 token 集合，U 为其中实际参与 attention 的集合，T 为
其中经过本阶段 Retry RoPE 的集合。三项分别为 `|U \ T|`、`|R \ U|`、`|U ∩ T|`，
互斥且总和为 `|R| <= prompt_tokens`。普通请求没有 `repos_tokens` 字段；两个普通
明细均为零时，仍省略 `prompt_tokens_details`。

mask Prefill 和 mask-free Extend 均可有非零 `drop_skipped_tokens`。一个缓存 token
先被本 stage 的 query 使用、后来才被 Drop，仍计入复用：普通 KV 归 `cached_tokens`，
本 stage Retry RoPE 的 KV 归 `repos_tokens`。例如 Drop 在 query 49 前隐藏 token 0–24：
已缓存 49 个 token 时，这 25 个 token 全程跳过；只缓存 46 个时，query 46–48 仍需使用
它们，因此不能计为跳过。计数在最终 KV 压缩前固定，不随 decode、层数或 segment 数累加。
这里按完整 attention 的可见性计数，不把 sliding-window 层的窗口裁剪视为 Drop。

最后一个 Reposition stage 可以复用同一 HTTP 请求前置 stage 写入的 KV；这些 usage
不是整个 HTTP 请求开始前已有缓存的命中量，也不是累计计算成本。性能指标中的转换
操作量、耗时和传输字节继续按原来的累计口径记录。

流式请求设置 `"stream_options":{"include_usage":true}` 后，会在 `[DONE]` 前收到
`choices: []` 的最终 usage chunk，其口径与非流式响应相同。

实现依据：`Req.record_context_cache_usage` 和 `reported_cached_tokens` 分别固定实际复用
集合的分类及返回普通复用数（`python/minisgl/core.py`）；
`build_context_attention_batch` 从实际 full-attention segment 提供缓存位置
（`python/minisgl/attention/base.py`）；`PrefillAdder` 保存 Retry 的转换标记并传递分块
计数（`python/minisgl/scheduler/prefill.py`）；`CacheUsageReport.from_reply` 与
`_build_usage` 在 HTTP 边界保留完整报告、校验三项总和
（`python/minisgl/server/api_server.py`）。

## Overlap Scheduling

To further reduce CPU overhead, Mini-SGLang employs overlap scheduling, a technique proposed in [NanoFlow](https://arxiv.org/abs/2408.12757). This approach overlaps the CPU scheduling overhead with GPU computation, improving overall system throughput.

![overlap](https://lmsys.org/images/blog/sglang_v0_4/scheduler.jpg)
*Illustration of Overlap Scheduling from [LMSYS Blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/).*
