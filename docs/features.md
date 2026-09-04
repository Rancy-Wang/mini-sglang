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

`prompt_tokens` 始终是完整 chat-template prompt 的 token 数；`cached_tokens` 是本次
contextual prefill 真正复用的 KV token 数；`drop_skipped_tokens` 是因为等价的
mask-free Drop 而没有进入 Prefill Attention 的完整 prompt token 数。只有 mask-free
路径会报告非零 `drop_skipped_tokens`；mask fallback、staged 和无 Drop 请求均报告零。
当两个明细值都为零时，响应省略 `prompt_tokens_details`。流式请求需要设置
`"stream_options":{"include_usage":true}`，服务器会在 `[DONE]` 前发送一个
`choices: []` 的最终 usage chunk。旧的顶层 `cache_hit_ratio` 不再返回。

实现依据：tokenizer 将规则编译为按位置排序的稀疏 event/range wire，并在边界合并不
可判定时标记回退（`python/minisgl/tokenizer/tokenize.py:1004`）；CPU AOT kernel 只
检查未缓存 query 与有效 Drop 区间是否冲突
（`python/minisgl/kernel/csrc/src/context_plan.cpp:21`）；调度器的快速路径与 O(N)
精确参考回退位于 `python/minisgl/scheduler/prefill.py:68`，usage 的边界检查与结构位于
`python/minisgl/server/api_server.py:405`。

## Overlap Scheduling

To further reduce CPU overhead, Mini-SGLang employs overlap scheduling, a technique proposed in [NanoFlow](https://arxiv.org/abs/2408.12757). This approach overlaps the CPU scheduling overhead with GPU computation, improving overall system throughput.

![overlap](https://lmsys.org/images/blog/sglang_v0_4/scheduler.jpg)
*Illustration of Overlap Scheduling from [LMSYS Blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/).*
