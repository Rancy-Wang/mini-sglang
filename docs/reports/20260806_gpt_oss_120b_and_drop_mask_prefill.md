# 修改报告 2：GPT-OSS 120B 与 Drop Mask Prefill（2026-08-06）

## mini-sglang 如何支持 GPT-OSS 120B

原版 mini-sglang 只有 Llama/Qwen/Mistral 类模型的通用链路，不能直接加载 GPT-OSS：缺少 `GptOssForCausalLM` 注册与专用配置字段、packed MXFP4 MoE 算子及权重分片、交替 full/sliding attention 与 attention sinks、Harmony 对话/工具协议，以及相应的运行约束和依赖 ABI。

本次补齐了以下链路：

- 配置与模型：解析 expert、`layer_types`、sliding window 等字段，注册 GPT-OSS，并实现 alternating attention、sink bias 和 sparse MoE（[`config.py`](../../python/minisgl/models/config.py#L49)、[`gpt_oss.py`](../../python/minisgl/models/gpt_oss.py#L28)、[`register.py`](../../python/minisgl/models/register.py#L5)）。
- MXFP4 与权重：新增兼容 bundled `triton_kernels` ABI 的 packed expert runtime、布局转换、routing/SwiGLU/two-stage matmul，以及 packed-byte TP 切片和官方权重名映射（[`mxfp4.py`](../../python/minisgl/moe/mxfp4.py#L11)、[`weight.py`](../../python/minisgl/models/weight.py#L44)）。
- 协议与服务：用 `openai-harmony` 渲染 system/developer/user/assistant/tool 消息，解析 analysis/final/tool recipient，并接入流式与非流式响应（[`tokenize.py`](../../python/minisgl/tokenizer/tokenize.py#L106)、[`api_server.py`](../../python/minisgl/server/api_server.py#L511)）。
- 120B 约束：当前后端要求 BF16；128 experts 的 120B 配置要求 TP8，暂不支持 EP（[`engine.py`](../../python/minisgl/engine/engine.py#L48)）。依赖版本固定在 [`pyproject.toml`](../../pyproject.toml#L24)，避免 FlashInfer、CUTLASS DSL 与 Triton kernel ABI 漂移。

## Batch 内多请求的 Drop Mask Prefill

Drop 后每个请求可能拥有不同的可见 KV 集合。后端先把每个请求的可见性编译成保持绝对位置的 causal segments，再聚合为 `ContextAttentionBatch`；Triton `_compile_context_page_tables_kernel` 一次并行编译全部 segment 的 page table、query offset 和有效长度，避免 Python 按 segment 逐段构造带来的线性调度开销（[`base.py`](../../python/minisgl/attention/base.py#L30)、[`base.py`](../../python/minisgl/attention/base.py#L60)）。调度器只把连续的 masked requests 合并，并以 backend capability gate 隔离不支持该模式的路径（[`prefill.py`](../../python/minisgl/scheduler/prefill.py#L23)、[`prefill.py`](../../python/minisgl/scheduler/prefill.py#L242)）。FA3 聚合后只调用一次 context-mask kernel（[`fa.py`](../../python/minisgl/attention/fa.py#L347)）；FlashInfer 同样对整批只做一次 plan/run（[`fi.py`](../../python/minisgl/attention/fi.py#L144)）。

远端 A800 验证中，2 个请求、10 个 query 的 full/sliding 用例分别生成 5/10 个 segments，FA3 均为 1 次调用，FlashInfer 均为 1 次 plan + 1 次 run；65 个相关回归测试通过。8/32/128 segments 的编译中位数为 0.112/0.143/0.154 ms，其中 GPU kernel 为 0.026/0.032/0.034 ms。

## 后端 token 粒度 Drop

外部 API 仍按 message 指定 Drop；tokenizer 用 provenance 把 message owner 编译成**绝对 token-position 半开区间**，并生成每个 token 的 `visible_until`。调度阶段把规范化区间 intern 为负数虚拟 delta marker：marker 参与 Radix key 匹配但 page value 为 `-1`，因此不占 KV；同一 token 历史复用同一分支，不同 Drop 历史可靠分叉（[`radix_delta.py`](../../python/minisgl/scheduler/radix_delta.py#L12)、[`radix_cache.py`](../../python/minisgl/kvcache/radix_cache.py#L18)）。Prefill 再按 token 可见性构造 active-KV segments，保留原始 absolute positions；被 Drop 的 token 仍可留在 full-token/Radix 状态中供其他历史复用，但不会进入当前请求的 attention page table。区间编译入口见 [`tokenize.py`](../../python/minisgl/tokenizer/tokenize.py#L527) 和 [`tokenize.py`](../../python/minisgl/tokenizer/tokenize.py#L576)。
