# 固定并发 BCP throughput

入口：`tests/benchmark/test_throughput.py`。默认 Concurrency=8，task 数为
`min(3*Concurrency,80)`，每个 turn 最多生成 10000 token。task 数必须是偶数、
不少于并发且不超过80。一个 task 是完整 BCP 多轮轨迹，不是一条 HTTP 请求。

## 数据与运行

```bash
PYTHONPATH=python python tests/benchmark/test_throughput.py prepare \
  --source /path/to/evaluation/run --tokenizer /path/to/gpt-oss-120b \
  --output /share/public/wangruoxi/local/throughput_bcp
python tests/benchmark/test_throughput.py run --host 127.0.0.1 --port 8000 \
  --post /v1/chat/completions --model gpt-oss-120b --concurrency 8 \
  --drop --output /path/outside/repo/results
```

`--requests-path` 默认指向上述目录的 `manifest.json`；支持 `--no-drop`、
`--num-requests`、`--max-token-len`、`--tokenizer`、`--timeout`。
HTTP 客户端依赖 aiohttp/numpy/transformers；数据准备还使用 mini-sglang 生产 tokenizer。
POST 采用 OpenAI Chat Completions SSE，工具定义和历史记录来自 manifest。
生成输出完整保存，但下一 turn 继续使用源轨迹中的 assistant/tool 历史，不执行工具。
通过 `--api-key-env ENV_NAME` 读取鉴权，结果不保存密钥。

数据为80个去重的真实 BCP task，40长、40短交替排列。长定义为某个实际生成 turn 的
完整模板输入超过131072 token。数据准备记录源文件 SHA256、逐 turn 消息摘要、完整
token 数、Drop/Reposition 计划，并逐 turn 用生产编译器验证 position 长度。
前12/24/48个分别用于并发4/8/16，保证一半长任务。不能把字符数或 active tokens
当作 full context。同task有多次真实轨迹时，在长/短类别内按source顺序取首次合格轨迹；
优先选择只有长轨迹的task，为短组保留足够不同ID，不按测速表现选轨迹。

Rolling Drop 保留最近12条完整工具响应：TR13后删除TR1，TR14后删除TR2。
Reposition 独立于 Drop 频率：当前 position 长度达到98304后，在消息结束边界压缩，
压缩后重新判断下一次阈值。没有新增可压缩空洞时不重复添加无效 Reposition。
完整消息 token ownership 来自完整 Harmony render；不逐条 tokenize 再拼接。
公开 message ID 到 token 边界由 `tokenizer/tokenize.py::resolve_reposition_token_boundaries`
核对；`rolling_schedule` 与生产 `TokenizeManager.tokenize` 的 next_position 必须一致。

no_drop 不截断历史。prompt 达131072时标记正常的 `context_limit_reached`；prompt尚能
放入而输出预算不足时，把 max_tokens 缩至剩余位置，并记录实际预算。

## 并发、轮次和停止时刻

`Scheduler` 的一个槽持续运行一个 task 的全部 turn。结束后立即补入尚未启动的 task；
尾部从已经完成的 task 中循环选择一个当前未执行的 task，从第一 turn 重放。
同一 case_id 不会同时占两个槽。HTTP 序列化和网络切换会有实际间隙，报告记录
`actual_http_concurrency`、`idle_slot_seconds`，不声称 GPU 永远有 C 个 decode 请求。

每 C 个首遍 task 完成就形成一轮，按完成顺序，无 barrier。补位不增加首遍完成数。
所有首遍完成的时刻立刻作为 cutoff，取消其他槽；成功完成的补位 turn 计入系统工作量，
截点尚未完成的 HTTP turn 单独保存，不把它已生成的部分补入成功吞吐。
已成功结束的此前 turn 不会因同一 task 后续中止而撤销。

每轮连续时间窗口内结束的 HTTP turn 归该轮，每个 turn 仅一次，跨窗口 turn 单独计数。
同时给出累计、首遍、补位指标，本轮 C 个 task 全生命周期工作量及相邻轮次变化率。
不足C个的最后一组不形成轮次，但进入 overall。每轮指标在后台写入 `.round-N.json`，
最终文件整合 `overall/rounds/tasks/turns/excluded_at_cutoff`；事件流水逐条持久化。

## 实际模型前向吞吐（正式报告口径）

正式结果读取 `compute_metrics`。`metrics`/`sglang_logical_metrics` 仅保留 SGLang
兼容性，**不能再把其中 input_throughput 标成实际 Prefill throughput**。

- Prefill token 数：逐次模型前向之前的真实请求 `Req.extend_len` 累计。
  Chunked Prefill 每个 chunk 分别累计；不计 cache hit、drop skip、仅做 RoPE
  变换的 Reposition token，也不计 CUDA Graph 的 padding 行。
- Decode token 数：真实 Decode batch 的请求输入 token 累计。Prefill 已产生首个
  输出 token，不再把它当一次 Decode。计数包含结束信号确认前已发起的 overlap
  Decode；它与 `usage.completion_tokens`（采样输出数）不相同。
- Prefill/Decode/All throughput 分别为上述两个计数及其和，除以同一个测量窗口的
  墙钟时间。不是 GPU 忙碌时间吞吐，也不是 FLOPS。
- 服务端 `server_metrics.prefill_compute_tokens/decode_compute_tokens` 返回直接
  计数。采样位置在 `scheduler/scheduler.py::Scheduler._forward` 的
  `engine.forward_batch` 周围；必须在 `Req.complete_one()` 改变长度前取样。
  `message/metrics.py::RequestMetricsState.finish` 将计数带入最终 SSE/JSON。
- 当前支持请求内的 mask / paged-occurrence 路径。缺计数或多次内部 staged 请求
  没有完整聚合时返回 null，不能将缺失当作零或退回逻辑 prompt 数。

成功请求采用 strict_success（完整 usage/DONE/正常结束且无错误）；失败和 Abort
的部分工作排除，分母仍包含其占用时间。这是成功请求的实际前向 token 吞吐，不是
包含所有失败计算、padding、RoPE 操作的 GPU 总工作量。SGLang 适配器的宽松成功
行为仍在兼容列中保留。成功的补位 turn 正常计入，cutoff 未完成 turn 排除。

### 旧实验离线重算

入口 `tests/benchmark/throughput_compute.py`，输出 `.compute.json`，保留原始 JSON
及 SHA256，记录 engine_head 和 analysis_head。整体、逐轮、累计、首遍及补位均重算。

```bash
# 仅 CPU 分词；PYTHONPATH 指向当时运行的引擎 checkout。
PYTHONPATH=/path/to/old-engine/python python tests/benchmark/throughput_compute.py evidence \
  --manifest /path/manifest.json --engine-repo /path/to/old-engine \
  --launch /path/server/launch.json --output /path/compute-evidence.json
python tests/benchmark/throughput_compute.py matrix \
  --root /path/matrix-v1 --evidence /path/compute-evidence.json
```

旧版本 f089bf6 的 mask、paged-occurrence、page-size=1，普通 eviction 或整场 no_drop，
且终场两 rank 审计通过时，可以还原 Prefill：三类 usage 互斥，先求
`M = cached_tokens + drop_skipped_tokens + repos_tokens`。
普通/full-mask/paged-occurrence 的 Extend 总数为 `prompt_tokens - M`。
mask-free 路径通过原轨迹生产 tokenizer 的 Drop 边界重新判断 planner 分支，另外扣除
尚未命中但已不必计算的死 token。不能对任意系统或存在孔洞恢复的运行机械套用相减。
特别注意：Drop-aware + drop 在 cache 写入时就可能插入 -1 空洞，并不增加
`drop_pages`。即便 drop_pages/hole_fills 全部为零，也不能把 resident token 数当成
logical matched prefix 长度。旧响应缺这个长度，因而该组合的 Prefill/All 必须是
unknown/null；单凭相减得到的数字不能当实际计算量。
证据不满足时输出 unknown/null；审计和元数据不能证明未记录的工作量。

旧服务没记录 overlap 前向次数。若成功输出 G 个 token，已确认 Decode 至少 G-1，
至多 G（多一次尚未确认终止的前向）；达到请求输出上限时没有额外预算，区间收窄。
因此旧报告在可恢复时给精确 Prefill 和 Decode/All 上下界；DA+drop 只给 Decode
上下界，Prefill/All 留空，缺精确值的字段为 null。
`generated_output_throughput` 另存 usage 输出吞吐，不冒充实际 Decode 次数。
离线重算本身不执行推理，不通过估计填造精确值。逐轮按完成窗口归属整条 HTTP turn，跨轮的 GPU
计算时间无法从旧终场信息重新切分，因此它是完成归属的轮次吞吐，不是 GPU 时间切片。

若最终验收要求全部精确，而旧记录无法恢复计数，应使用有直接前向计数的版本重测
受影响项，保留旧记录并使用新的输出目录。2026-09-16 用户已明确授权此类重测，
替代此前“已完成实验不重跑”的限制：旧1–7项需要重测，8–10项首次运行，
统一写入 `matrix-exact-v3`。旧 HEAD 和原始数据保留作历史证据，不覆盖或拼接成新结果。

新矩阵每项初始审计检查实际计数字段，缺失则拒绝正式测速。结束时
`throughput_compute.require_exact_result` 从保存的终场服务端计数重新汇总，检查
整体、每轮、累计、首遍及补位的 token 数和吞吐公式、轮数和时间窗口。
只有 `valid=true`、所有窗口 `exact=true`、计数与复算一致且两 rank 终场页面
审计通过，启动器才将该项标为 completed。范围、缺失值和逻辑吞吐不能通过验收。

## SGLang 指标兼容性与 Abort

对照版本：SGLang `03ea13a54557de52da5faab2c422da07c3727407`，
[`bench_serving.py::calculate_metrics`](https://github.com/sgl-project/sglang/blob/03ea13a54557de52da5faab2c422da07c3727407/python/sglang/bench_serving.py)。
输入为逐 HTTP turn 的 prompt_len、success、output_len、生成文本、start_time、
latency、TTFT、ITL，加整段持续时间及 tokenizer。输出为 BenchmarkMetrics 全部数值字段
与逐请求 output_lens。单元测试提取该版本函数，随机输入逐字段比对。

- 逻辑输入 input_throughput = 成功请求的逻辑 prompt token 总数 / 墙钟秒数。
- 输出 output_throughput = 成功请求 usage.completion_tokens 总数 / 墙钟秒数。
- 逻辑总量 total_throughput = 前两者之和。
- output_throughput_retokenized 使用生成文本重新 tokenize；同时提供对应 All。
- TPOT = (latency - TTFT)/(output_len-1)，output_len<=1不进入 TPOT 分布。
- TTFT/TPOT/ITL/E2E 提供 mean/median/std/p90/p95/p99，ITL还含max。
- concurrency = 成功 HTTP 请求 latency 总和/墙钟时间；与配置的对话并发不同。
- peak 与 upstream 一样采用1秒桶、非空文本 SSE事件；不是精确 GPU token 时间。

失败请求在 SGLang calculate_metrics 中不累计输入输出，output_lens=0。但其 OAI-chat
适配器把 HTTP200 的正常 EOF 视为 success，即使最后事件有 error/Abort。因此同时保留：
`metrics` 复现这一适配器；`strict_metrics` 还要求 DONE、完整usage、正常finish_reason，
并排除显式error。缺usage时兼容列保留upstream的请求max_tokens回退，但整组标记invalid，
不能当成有效测速结果。全失败时返回null E2E并标记invalid，避免upstream空percentile异常。

计数使用 full prompt，并单独核对服务端 usage；工具调用参数事件完整保存，但与上述
SGLang adapter一样不并入其 generated_text。这意味着重分词数可以小于completion_tokens。
没有speculative accept_length配置，ITL不做猜测性的token均分。

Drop 请求通过返回的 `drop_skipped_tokens` / `repos_tokens` 扩展字段确认协议支持；
整组有 Drop 却没有任何确认时标为 invalid。单 turn 未返回这些字段时保存为未知，
不会把冷缓存的零节省判成 Drop 失败。这是协议确认，不是逐 token 的 attention mask
正确性证明；机制正确性仍依赖系统测试。完整 usage 和错误信息均保留供复核。

这些是**请求级、端到端逻辑吞吐**：缓存命中、Drop跳过也在逻辑prompt中；不能将
Prefill值解释为实际执行的GPU prefill tokens/s。缓存与服务端metadata原样保留在事件中。
只凭最终 response.json 无法恢复流式TTFT/ITL；客户端必须对SSE到达时间计时。
窗口边界有跨轮请求时，兼容concurrency/peak仍覆盖这些请求的完整生命周期，不能当成
严格裁剪后的窗口内GPU并发。判断变慢需结合任务长度、输出长度、缓存和补位比例。

## 验证与实验驱动

```bash
SGLANG_REFERENCE=/path/to/pinned/bench_serving.py \
  python -m unittest discover -s tests/benchmark -p test_throughput_unit.py -v
python scripts/run_throughput_matrix.py --model /path/to/gpt-oss-120b \
  --output /path/outside/repo/smoke --smoke
python scripts/run_throughput_matrix.py --model /path/to/gpt-oss-120b \
  --output /path/outside/repo/matrix --pages VERIFIED_EQUAL_CAPACITY
```

smoke为C2/N4/2turn/16tokens及一组末轮长上下文请求，不是正式数据。
正式两服务分别使用GPU0,1和2,3、独立端口/cache/log。TP2、CUDA Graph最大batch16、
上下文131072、prefill chunk16384；两者只切换drop-aware eviction。
每项前后在系统空闲时检查page所有权、引用归零并排空缓存；检查不进入测速。
模型初始化和编译不进入计时。同一服务复用模型，每项从相同空缓存开始。
`status.json` 支持同HEAD和输入hash下跳过成功项；错误不自动算成功，保留日志诊断。
可用 `--drop-aware-gpus 2,3 --ordinary-gpus 0,1` 调换两组物理卡；实际映射写入报告。
某组卡被占用时，该服务排队等待，另一组独立执行；不会终止其他任务。
每项开始前核对代码 HEAD，避免同一矩阵混入不同代码版本。
