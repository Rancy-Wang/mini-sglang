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

## SGLang 指标兼容性与 Abort

对照版本：SGLang `03ea13a54557de52da5faab2c422da07c3727407`，
[`bench_serving.py::calculate_metrics`](https://github.com/sgl-project/sglang/blob/03ea13a54557de52da5faab2c422da07c3727407/python/sglang/bench_serving.py)。
输入为逐 HTTP turn 的 prompt_len、success、output_len、生成文本、start_time、
latency、TTFT、ITL，加整段持续时间及 tokenizer。输出为 BenchmarkMetrics 全部数值字段
与逐请求 output_lens。单元测试提取该版本函数，随机输入逐字段比对。

- Prefill = input_throughput = 成功请求的逻辑 prompt token 总数 / 墙钟秒数。
- Decode = output_throughput = 成功请求 usage.completion_tokens 总数 / 墙钟秒数。
- All = total_throughput = 前两者之和。
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
