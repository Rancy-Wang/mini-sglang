# 多轮 HTTP serving 测评

本实现对应 `PLAN-CS-20260917-R3`。客户端发送 HTTP 请求；可选的本地模板适配器只负责长度和 Drop 消息边界。`test_throughput.py` 的旧请求构造及旧矩阵不参与新流程，仅复用固定版本 SGLang 的统计函数和 SSE 帧解析器。SGLang 参考版本为 `03ea13a54557de52da5faab2c422da07c3727407`。

## 数据与历史

`tests/benchmark/test_serving.py::prepare` 联合 registry 的 `full_context/{trajectories,records,sources}.jsonl`，按 `(case_id,trial)` 对齐。合格记录必须是 `agent_stop` 或 `token_limit_answer`，agent 调用与 assistant 消息一一对应，全部 context 严格为 `{"policy":"full_context"}`，结构化 Drop/Reposition 及其计数为空，每轮 `completion_tokens` 为正整数。答案正确性不影响可用性。

输出目录默认 `/mnt/public/wangruoxi/local/throughput_bcp_full_context`，必须为新目录：

- `trajectories.jsonl`：全部合格 trial；每行包含完整原始 trajectory、完整 source record、source provenance、每轮新增输入及原始输出长度。
- `tasks_unique.jsonl`：每个 case 选择最早合格 trial；包含 UTF-8 字节 offset、长度、SHA256，读取时验证身份与哈希。
- `excluded.jsonl`：排除原因。
- `manifest.json`：源文件和输出文件哈希、工具定义、数量、逐轨迹概况。
- `README.md`：查阅说明。

原始历史不裁剪、不覆盖。无权限的原始 HTTP 文件不属于已核实证据。加载时先固定种子打乱去重列表，再选择前 N 个；各并发度使用嵌套前缀。

每轮发送新增源输入和累计历史；长度使用该轮源 `completion_tokens`。参数 `ignore_eos=true`，温度 0，不发送 stop 字符串。GPT-OSS 的自动 Harmony 终止 token 也必须遵守该开关：tokenizer 在普通和 Context 路径均不为 `ignore_eos=true` 追加自动 stop 序列；用户显式提供的 stop 字符串仍生效。修复前 GPU 对照发现开关两种取值都在 20 tokens 提前停止，不能仅凭 API 接收参数断言支持。响应中的实际 reasoning/content/tool_calls 增量合并成 assistant 消息后才进入下一轮。源 assistant 输出仅用于保留来源，不用于后续请求。源 tool response 的名称由源 tool-call ID 恢复，所以不会依赖新生成的 tool-call ID。

这种 message 重建有损，未验证原始 sampled token 能全部往返；它模拟用户提交行为。`server_metrics.generated_tokens` 是长度检查的优先依据，因为隐藏的终止 token 不一定计入 API 的可见 completion_tokens；无服务扩展时退回 usage。任何少生成、多生成、缺 usage/DONE、流内 error、异常终止均不算成功。服务端可能按上下文余量裁剪长度，客户端将其视为失败，不主动缩小 max_tokens。

## Drop 与位置

`RollingState.extend` 在实际累积的历史上，只追加事件。按 task 独立数 tool response：TR13 Drop TR1，保留最新 12 条。Drop 完整工具结果，不删除 assistant reasoning/tool call。

`TemplateAdapter.render` 的 minisgl-harmony 适配器调用生产模板的完整渲染和 message owner 映射；不是逐消息独立 tokenize 后相加。每次核对旧消息哈希和旧边界，防止改写已提交历史。Drop 后幸存 token 的位置不重排；只有当前绝对位置达到 98,304，且存在新空洞时，在合法消息末尾追加 Reposition。Reposition 后以压缩位置继续计数。generation suffix 没有公开 message owner，不凭空创建事件。后续请求携带全部旧事件及新增事件。

本改动没有修改 Drop/Radix key、full/active KV、page ownership 或调度算法。

## 并发与结束

`Scheduler` 维护 C 个不同 case 的活跃 task。task 内所有 turn 串行，完整结束才替换。未开始 task 优先；耗尽后用已终止且当前不活跃的 task 从头填充尾部，标记 filler。失败也是终态，但不是成功，且记录具体状态；不无限重试。

全部首次执行 task 终止的瞬间固定 cutoff，停止补发并取消其他 worker。截止前成功完成的 filler turn 进入总体吞吐；截止取消及未完成部分不进入成功分子，时间仍计入墙钟。每 C 个首次执行 task 终止产生一轮，没有 barrier；不满 C 的末轮只进入整体。

客户端保证活跃 task 槽位，不保证每个瞬间 GPU batch 都是 C。记录实际 HTTP 平均并发和组装时间。逐轮 window 把完整请求归到完成窗口；它不是精确按 GPU 时间分割的 token 工作量。另有 cohort 生命周期、累计、first_pass 和 filler 视图。

## 时间与计算指标

`python/minisgl/message/metrics.py::RequestMetricsState.observe_token` 可选记录相邻采样 token 的观察时间差；设置 `MINISGL_RECORD_TOKEN_TIMINGS=1` 启用。默认关闭。`ServerMetrics` 验证数量、单调性和时长，终止响应输出 `token_intervals_ns`。没有增加 CUDA 同步，观察点沿用 scheduler 完成 token 拷贝后的单调时钟；不是 GPU 硬件时间戳。

客户端与服务端时钟从不相减：

- TTFT：客户端发送到首个有效内容/推理/tool-call delta。
- E2E：客户端发送到流结束。
- 标准 TPOT：`(E2E-TTFT)/(completion_tokens-1)`，遵循固定 SGLang 公式。
- 服务端 TBT：相邻 sampled token 间隔，包含首个 decode 间隔和隐藏 token。
- 自定义 TPOT：去掉第一个服务端 TBT 后的均值；不足三 token 时不可用。
- first_decode_gap：单独保存第一个 TBT；包含等待及 decode 等开销，不能据此精确隔离其他 prefill 的等待。
- SSE chunk 间隔另存 `itl/chunk_times`；流可能合并或隐藏 token，不能冒充 TBT。

成功请求的实际 prefill/decode forward 计数除以测量时长，是主要吞吐。cache hit 不算新 forward，CUDA Graph padding 不算请求 token，RoPE 搬移不算 model forward，首个输出通常属于 prefill。多阶段或缺计数返回 unknown，不能用逻辑输入长度代替。失败的部分工作不计成功分子，因此这不是设备执行的全部工作量。SGLang 逻辑输入/输出/总吞吐、mean/median/std/P90/P95/P99 延迟及峰值字段另存；缺 tokenizer 时 retokenized 指标为 null。

失败/abort/timeout/cutoff_cancelled 保存记录，只有严格成功进入吞吐/延迟统计。所有请求有 TBT 字段；失败中断可能取不到服务端完整 trace，此时明确 unavailable/incomplete。无成功样本的实际指标为 null。

## SLO

基线是实际 A800 TP2；每种配置各跑 96 个 task，C=1，无 filler，每个 task 前用经过空闲 page 检查的清理钩子隔离缓存，task 内正常保留。模型预热和清理不进入请求延迟；基线整段 wall time 包含清理，不能拿它的总体 throughput 当作正式 C=1 结果。

按配置签名和 case/trial/turn 匹配，TTFT/标准 TPOT/E2E 的 slowdown 是 loaded/baseline；TBT 按 token 间隔序号匹配。分母为零、trace 缺失或长度不符记不可用。历史重建可能不同，另存 prompt 长度差，不声称完全同一输入。

阈值 P50/P90/P99：TTFT 为 2/3/6 倍，TPOT、TBT、E2E 为 1.25/1.5/5 倍。SLO 只以首次执行请求判定。TBT 主分布每个请求总权重为 1，使用加权经验 CDF 分位数；另外保存 token 加权分位数。全部 12 项通过且覆盖完整才 pass；首次执行失败不能获得通过。小样本 P99 不代表稳定的总体尾延迟。

## 命令与分阶段门禁

在已同步的 Zhangyudong-BUS 仓库运行，使用能导入 minisgl、aiohttp、numpy、transformers 和 openai_harmony 的环境：

```bash
PYTHONPATH=python python tests/benchmark/test_serving.py prepare
PYTHONPATH=python python scripts/run_serving_matrix.py --phase smoke --output-dir /mnt/public/wangruoxi/local/serving_smoke_UNIQUE
```

smoke 单独占 GPU0,1，完成 EOS 对照、故意拒绝、合成 96K Drop/Reposition 增量历史和一个完整真实 source trajectory。`--smoke` 只是标签，不截断源 turn，不减少源输出长度。合成数据单独标记，不混入 full-context 数据集。

用户已在批准 `PLAN-CS-20260917-R4` 时授权：修复后完整 smoke 通过即可直接执行基线和正式矩阵，无需再次审批：

```bash
PYTHONPATH=python python scripts/run_serving_matrix.py --phase baseline --output-dir /mnt/public/wangruoxi/local/serving_baseline_UNIQUE
PYTHONPATH=python python scripts/run_serving_matrix.py --phase matrix --baseline-root /mnt/public/wangruoxi/local/serving_baseline_UNIQUE --output-dir /mnt/public/wangruoxi/local/serving_matrix_UNIQUE
```

两套服务分别为 GPU0,1 drop-aware+drop，GPU2,3 ordinary+no_drop；C=1/2/4/8/16/32，N=3C。端口默认31080/31081；CUDA Graph和容量覆盖32。相同 KV pages 是启动门槛，不一致则要求显式公共 `--pages`。每对完成写进度；任务监控每半小时检查状态，每对完成汇报，阶段失败立即保留证据。脚本不会杀其他用户进程，只停止自己的子进程组。

各阶段独立调用，没有 smoke 自动进入 baseline/matrix 的路径。恢复只跳过代码版本、数据哈希、模式、模型、容量参数、seed、基线哈希相同且结果文件哈希/valid 均通过的单元；失败的单元重新从头测量，不拼接跨进程时钟。

直接使用其他服务：

```bash
python tests/benchmark/test_serving.py run --host http://example --port 8000 --post-path /v1/chat/completions --concurrency 8 --num-tasks 40 --no-drop --requests-path /path/tasks_unique.jsonl --output-dir /path/results
```

其他服务需接受对应 ignore_eos 请求参数，并支持流式 usage。无法提供 sampled-token TBT 或实际 forward 计数时保留 null。Drop 目前是明确的 minisgl-harmony 协议适配器；不能把其特有字段直接假定为所有服务支持。

## 验证

```bash
python -m unittest discover -s tests/benchmark -p 'test_serving_unit.py'
python -m unittest discover -s tests/scripts -p 'test_run_serving_matrix.py'
PYTHONPATH=python python -m pytest -q tests/server/test_serving_metrics.py
```

CPU 测试覆盖 source 对齐、增量历史、滚动窗口/当前位置、不可变前缀、SSE fragments、长度不足、错误/EOF、TBT边界、不同task补齐及截止、成功分子和 SLO。远端服务测试覆盖新增字段的 IPC 往返，GPU smoke 检查实际 ignore_eos 与生产协议。正式吞吐曲线未运行前，不输出性能或 SLO 通过结论。

## 当前代码证据索引

以下起始行对应本 checkpoint 的工作树，符号是跨版本定位依据：

| 事实 | 文件与起始行、符号 |
|---|---|
| 源 history 与新输入切片 | `tests/benchmark/test_serving.py:52`，`compile_case` |
| append-only Drop/位置检查 | `tests/benchmark/test_serving.py:167`，`RollingState` |
| 完整模板适配 | `tests/benchmark/test_serving.py:210`，`TemplateAdapter` |
| 流、长度及 TBT | `tests/benchmark/test_serving.py:273`，`request` |
| 不同 task 的补齐和 cutoff | `tests/benchmark/test_serving.py:347`，`Scheduler` |
| 成功分子与计算计数 | `tests/benchmark/test_serving.py:428`，`summary` |
| slowdown 匹配与阈值 | `tests/benchmark/test_serving.py:473`，`slo_report` |
| 每轮实际 assistant 历史 | `tests/benchmark/test_serving.py:519`，`run` |
| 可选 token 观察 | `python/minisgl/message/metrics.py:147`，`RequestMetricsState.observe_token` |
| API 传递 ignore_eos | `python/minisgl/server/api_server.py:977`，chat handler 的 `SamplingParams` |
| EOS 停止门槛 | `python/minisgl/scheduler/scheduler.py:350`，`Scheduler._process_last_data` |
| forward 计数位置 | `python/minisgl/scheduler/scheduler.py:983`，`compute_tokens` |
| 真正 EOS 对照及 96K smoke | `scripts/run_serving_matrix.py:160`，`protocol_smoke`；`:179`，`drop_smoke` |
