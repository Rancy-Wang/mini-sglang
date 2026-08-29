# Context System 请求正确性与 Serving 性能测试

本文说明如何从 Contextualize 的 Tau3 或 BrowseComp-Plus 实际任务中选择请求，冻结为可复现
workload，并对 mini-sglang System-test 进行请求级正确性和 serving engine 性能测试。测试工具不
内置实验 payload；每次正式测试以带 SHA-256 的 JSONL manifest 为唯一输入。

## 一、范围

第一阶段正确性范围是：

- 四个指定模型；
- `full`、`summary` 两种方法；
- 历史 assistant message 中 `reasoning_content`、`content`、`tool_calls` 的七种非空组合。

理论矩阵为 `4 × 2 × 7 = 56` 个单元。性能测试可使用 `full`、`drop_kv`、`summary`、
`summary_drop_kv` 四种配置，但 DropKV 正确性明确延期；`record-oracle` 和 `verify` 会拒绝
把包含 Drop payload 的请求当成已支持的正确性方法。

测试代码提供捕获、冻结、oracle、重放、匹配、并发和报告能力。最终 payload 选择、GPU 服务
启动、实际实验执行和结果解释由测试执行者负责。

## 二、数据流

```text
Contextualize 实际任务
        │  agent backend 指向 capture proxy
        ▼
captures.jsonl（原始 OpenAI request）
        │  prepare + 人工筛选/标注
        ▼
manifest.jsonl（冻结 request + hash + method/source/model）
        ├── 标准 SGLang ── record-oracle ── oracle.jsonl
        ├── System-test ── verify ───────── correctness-report.json
        └── System-test ── bench ────────── performance-report.json
```

正确性比较的是同一次模型调用，而不是两套系统各自运行后已经分叉的完整 agent trajectory。
Contextualize 只在准备阶段生成真实 post-policy workload；正式性能计时直接重放冻结请求，因而
不包含 Contextualize 策略、Summary 模型或工具执行耗时。System-test 为 DropKV 所做的内部
warmup 属于 serving engine 工作，计入服务器 TTFT 和 E2E。

冻结、正确性范围门禁和覆盖矩阵分别由
`python/minisgl/benchmark/contextualize/manifest.py:44-192` 的 `request_hash`、
`ManifestCase.ensure_correctness_scope`、`ManifestCase.ensure_performance_scope` 和同文件
`:300-339` 的 `coverage_matrix` 实现。

## 三、Serving 端指标

`/v1/chat/completions` 的非流式响应顶层，以及流式响应最后一个 finish chunk，返回：

```json
{
  "server_metrics": {
    "request_received_ns": 100,
    "first_token_generated_ns": 200,
    "request_finished_ns": 500,
    "prompt_tokens": 1000,
    "active_prompt_tokens": 700,
    "generated_tokens": 65,
    "completion_tokens": 64
  }
}
```

时间戳来自同一台 serving 主机的 `time.perf_counter_ns()`：

- `request_received_ns`：进入 OpenAI handler、Context warmup 开始前；
- `first_token_generated_ns`：scheduler 同步并取得第一个采样 token；
- `request_finished_ns`：scheduler 取得 terminal token；
- `prompt_tokens`：完整 canonical prompt token 数；
- `active_prompt_tokens`：Drop 后最终请求保留的 prompt token 数；
- `generated_tokens`：真实采样步数，包括被响应层抑制的 terminal EOS；
- `completion_tokens`：除被抑制 terminal EOS 外的生成 token 数。对 GPT-OSS，这仍包含模型实际
  生成的 reasoning 和 Harmony protocol tokens，不能理解成最终 `content` 的重新分词长度。

对应公式：

```text
TTFT = first_token_generated_ns - request_received_ns
E2E  = request_finished_ns - request_received_ns
TPOT = (request_finished_ns - first_token_generated_ns) / (generated_tokens - 1)
```

只有一个 generated token 时 TPOT 为 `null`。`usage` 与 `server_metrics` 是两条独立数据链：

- 非流式 `usage` 完全保留新版 System-test 已有的 `Req.prompt_tokens` 和
  `Req.completion_tokens` 口径；
- 流式响应不额外新增 `usage`；
- benchmark 只读取 `server_metrics`，不会用 `usage` 兜底或对两套计数做合并；
- `server_metrics.completion_tokens` 排除被响应层抑制的 terminal EOS，因此可能比既有
  `usage.completion_tokens` 少一；`generated_tokens` 始终保留真实采样步数。

这种分离保证新增观测不会改变既有公共 usage 行为。内部时间也不会受 GPT-OSS Harmony 输出在
HTTP 层被缓存或拆分的影响。

wire 数据结构和计数状态见 `python/minisgl/message/metrics.py:8-84` 的 `ServerMetrics` 与
`RequestMetricsState`。HTTP 起点见 `python/minisgl/server/api_server.py:754-864` 的
`v1_completions`；首/末 token 记录与计数状态初始化见
`python/minisgl/scheduler/scheduler.py:209-282,398-420`。流式 terminal chunk 和非流式响应
输出分别见 `python/minisgl/server/api_server.py:581-683` 的
`FrontendManager.stream_chat_completions` 与
同文件 `:865-935` 的 `v1_completions` 非流式分支。

## 四、捕获 Contextualize 请求

先启动标准 SGLang 或 System-test 服务，例如服务根地址为 `http://127.0.0.1:30000`。然后启动
透明 capture proxy：

```bash
python -m minisgl.benchmark.contextualize capture \
  --upstream-base-url http://127.0.0.1:30000 \
  --host 127.0.0.1 \
  --port 18000 \
  --output artifacts/captures.jsonl
```

把 Contextualize 的 agent backend base URL 指向 `http://127.0.0.1:18000/v1`，再运行所选
Tau3 或 BrowseComp-Plus 任务。若 Summary 使用独立 backend，应让 Summary backend 继续直连其
固定服务，只代理 agent backend，避免把 Summary 模型调用混进 workload。

proxy 在转发前记录 `/v1/chat/completions` 的原始 JSON body、捕获时间、capture ID 和
request SHA-256。proxy 只用于准备数据；不要经由 proxy 运行正式性能测试。

透明转发和请求落盘实现见
`python/minisgl/benchmark/contextualize/capture_proxy.py:54-110` 的 `create_capture_app`。

## 五、准备与检查 manifest

Full 示例：

```bash
python -m minisgl.benchmark.contextualize prepare \
  --capture artifacts/full-captures.jsonl \
  --output artifacts/full-manifest.jsonl \
  --source tau3 \
  --method full
```

Summary 示例：

```bash
python -m minisgl.benchmark.contextualize prepare \
  --capture artifacts/summary-captures.jsonl \
  --output artifacts/summary-manifest.jsonl \
  --source browsecomp-plus \
  --method summary \
  --summary-triggered
```

capture proxy 无法仅从 HTTP body 可靠判断 Summary 是否触发。只有在任务轨迹确认该调用已经
插入 Summary 时，才能使用 `--summary-triggered`。未确认或显式标为 false 的 Summary case 不会
计入正确性覆盖。

每条 manifest case 包含：

- `case_id`；
- 完整 `request`；
- `request_sha256`；
- `metadata.source/method/model/summary_triggered`；
- 可选的 `target_message_index`、tags、matcher 和 oracle。

如果一个真实请求包含多种 assistant message 结构，它默认可以覆盖多种 shape。需要严格指定
被测 message 时，在 metadata 中设置 `target_message_index`；该索引必须指向非空 assistant
message。修改 request 本体后必须重新生成 hash，不能手工保留旧 hash。

检查矩阵：

```bash
python -m minisgl.benchmark.contextualize coverage \
  --manifest artifacts/correctness-manifest.jsonl \
  --models MODEL_A MODEL_B MODEL_C MODEL_D \
  --output artifacts/coverage.json
```

工具会逐单元输出 count，并把缺少真实 case 的单元列入 `missing_cells`，不会把缺失覆盖报告为
通过。

capture 到 manifest 的转换见
`python/minisgl/benchmark/contextualize/runner.py:256-283` 的 `prepare_manifest`。

## 六、生成 oracle 与验证

用标准 SGLang 生成 reference：

```bash
python -m minisgl.benchmark.contextualize record-oracle \
  --manifest artifacts/correctness-manifest.jsonl \
  --base-url http://127.0.0.1:30000/v1 \
  --output artifacts/oracle.jsonl
```

再把完全相同的冻结请求发给 System-test：

```bash
python -m minisgl.benchmark.contextualize verify \
  --manifest artifacts/oracle.jsonl \
  --base-url http://127.0.0.1:8000/v1 \
  --output artifacts/correctness-report.json
```

`record-oracle` 和 `verify` 强制使用非流式 HTTP 响应，改变的只有响应传输形式，不改变 messages、
tools、sampling 或 Drop payload。服务端模型名称不一致时可使用 `--model-override`，manifest 中
的原始请求和 hash 不会因此被改写。

默认 matcher 是规范化文本 exact match，并结构化比较 tool call 的 function name 和 JSON
arguments，忽略随机 tool-call ID。每条 case 也可以配置：

```json
{"matcher":{"mode":"prefix","prefix_chars":64,"compare_reasoning":true}}
```

或：

```json
{"matcher":{"mode":"keywords","keywords":["required phrase"],"compare_tool_calls":true}}
```

prefix 以 reference 的前 N 个字符为准，适用于中文和无空格文本。keywords 要求所有关键词出现
在 target reasoning/content 的合并文本中。

matcher、oracle 记录和 target 重放入口见
`python/minisgl/benchmark/contextualize/runner.py:87-121` 的 `compare_messages`、同文件
`:286-316` 的 `record_oracles` 与 `:319-370` 的 `verify_cases`。

## 七、性能测试

性能测试支持 Full/Summary × DropKV 开/关四种配置：

| method | 实际 Summary 状态 | 请求中的 Drop payload | 用途 |
| --- | --- | --- | --- |
| `full` | `summary_triggered=false` | 无 | full-context 基线 |
| `drop_kv` | `summary_triggered=false` | 有 | 单独观察 DropKV |
| `summary` | `summary_triggered=true` | 无 | 单独观察 Summary 后的 serving workload |
| `summary_drop_kv` | `summary_triggered=true` | 有 | 组合方案 |

这里的 Summary 状态是实际轨迹事实，不是根据请求 body 自动推断。性能 manifest 必须用
`--summary-triggered` 或 `--summary-not-triggered` 明确标注，且 method 必须与 Summary 状态及
Drop payload 一致。为了让吞吐量和延迟可比较，一次 `bench` 只接受一种 method；四种配置应使用
四份 manifest 分别运行。最好从同一个真实任务集合构造配对 workload，并用 metadata tags 保存
共同的场景 ID。Summary 开/关时 post-policy messages 可以不同，但采样参数、输出长度约束和案例
数量应尽量一致。

下面以 `summary_drop_kv` 为例：

```bash
python -m minisgl.benchmark.contextualize bench \
  --manifest artifacts/summary-dropkv-manifest.jsonl \
  --base-url http://127.0.0.1:8000/v1 \
  --concurrency 1 4 8 \
  --num-requests 100 \
  --warmup-requests 3 \
  --preserve-stream \
  --output artifacts/performance-report.json
```

同一次命令的 1/4/8 三组按相同顺序循环使用同一冻结 manifest。`--preserve-stream` 保留原始
`stream` 字段；不提供时强制非流式，便于做纯 serving engine 对比。GPT-OSS 实际 Contextualize
调用使用 streaming 时建议提供该参数。

每组报告：

- 成功/失败请求数；
- requests/s；
- completion tokens/s；
- generated tokens/s；
- TTFT、TPOT、E2E 的 mean、P50、P95、P99；
- 客户端从发送请求到完整读取响应的 E2E mean、P50、P95、P99；
- prompt、active prompt、dropped prompt token 数与 prompt retention ratio 的分布；
- `drop_requested` 和 `drop_effective_requests`；
- 每条请求的原始 server-derived sample 或错误。

`drop_requested` 表示请求带有 `drop_rule` 或 `drop_message`。`drop_effective` 由服务端真实计数
`active_prompt_tokens < prompt_tokens` 判定；因此配置了 DropKV 但本次没有实际减少 token 的
请求仍会保留在报告中，而不是被误报为已发生 Drop。无 Drop 配置也可以正常测量，其 active
prompt 应等于完整 prompt。

吞吐量以整组客户端 wall time 为分母，不把各 worker 的局部 tokens/s 相加。正式请求数建议至少
为最大并发的五倍，并尽量让 manifest 本身包含足量的唯一请求，避免循环少量 case 造成不真实的
Radix 热缓存。正式比较四种 method 和并发 1/4/8 时，应为每个 method × concurrency 单元恢复
相同的初始服务/cache 状态，建议分别重启服务并每次只传一个 `--concurrency`。同一服务实例中
一次传入 `1 4 8` 适合功能 smoke，后组可能继承前组的 Radix cache，不能直接当作严格的并发
对比结果。

server sample 派生与并发汇总见
`python/minisgl/benchmark/contextualize/runner.py:388-435` 的 `_distribution`、
`_derive_server_sample` 和同文件 `:438-578` 的 `benchmark_cases`。

## 八、完整多轮 trajectory 性能测试

`bench-trajectories` 用于比较同一批完整多轮任务在 `no_drop` 和 `thinking_drop` 下的逐 turn
serving 性能。它与上一节的 `bench` 是两个独立入口：`bench` 把 manifest case 当作互相独立的
请求并发，`bench-trajectories` 则保证同一个 task 的请求严格按照 turn 顺序发送，同时让不同
task 并发。

### 8.1 输入目录

`--trajectory-dir` 下递归查找 `*.jsonl`，每个文件代表一个 task，每行继续使用 capture proxy
产生的 `CaptureRecord` 格式，文件行序就是 turn 序：

```text
full-trajectories/
├── task-001.jsonl
├── task-002.jsonl
└── subset/
    └── task-003.jsonl
```

```json
{"capture_id":"...","captured_at_ns":123,"request":{"model":"...","messages":[]},"request_sha256":"..."}
```

loader 会重新检查每一行的 SHA-256，并执行 fail-closed 门禁：请求必须有非空 `messages`，不能
携带 `drop_rule` 或 `drop_message`，后一 turn 的 `messages` 必须以前一 turn 的完整
`messages` 为前缀，每个 task 必须至少有 `--max-turns` 行，并且 task 数不能少于最大并发数。
因此 Summary 或 Summary-Drop 记录不能混入该目录。

目录读取与上述门禁见
`python/minisgl/benchmark/contextualize/manifest.py:214-280` 的 `load_full_trajectories`。

### 8.2 命令和采样参数

默认运行方式是：

```bash
python -m minisgl.benchmark.contextualize bench-trajectories \
  --trajectory-dir artifacts/full-trajectories \
  --base-url http://127.0.0.1:30000/v1 \
  --output artifacts/trajectory-performance.json
```

默认参数等价于：

```text
--concurrency 1 2 4 8
--variants no_drop thinking_drop
--max-turns 10
--max-tokens 2048
--temperature 0.0
--top-p 1.0
--top-k -1
```

参数默认值见 `python/minisgl/benchmark/contextualize/runner.py:1055-1074` 的
`bench-trajectories` parser。

`temperature=0.0, top_p=1.0, top_k=-1` 是当前服务的 greedy 组合。runner 深拷贝每条冻结请求
后覆盖这些 sampling 字段，并固定 `n=1`、`ignore_eos=true`、`stop=null`，同时移除优先级更高
的 `max_completion_tokens`。原 capture 和原 hash 不会改变；报告同时保存 source hash 与实际
wire request 的 effective hash。

固定长度指 `server_metrics.generated_tokens == max_tokens`，不是最终可见文本的重新分词长度。
如果服务端因为最大序列长度或其他限制没有生成满，原始样本仍写入报告，但
`fixed_length_ok=false`、`passed=false`，且不会进入该 turn 的可比统计样本。

不带 `--preserve-stream` 时强制非流式；带该选项时保留原始 `stream` 字段，并自动设置
`stream_options.include_usage=true`。流式 usage 是 `choices=[]` 的独立末尾 chunk，runner 会
单独读取该 chunk，不依赖普通 delta choice。

采样覆盖、ThinkingDrop payload 和 stream usage 请求由
`python/minisgl/benchmark/contextualize/runner.py:674-719` 的 `_build_trajectory_request`
构造；响应解析见同文件 `:144-226` 的 `_parse_nonstream` 与 `_post_streaming`。

### 8.3 逐 task 调度与 ThinkingDrop

并发单位是 task trajectory，而不是单条 HTTP 请求：

```text
semaphore(concurrency)
├── task A: turn 1 -> 等待响应 -> turn 2 -> ... -> turn N
├── task B: turn 1 -> 等待响应 -> turn 2 -> ... -> turn N
└── task C: turn 1 -> 等待响应 -> turn 2 -> ... -> turn N
```

因此 `--concurrency 4` 表示最多同时推进四个 task；同一个 task 永远只有一个在途请求。后续
turn 仍然重放冻结的完整 history，不会把本次 benchmark 的响应改写回下一个捕获请求。

`no_drop` 不添加 Drop payload。`thinking_drop` 的 turn 1 作为两种模式共同的 bootstrap，也
不添加 Drop；从 turn 2 开始添加：

```json
{"drop_rule":{"type":"thinking_drop"}}
```

为避免把无效 Drop 静默当成对照结果，运行前要求 thinking-drop 的 turn 2 到 N 都至少含一条
历史 assistant `reasoning_content` 或合法的前置 `<think>...</think>`。不满足条件时在发送任何
请求前拒绝整个 workload。

预检、task 级 semaphore 和逐 turn 顺序执行见
`python/minisgl/benchmark/contextualize/runner.py:766-960` 的 `benchmark_trajectories`。

### 8.4 报告格式

顶层 `cells` 的每一项对应一个 `variant × concurrency` 实验单元。`requests` 保存逐 task、逐
turn 的 source/effective hash、TTFT、E2E、TPOT、客户端 E2E、原始 `server_metrics`、标准
OpenAI usage、Drop 是否请求/生效以及固定长度检查。usage 统一为：

```json
{
  "prompt_tokens": 8000,
  "completion_tokens": 2048,
  "total_tokens": 10048,
  "prompt_tokens_details": {
    "cached_tokens": 7200,
    "drop_skipped_tokens": 350
  }
}
```

服务端未返回 `prompt_tokens_details` 时，两个 detail 数值在报告中规范化为 0，同时
`usage_details_reported=false`，以区分“服务端没有报告细节”和“服务端明确报告了零”。

每个 cell 的 `turns` 对同一 turn 的所有有效 task 样本分别汇总 TTFT、TPOT、E2E、客户端 E2E
以及五项 usage 计数，输出 mean、P50、P95、P99。`sample_count` 是真正进入分布计算的固定长度
样本数；只有八个 task 时，P95/P99 只能视作小样本长尾信号。

usage 校验和规范化见 `python/minisgl/benchmark/contextualize/runner.py:585-624` 的
`_normalize_usage`；逐 turn 汇总见同文件 `:722-763` 的 `_aggregate_trajectory_turn`。

### 8.5 Cache 隔离

runner 不会重启服务，也不会清空 Radix cache。一次命令运行多个 cell 时，报告明确标记
`cache_isolation=shared_endpoint` 并保存 `execution_order`；这种运行适合功能 smoke，不是严格
冷 cache 性能结论。

报告中的输入参数、cache 隔离状态和执行顺序见
`python/minisgl/benchmark/contextualize/runner.py:962-987`。

正式对照应在每个 cell 前重启到相同初始状态，并且每次只传一个 variant 和一个并发数，例如：

```bash
python -m minisgl.benchmark.contextualize bench-trajectories \
  --trajectory-dir artifacts/full-trajectories \
  --base-url http://127.0.0.1:30000/v1 \
  --concurrency 4 \
  --variants thinking_drop \
  --output artifacts/thinking-drop-c4.json
```

完整的正式矩阵需要 `2 variants × 4 concurrencies = 8` 次独立服务运行。

## 九、限制与结果解释

- 标准 SGLang 输出是行为 reference，不是底层数值的绝对真理。exact mismatch 应结合 prefix、
  keywords、tool-call 结构和服务端日志归因。
- 黑盒输出一致不能证明所有内部 message boundary 都正确；它是 request-level integration
  evidence。七组合 coverage 只能说明实际 payload 覆盖情况。
- `summary` case 必须冻结已经插入 Summary 的 post-policy request。不要让 reference 和 target
  各自重新生成 Summary。
- 任意包含 DropKV 的性能结果都不构成 DropKV 正确性结论。
- Summary 配置只测量冻结后的 serving workload；不包含 Contextualize 策略判断和 Summary
  模型本身的执行耗时。
- server timestamps 只能在同一 serving 主机内部相减，不应当作跨机器 wall-clock 时间使用。
- 本地 CPU tests 不能替代 GPT-OSS-120B 四种性能配置的 Linux/CUDA 端到端验证。
