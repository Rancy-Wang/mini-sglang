# BCP 逐轮 eviction 对照（PLAN-CS-20260914-R2）

入口为 `scripts/benchmark_bcp_eviction.py`，统一完成真实生成、指标统计和最终 pages 验证。
两种 eviction 使用同一提交，仅切换 `--drop-aware-eviction`。不执行记录中的工具调用。

## 数据口径

`prepare` 用生产 Harmony renderer 和 GPT-OSS-120B tokenizer 统计完整输入，不用字符数、
旧 rollout 的 active usage 或重复文本替代长度。困难请求定义为最后一次请求 full tokens
严格大于 131072。按命令行 source 顺序保留每个问题的首个合格轨迹，再按数字 case ID
排序取32个不同问题。每个 case 只保存一份轨迹，并记录来源 SHA256、逐轮消息 SHA256、
轮数和完整长度。`active_tokens_hint` 是消息所有权估计；运行时以服务端 active 指标为准。

每个 assistant 消息之前构成一次请求。当前生成完成后，提交下一段记录前缀；当前输出
保存供审阅，下一轮仍使用记录中的 assistant/tool 消息，使 OFF/ON 的输入固定。
每轮最多4096个生成 token，temperature=0，允许 EOS，报告实际长度和 length 结束比例。
这测量固定轨迹的生成服务能力，不测重新执行 BCP 搜索后的答题准确率。

Rolling K=12 只计完整 tool response。第13条响应后删除第1条并 Reposition，随后每条
新响应触发一次，按每个对话自己的零基 message ID 构造接口。生产执行为 paged-occurrence。

## 矩阵和计时

- 32问题、并发8、TP2：普通与 Drop-aware，各自从第一轮回放至完整轨迹末尾。
- TP2/4 × 并发1/2/4/8 × no_drop/rolling × 两种 eviction：32个共同轮次单元。
- TP2/4 × 并发1/2/4/8 × 两种 eviction：16个完整 rolling 单元。

共同轮次是从开头连续满足 full input +4096 <=131072 的前缀，两种上下文策略完全相同。
完整 rolling 从第一轮独立回放，其吞吐包含建立历史的开销，不把前面轮次藏进 warmup。
1/2/4/8采用同一排序的嵌套 case 集合。并发是客户端对话槽上限；报告实际 GPU batch 分布。

TTFT=(首 token 生成时间-服务端收到请求时间)。
TPOT=(最后 token 生成时间-首 token 生成时间)/(generated_tokens-1)，单 token 为缺失值。
吞吐=完成请求实际 generated_tokens 总数/整个回放墙钟时间；另报轮次/s。
保留 reasoning、终止 token 在内的服务端生成计数，不按输出字符或 SSE 事件估算。
服务端时钟与同机驱动计时均为 monotonic；GPU batch census 使用客户端计时窗口过滤。
失败和未完成轮次显式计数，失败单元不产生加速比。完整轨迹和共同轮次不交叉比较。

## 验证与开销

计时中维持生产检查；测试观察器每次 forward 仅保存 CPU 时间及 batch/graph 信息，
在 eviction 调用处累计实际页释放和存在活跃 handle 时的内部 eviction 次数。
不在 token 热路径遍历树、复制 logits/KV 或加入 CUDA 同步。

计时前用短请求预热并排空缓存；结束后每个 TP rank 检查所有非根节点引用归零，空闲
pages 无重复且不与驻留 pages 重叠，全部物理 pages 有所有者。随后实际分配所有
available pages，确认树中不再有驻留 pages，再归还；最后执行正常健康请求。
共享驻留 page 允许被多个合法节点引用。叶子、内部 Drop 和回填计数分别保存。
普通 eviction 的 leaf_pages 计数原本不更新，因此同时记录公共 evict 返回的实际页数。
额外短健康请求可能影响最终 eviction 累计少量页；它们不进入计时 batch census 或吞吐。

`PASS` 表示本单元全部预期请求成功且最终生命周期检查成功，不代表数学输出等价性证明。
`drop_eviction_covered=false` 表示未覆盖自然内部 Drop eviction，不能宣称该项通过。
请求越界、解析错误、worker 错误和超时保留在逐轮文件及服务日志中，不静默跳过。

## 执行

所有数据、编译缓存和输出必须在仓库外。以下命令在所选 Linux/CUDA 主机执行，
`PYTHONPATH=python`，使用已配置的 GPT-OSS Python 环境。

```bash
python scripts/benchmark_bcp_eviction.py prepare \
  --source /path/to/repos_token96k_repeat2_local8_20260908 \
  --source /path/to/repos36_every24_long55_3xtp2_20260907 \
  --model /path/to/gpt-oss-120b --output /experiment/inputs
python scripts/benchmark_bcp_eviction.py list-cells
python scripts/benchmark_bcp_eviction.py run \
  --input /experiment/inputs --output /experiment/smoke \
  --model /path/to/gpt-oss-120b --gpus 0,1,2,3 \
  --cell scaling-tp2-c1-no_drop-common-ordinary --turn-limit 2
python scripts/benchmark_bcp_eviction.py run \
  --input /experiment/inputs --output /experiment/full \
  --model /path/to/gpt-oss-120b --gpus 0,1,2,3
python scripts/benchmark_bcp_eviction.py report --output /experiment/full --plot
```

默认 chunk=16384、memory_ratio=0.9，保留 CUDA Graph。按 TP 与 eviction 分组复用服务，
完整矩阵只需四次模型/tokenizer初始化；每个单元之前实际排空缓存。服务捕获该组需要的
batch sizes，客户端控制当前单元并发。每个 TP 的第一个单元确定 pages 容量，随后锁定
该 TP 的容量；完整矩阵先运行并发8，避免先用小并发过高估算。
GPU分配：TP2取前两张，TP4取前四张。运行前检查指定卡无任务；仅停止自己启动的进程组。
模型、输入、chunk、内存参数或 smoke 设置改变时必须使用新输出目录。已产生 summary 的
单元不会自动重跑；重试失败单元使用单独目录并保留失败证据。

`--cell` 可限制单元，`--turn-limit` 只用于冒烟，报告中禁止冒烟加速比。
`run.json`、每单元 `launch.json` 保存配置、GPU映射、Git HEAD和精确启动命令；共享
服务日志和逐 rank audit 保存在 `servers/`，单元 summary 内也保存其 audit 结果。
`turns.jsonl.gz` 保存实际输出与原始指标，`summary.json` 保存验证和汇总；
`results.csv`、`comparisons.json` 和 `scaling-tp2/4.png` 提供对比。

本地验证：

```bash
python -m pytest -q -o addopts='' tests/scripts/test_benchmark_bcp_eviction.py
python -m py_compile scripts/benchmark_bcp_eviction.py
```
