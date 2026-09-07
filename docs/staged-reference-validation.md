# Staged 顺序参考与 Qwen 工具调用验证（R7/R8）

`contextual_prefill_mode=staged` 在无 Reposition、存在有效 Drop 时运行一个请求内的顺序参考。它用于检查 mask 的语义，不是缓存命中率或速度的对照基线。默认生产模式仍是 mask；无有效 Drop 请求仍走普通 prefill/extend。

## 请求与 KV 的生命周期

完整 messages 只套一次完整模板。Tokenizer 保留 canonical token IDs、原始绝对位置和稀疏 Drop 事件，不为 reference 编译 Radix 布局。切点来自该完整 token 流，而不是重新渲染 `messages[:end]`；所以依赖总消息数的模板也不会造成各段输入不同。入口为 `TokenizeManager._chat_tokenize` 的 `msg.staged_reference` 分支。

`ReferencePendingReq` 在多个调度回合中持有同一个 `Req` 和 `StagedReferenceState`：

1. 从空 KV 开始，用普通 causal extend 计算到下一个 Drop 生效边界；prefill budget 可以提前分块，但不能越过事件。
2. 前向完成后，`finish_segment` 才删除该事件的 KV，并整理 page table、token pool、IDs 和 raw/true positions。
3. 幸存 KV 保留原物理页和绝对位置，下一段直接使用本请求持有的页。Reference 不读取、不提交跨请求 Radix，不通过再次匹配恢复自己的历史。
4. 中间段的采样结果不交付、不追加到生成输入、不计 completion。最终段产生的首 token 在最终 Drop 完成后保留一次，然后按正常路径 decode。
5. 完成或取消时只释放私有页；已经释放的状态再次释放会报错。容量不足会推迟调度，无法容纳的请求返回 `reference_kv_capacity_exhausted`。

Reference 的三个外部复用计数恒为 `cached_tokens=drop_skipped_tokens=repos_tokens=0`。各段使用本请求前段的 KV 不属于 Radix 命中，不能累计成多个 prompt。`prompt_tokens` 只记一次完整输入；`context_stage_count` 记录实际完成段数。

此路径不处理 Reposition；Reposition sequence、retry/RoPE 和其 usage 定义保留原有实现。本次结论不等于完成了 RoPE 语义验证。

## Qwen 工具调用解析

`_QwenToolStream` 同时服务完整响应与流式响应。它跟踪字符串、反斜杠和外层 marker，允许 marker 周围没有换行；字符串内合法的 `</tool_call>` 仍属于参数。只有完整且可解析、名称已知的调用才会进入 `tool_calls`。

非法、未知或未完成的块保留在 `content`，并产生内部 `ToolParseDiagnostic`，包含原因、字符偏移和块序号。参数无法以标准 JSON 表达时同样保留原文。一个块中的多调用先整体校验再发布，防止部分发布后抛异常。默认日志只记录诊断位置与类型。

这修复了错误块及调用前后正文被吞、full/stream 结果不一致等问题。它不修改模型已经生成的 token，不补引号，不重试，也不提供 JSON grammar 约束。模型原始 JSON 非法与 parser 是否保真是两个独立的检查结果。

## R8：生成期工具 JSON 结构约束

R7 的 parser 保真结论不变；R8 没有给 parser 增加补引号、重试或事后修复。对于
AgenticQwen/Qwen3（不含 Qwen3-Coder）的有效工具请求，tokenizer 额外生成一个可序列化
的 grammar descriptor，包含所选工具、`auto` / `required` / 指定函数、结构标签版本，以及
canonical generation prompt 是否已经打开 `<think>`。这里不能直接使用 nullable 的
`enable_thinking` 偏好：AgenticQwen 接受该参数，但它的 generation prompt 不一定实际输出
开始标签；XGrammar 的 reasoning 形式却假设 prompt 已经打开 `<think>`，否则会把工具块当成
任意 reasoning 文本。当前实现因此从最终 prompt 尾部推导该状态。无工具、
`tool_choice=none`、warmup、模板降级 safe mode 及不兼容模型都不生成 descriptor
（`TokenizeManager.__init__`、`_set_tool_grammar`，
`python/minisgl/tokenizer/tokenize.py:121-145,919-964`）。

采样器使用 XGrammar 的 Qwen3 structural tag，在 GPU logits 上只屏蔽本请求当前语法状态
不允许的 token；不存在全局 token 黑名单或硬编码 token ID。编译结果按 tokenizer、词表、
stop token 和完整 descriptor 缓存，而 matcher 始终按 `Req` 对象隔离。CPU matcher 在独立
线程中接受已经提交的 token 并准备下一步 bitmask，与下一次 GPU forward 重叠；GPU 侧只在
采样前等待该请求的 bitmask（`ToolGrammarManager._compile`、`prepare`、`apply`、
`accept_sampled_tokens`，`python/minisgl/engine/tool_grammar.py:26-46,68-241`；
`Sampler.prepare`、`sample`，`python/minisgl/engine/sample.py:57-124`）。

普通 prefill 的第一个生成 token 受约束且 matcher 只前进一次。staged reference 的中间段
sample 和 `ChunkedReq` padding 不创建也不推进 matcher；只有最后一段保留的首 token 才推进
（`Req.sample_is_committed`，`python/minisgl/core.py:398-410`）。批次换序通过 `Req` 身份重新
定位 mask；完成、取消或 UID 复用时丢弃 matcher（`Scheduler._free_req_resources`，
`python/minisgl/scheduler/scheduler.py:579-595`）。无 descriptor 的路径不导入 XGrammar、
不创建 executor/matcher/cache，也不等待 D2H event。

依赖固定为 PyPI 实际发布且可安装的 `xgrammar==0.2.5.post1`；它要求
`apache-tvm-ffi>=0.1.9`。grammar 要求完整 JSON schema、合法转义、Unicode、嵌套对象/数组
和闭合的 tool marker 后才允许 stop token。若 `max_tokens` 先耗尽，服务仍返回
`finish_reason=length`，parser 保留未完成原文与诊断，不把它修成 `tool_calls`。

R8 验证同时覆盖：822/844 的旧 logits 从何处首次被 grammar 拒绝；mask/staged 的真实自由
生成；full/stream HTTP 一致性；`required` 和指定函数；长度截断；多个调用、Unicode、转义
与嵌套 JSON；编译共享、matcher 隔离、批次换序、取消/UID 复用、无工具零初始化和 CUDA
Graph replay。对应入口为 `tests/contextual/mask_staged_runner.py:1138-1346`、
`tests/engine/test_tool_grammar.py:149-226` 和
`tests/server/test_tool_json_generation.py:17-349`。

## R8 实测结果：已知 BCP JSON 故障通过

2026-09-07，AgenticQwen-30B-A3B，A800 80GB，BF16、TP1、生产 FlashAttention 与开启的
CUDA Graph。生产修复运行于 `50cef9ac0cd443b6e9620f8746fc13b937017ad9`；最终 HTTP
验证脚本运行于 `e5d3bb0506605df4a6d418fb76392f832ccf0c81`，后续两个 commit 只调整测试
请求的 `model` 字段与 full/stream 对照隔离，没有改变生产 grammar。冻结输入 gzip SHA256
为 `cfd6f2ed4246cab31c6bb0729b2b63e48bcd02bef372745faafae0d0e114ff8c`。

| 检查 | 结果 |
|---|---|
| 旧 logits 首次拒绝点 | 822 在 step 32 拒绝原 token 11248，改选 95642；844 在 step 30 拒绝原 token 11248，改选 330。两条旧轨迹的 parser 诊断均为 `invalid_json` |
| 真实自由生成 | 822 的 mask/staged 均生成 35 token，844 均生成 34 token；四条结果都能标准 JSON 解码、parser diagnostics 为空并以 `finish_reason=tool_calls` 完成 |
| 已知失败请求 | 822 生成 `search` 参数 `{"query":"\"studied anthropology\" \"musician\" \"University\" \"1980\""}`；844 生成包含 `university instructor`、`books translated`、`twenty languages`、`editor` 的合法 `search` 参数；原来的缺闭合引号错误未再出现 |
| grammar 与 graph | 两个 case 共用一个编译结果；in-process 共发生 135 次 graph replay。无工具控制不初始化 grammar，仍正常生成 `OK`，没有以关闭 CUDA Graph 换取通过 |
| 真实 HTTP | 822/844 的 cold priming、相同 exact-Radix 热路径 full、热路径 stream 全部产生可解码对象；每个 case 的热 full/stream 归一化调用完全相同。`required` 与指定函数均通过 |
| 截断 | `max_tokens=1` 返回 `finish_reason=length` 和原始 `<tool_call>`，没有伪造为完整 `tool_calls` |

HTTP 对照先验证 cold priming，再比较同一热缓存路径的 full/stream。一次诊断中 cold 与 hot
的 844 查询文字不同，但都为合法 JSON；这是不同 prefill/Radix 数值路径，而不是流式组装
差异，所以测试没有用 cold-full 对 hot-stream 作错误的 transport 等价比较。最终 HTTP
结果为 `PASS`，服务端实际捕获 batch size 1/2/4 的 CUDA Graph。

由此，R7 中明确保留的 822/844 原始 JSON 格式失败已经关闭。此结论只保证 grammar 支持的
AgenticQwen/Qwen3 有效工具请求生成结构合法；它不保证工具参数的事实正确性，也不把截断、
不兼容模型或 safe-mode 请求伪装成成功调用。

## 验证方法与边界

CPU 回归覆盖完整模板唯一输入、非可拼接前缀、重复文字与不同 Drop history、无事件快速路径、事件前后预算切分、多个同边界事件、取消和容量拒绝、页面隔离、首 token 及三类 usage。Qwen fixtures 包含本次 822/844 的非法原文、合法嵌套参数、Unicode、转义、连续调用和截断；整段、逐字符、所有单切点、固定随机切块必须给出一致结果。

GPU runner 使用同一个 AgenticQwen 模型与 tokenizer、BF16、TP1 和生产 FlashAttention。仅替换进程间传输，调度、真实前向、采样、页面管理与 detokenize 仍用生产实现。它顺序驱动调度，不覆盖真实网络延迟或 serving overlap 并发的全部时序。

- 实际 FA page table 映射回 raw token，逐组核对 causal 可见集合、IDs/位置和缓存页来源。组内 query 连续、没有 Drop 事件，且 causal suffix 与 query 一一对应；端点相等加这些条件证明中间每个 query，避免长输入的二次复杂度枚举。
- 图片用例固定 93 个 token 的位置，在 tokenized 后端入口送入；它验证图片的数值边界，不声称 AgenticQwen 会把图片文字恰好 tokenize 为 93。命中 49/46 的预期 usage 分别为 24/25/0、46/0/0；最终 active raw 为 `[49,93)`。
- 独立 no-drop 控制比较一次 prefill、同路径重复和相同 causal 语义下的虚拟分段。先冻结数值阈值，再看 10 条长 BCP 的结果。
- 全词表 max_abs、relative RMS、softmax TV 的硬上限分别为 0.1、1%、1%。实际阈值取控制组最大值三倍与硬上限的较小值，下限 1e-6。控制组超硬上限不得通过或提高阈值。
- 自由生成发生分歧时，用事先固定的相同 token 前缀做诊断回放；强制前缀不算模型自行输出一致。argmax 不同时还检查双方候选 margin 是否处于 `2 × max_abs` 范围内。
- 失败后观察选定 raw token 的各层真实 Q/K/V、attention、router、实际 expert 选择、MLP 和残差，不使用手写 dense 全模型参考，不关闭 CUDA Graph。中间数值差异不能仅因最终文本一样而忽略。
- 完整原文、token IDs、usage、stop、JSON 诊断、重复 n-gram 和替换字符分别记录。JSON 合法不代替语义正确；有限样本通过也不证明所有生成绝对正确。

## 可复用命令

两个 `test_staged_reference.py` 位于不同目录，当前 pytest 默认导入模式下分开运行，避免同名模块冲突。CPU 阶段不要设置 `MINISGL_R3_MODEL`，防止触发独立的旧模型测试入口。

```bash
python -B -m pytest -o addopts= -p no:cacheprovider -q \
  tests/server/test_tool_protocol.py \
  tests/server/test_context_usage_regression.py \
  tests/server/test_usage_reporting.py \
  tests/core/test_staged_reference.py \
  tests/core/test_context_prefill_fast_path.py \
  tests/core/test_reposition_staged_prefill.py \
  tests/contextual/test_single_request_prefill.py \
  tests/contextual/test_message_drop_usage.py
python -B -m pytest -o addopts= -p no:cacheprovider -q \
  tests/tokenizer/test_staged_reference.py

CUDA_VISIBLE_DEVICES=2 \
MINISGL_R3_MODEL=/path/to/AgenticQwen-30B-A3B \
MINISGL_R7_INPUT=/absolute/path/to/frozen-bcp10-inputs.json.gz \
MINISGL_R7_OUTPUT=/absolute/path/to/new-unique-output \
MINISGL_R7_SUITE=reference_alignment \
python -B tests/contextual/mask_staged_runner.py
```

输出目录必须不存在；先做请求、模板、token fixture 和参数预检查，随后只初始化一个模型和 tokenizer。Decode graph 只捕获 batch size 1/4，实际 replay、初始化与实验耗时、峰值显存写入 `runtime.json.gz`。长 BCP 主矩阵 C1，短 no-drop/Drop 使用 C4 做并发隔离检查；reference 不用于效率排名。

## R7 实测结果：语义通过，数值未通过

2026-09-07，AgenticQwen-30B-A3B，A800 80GB，BF16、TP1、生产 FA/fused-MoE。主矩阵运行于 `d36ea691082e3bd1f928bf1316a553b5e4f23038`；最终 CPU 回归和第二轮局部探针运行于 `6da145c62a802dc6b573dcc2d68ba171b4394ee2`。后者仅新增测试观测，不改变生产实现。输入 gzip SHA256 为 `4385b15f3cad8603b8008e2bfa3b9050c1556cc03c366a1325dff8fbda5143f7`。

| 检查 | 结果与限制 |
|---|---|
| CPU 主回归 | 122 passed、1 skipped；跳过的是未开启独立模型入口的旧 GPU 测试 |
| Tokenizer 专项 | 4 passed |
| 图片 hit49 / hit46 | 实际 FA 可见集合正确；usage 分别为 C24/D25/R0、C46/D0/R0；各一次正式 prefill，保留首 token 后进入 decode |
| 10 条长 BCP 的 Drop 语义 | 实际 prefill page table、KV 来源、query 可见集合、原始位置和最终 active 检查通过；mask 一次 prefill，reference 五段，共用一个正式 UID |
| no-drop 数值控制 | 同路径重复逐位相同；相同 causal 语义但虚拟切段后超过冻结硬上限 |
| 10 对自由输出 | 5 对全部 token 相同，5 对分歧；全部正常 EOS；同前缀全词表数值检查 10 对均 FAIL |
| 原始 JSON | 16/20 合法；822、844 在两种方法下都生成缺少字符串闭合引号的非法 JSON；parser 保留错误原文与诊断，没有交付可执行调用 |
| C4 短请求 | 实际 batch 含四个请求，页面与可见性检查通过，graph 正常 replay；这不等于数值或输出质量通过 |

C4 Drop 的 reference 四条均生成 `The The The The`，mask 为 `The answer is the`。这是四 token 截止的短隔离检查，不能据此判定完整答案；但其重复和方法间分歧必须保留为未解决质量证据。10 条长 BCP 中另有 249 两路、802 mask 重复历史查询。没有发现这 20 条长 BCP 新输出中的 U+FFFD 或同一 8-token n-gram 出现三次；该筛查不等于排除了所有重复或推理问题。

主实验只初始化一次模型和 tokenizer，decode graph 捕获 batch size 1/4，实际 replay 908 次。初始化含 preflight、加载与 capture 共 95.36 秒；随后矩阵、第一轮探针、C4 与导出/收尾共 94.01 秒；峰值已分配显存 69,029,310,976 bytes。它不是完整 HTTP 网络压测，也没有执行 BCP 生成后的外部搜索工具，因此不能证明最终答案正确。

**当前 reference 不能认证为后续 mask 修改的数值基准。** `runtime.status=completed` 和 runner 退出 0 只表示实验执行完毕；必须读取 `summary.json.gz` 的 `numeric_status` 与 `failures`，本轮分别为 `FAIL` 和非空列表。不得因文本相同、语义通过或进程成功退出而把整体标为 PASS。

## 剩余数值差异的定位

第一轮在 806、844 各选 17 个 raw token，采集全部 48 层的真实算子。第 0 层被观测 token 的 QKV projection 输入逐位相同，但 generation header 的两个观测位置已经出现 projection 差异，最大绝对值 `6.103515625e-05`。这是所选观测中的首次差异，不是所有 token 的完整首次差异证明。

第二轮仅加载第 0 层 `[5120,2048]` QKV 权重，用相同的已捕获输入，比较原大矩阵形状 `[22112,2048]` / `[22399,2048]` 与 reference 末段 `[3,2048]`。其余独立行置零，保留被测行在矩阵中的位置；生产 `F.linear` 的两种形状分别逐位复现了原 capture，graph 与 eager 也逐位相同。这直接证明 batch 形状本身足以造成上述 projection 差异。此探针没有初始化完整模型，graph replay 四次；测量段 1.86 秒，不包括进程导入及权重载入。

同时观察到 MoE 临界路由分歧：806 第 1 层 raw 22109 的实际 expert 集合首次变化，双方 top8/top9 分差均为 0；844 第 2 层 raw 19598 首次改变集合，分差为 0.015625 / 0。层号从 0 开始。选定 token 的全部路由检查没有发现“放弃严格更高分而选更低分”的 top-k 错误；选中 expert 权重与 FP32 softmax 的最大差为 `1.1920928955078125e-07`。

这支持“分段形状引入数值差异，临界 expert 路由变化可能放大差异”的解释，但不能把全部误差归到上述两个 header token：844 的首次集合变化位于 header 之前，不能由未来 header 反向造成。没有证明哪个方法更接近理想高精度结果，也没有定位全部 token/算子的误差贡献。FA、GEMM、MoE 数学实现及 dtype 均未改动；进一步修复需要单独范围。

第二轮探针命令（输入目录必须已有对应 operator artifacts）：

```bash
CUDA_VISIBLE_DEVICES=2 \
MINISGL_R3_MODEL=/path/to/AgenticQwen-30B-A3B \
MINISGL_R7_SUITE=projection_replay \
MINISGL_R7_INPUT=/absolute/path/to/main-output \
MINISGL_R7_OUTPUT=/absolute/path/to/new-probe-output \
python -B tests/contextual/mask_staged_runner.py
```

## 当前源码证据索引

以下引用基于 `6da145c`，区间为包含两端的源码行；没有用旧 warmup 文档代替当前实现。

| 行为 | 文件、行与符号 |
|---|---|
| 单个正式请求分流，兼容 warmup 入口不再执行前向 | `python/minisgl/server/api_server.py:568-580`，`run_contextual_warmup`；`:862-899`，`v1_completions` |
| 无 Drop 普通快速返回；完整 canonical reference | `python/minisgl/tokenizer/tokenize.py:1424-1450`、`:1594-1626`，`TokenizeManager._chat_tokenize` |
| canonical / CSR 校验、事件切分、先前向再 Drop | `python/minisgl/scheduler/staged_reference.py:42-83`、`:89-102`、`:113-154`，`StagedReferenceState` |
| 私有页登记、释放与 Radix 旁路 | `python/minisgl/scheduler/cache.py:271-304`，`allocate_paged`、`free_reference_pages`、`cache_req`；当前私有路径要求 page_size=1 |
| 私有请求续段、容量与队列 | `python/minisgl/scheduler/prefill.py:678-729`、`:858-867`、`:925-941`，`_add_reference`、`complete_reference`、`abort_req` |
| 中间 sample 隔离、最终 sample 保留、overlap 屏障 | `python/minisgl/core.py:363-369`，`Req.complete_one`；`python/minisgl/scheduler/scheduler.py:179-187`、`:248-266` |
| 严格 Qwen block 解析与失败原文保留 | `python/minisgl/server/response_parser.py:159-313`，`_QwenToolStream`；`:873-903`，`ChatResponseParser` |
| 实际 FA 消费的分段 metadata | `python/minisgl/attention/fa.py:195-235`、`:300-315`，`prepare_metadata` / `_forward_context_segments`，本轮未修改 |
| 实际 page provenance 与逐 query 集合验证 | `tests/contextual/mask_staged_runner.py:580-680`，`visibility_oracle`、`VisibilityObserver`；`:1104-1114`，由原始 message Drop 独立重建到期位置 |
| 数值门槛、主实验、局部重放 | `tests/contextual/mask_staged_runner.py:897-915`、`:935-1075`、`:1139-1220` |
| 形状重放调用的生产线性层 | `python/minisgl/layers/linear.py:31-32`，`_LinearTPImpl.forward`，本轮未修改 |

原始长 BCP 输入、20 条完整生成轨迹、logits/operator tensors 和文件 hashes 保留在独立实验目录，未纳入代码仓库。原始 JSON 格式失败是本次明确保留的 FAIL；进一步数学修复不在 R7 的批准范围，不能用文档将此状态改写为通过。
