# Staged Drop 历史可见性修复与 BCP 对照

本轮按用户明确要求“修改 mini 的错误 stage，然后再作对照”执行，记录为
`PLAN-CS-20261107-R3`。mini 基线为 `fb24883`，生产修复为 `7505600`；
sglang 对照固定在 `9c156eb1d4`，未修改其生产代码或数值阈值。

## 修复行为

旧 staged Drop 先发送完整历史的 compact prefill 探测，再按 0.95 命中率决定是否执行
消息前缀 warmup。探测并非只读，可能用最终 Drop 状态计算较早的 query；fallback
又遗漏最后一条历史消息。BCP case 778 的 Qwen 正式请求因此从 raw 8690 开始计算时，
提前删除 raw `[4488,7242)` 的 2754 个键，虽然 Drop 应在 raw 10479 才生效。

`python/minisgl/server/api_server.py:613-649` 的
`FrontendManager.run_contextual_warmup` 现在：

1. 不发送完整历史探测，不按 0.95 比例短路。
2. 按 `messages[:1]` 至 `messages[:n]` 的顺序执行，包含最后消息。
3. 每条 warmup 请求启用既有历史可见性规划。缓存足够且读集等价时，原生
   `PrefillAdder.plan_context_prefill` 仍选择 compact mask-free Extend；若 eviction、
   模板前缀差异等造成缺失，则走已有 mask 恢复路径。不能保证缓存驻留时，不假定历史
   已计算完毕。
4. 正式 Drop 请求同样启用这项检查，防止最后 warmup 与正式请求之间的缓存丢失。
   入口为 `api_server.py:979-981` 的 `TokenizeMsg.use_context_mask`。

`python/minisgl/tokenizer/tokenize.py:1676-1680` 的 `warmup_commit_token_len`
对所有 warmup 限定历史提交边界，包括启用可见性检查的 warmup；临时生成 footer
不能成为下一条消息的缓存来源。

这是前端逐消息 staged 的正确性修复，仍有逐前缀模板开销，并非将 mini 改写为
sglang 的单一完整模板 executor。Drop 保持幸存绝对位置；Reposition 顺序协议、
普通请求和默认 mask 入口保持原有流程。没有修改模型、attention 内核、CUDA Graph、
SWA 窗口或采样算法。

usage 使用正式请求的原生 reply。各前缀的 reply 彼此重叠，且属于不同 Drop epoch，
不再用其中一条覆盖最终 usage。`cached_tokens` 因而可能包含本次内部 warmup 建立的
缓存，不能将它当成请求到达前的外部缓存命中数，也不能据此计算总 prefill 工作量。
采集器另行保存全部 warmup records。

## 回归验证

新增/更新测试覆盖：高/低命中率均按历史顺序执行并包含末消息；warmup 的可见性保护；
生成 footer 不发布；正式请求丢失缓存时仍携带恢复所需元数据；流式和非流式最终 usage。
既有 Context prefill/recovery 测试同时检查可见性等价判断与缺页恢复。

本地 macOS CPU：初次四个目标文件 57 passed；增加 footer 回归后 tokenizer 文件
4 passed。使用已有 myenv 和任务依赖，没有修改依赖版本。

BUS-2、mini `75056003f3056df3448316bfa1586ef54b79ab45`：

```bash
source /share/wangruoxi/local/akv/activate-bus2.sh
cd /share/wangruoxi/repo/mini-sglang
export PYTHONPATH=$PWD/python
CUDA_VISIBLE_DEVICES=99 /share/public/wangruoxi/.conda/envs/minisgl-gpt-oss-r4/bin/python \
  -B -m pytest -o addopts= -p no:cacheprovider --tb=short -q \
  tests/server tests/tokenizer tests/core/test_context_prefill_fast_path.py \
  tests/core/test_drop_aware_recovery.py
```

结果为 **161 passed、1 failed**。失败为
`tests/server/test_gpt_oss_startup.py::test_exact_tp4_cli_resolves_gpt_oss_dtype_before_engine_startup`，
其断言要求 `ServerArgs` 没有 `drop_aware_eviction` 字段，但基线 `fb24883` 的
`python/minisgl/scheduler/config.py:19` 已定义该字段；相关配置、参数解析和测试文件
本轮均无变化。归类 `PRE_EXISTING_FAILURE`，不以修改无关断言掩盖失败。

## BCP 对照方法

使用冻结的 case 778 历史及 tools：
`/share/public/wangruoxi/local/sglang-system-20260917/bcp778-native-tools.json`。
Qwen3-0.6B 输入 10482 tokens，GPT-OSS-20B 输入 9671 tokens；TP1、page size 1，
mini 生产 attention，sglang Triton，graph/overlap 开启。每个模型只初始化一次 mini
runner，比较分支之间按测试要求清空或保留缓存。正式输出为 8 tokens。

修复后的生产参考由 sglang 现有 `test/registered/context_system/mini_staged_oracle.py`
调用真实 mini API/tokenizer/scheduler/model 生成，不启用
`CONTEXT_MINI_DIAGNOSTIC_FINAL_WARMUP`。参考 mode 为 `staged`，并记录真实 mini commit。
随后运行 `test_staged_runtime.py::test_bcp_staged_reference`：无操作、Drop、
Drop+Reposition 的自由生成和固定 token 对照，以及 hot、Retry、连续 Reposition。
固定输出前缀与新 mini 参考一致；GPT-OSS 修复后生成前缀改变，不能直接复用旧 sglang
Drop logits 计算完整 8 行误差。

mini 环境为 PyTorch 2.9.1+cu128，sglang 为 PyTorch 2.13.0+cu130。

参考采集命令（Qwen；GPT-OSS 替换模型名和输出名）：

```bash
source /share/wangruoxi/local/akv/activate-bus2.sh
cd /share/wangruoxi/repo/sglang
export PYTHONPATH=/share/wangruoxi/repo/mini-sglang/python
export CONTEXT_MINI_ROOT=/share/wangruoxi/repo/mini-sglang
export CONTEXT_SERVER_MODEL=/share/public/wangruoxi/models/Qwen3-0.6B
export CONTEXT_BCP_FIXTURE=/share/public/wangruoxi/local/sglang-system-20260917/bcp778-native-tools.json
export CONTEXT_ORACLE_OUTPUT=/isolated/new-mini-qwen.json CONTEXT_NUMERIC_TOKENS=8
CUDA_VISIBLE_DEVICES=0 /share/public/wangruoxi/.conda/envs/minisgl-gpt-oss-r4/bin/python \
  -B test/registered/context_system/mini_staged_oracle.py
```

sglang 对照命令（GPT-OSS 另加 `--moe-runner-backend triton_kernel`，并替换模型、
oracle、输出目录、日志与端口）：

```bash
source /share/wangruoxi/local/akv/activate-bus2.sh
cd /share/wangruoxi/repo/sglang
export PYTHONPATH=$PWD/python:/share/wangruoxi/runtime/staged-20261107-r2/deps
export CONTEXT_SERVER_MODEL=/share/public/wangruoxi/models/Qwen3-0.6B
export CONTEXT_BCP_ORACLE=/isolated/new-mini-qwen.json
export CONTEXT_START_TIMEOUT=600 CONTEXT_CHUNK_SIZE=8192
export CONTEXT_MAX_LENGTH=24576 CONTEXT_KV_CAPACITY=24576
export CONTEXT_TRACE_DIR=/isolated/new-qwen-staged
export CONTEXT_SERVER_LOG=/isolated/new-qwen-server.log CONTEXT_SERVER_PORT=28771
export CONTEXT_TEST_ATTENTION_BACKEND=triton
export CONTEXT_SERVER_EXTRA_ARGS='["--context-execution-mode","staged"]'
CUDA_VISIBLE_DEVICES=1 python -B -m pytest -s --tb=short -o addopts= \
  -p no:cacheprovider -q \
  test/registered/context_system/test_staged_runtime.py::test_bcp_staged_reference
```

新产物根目录为 `/share/wangruoxi/runtime/staged-20261107-r3`；旧结果保留在
`staged-20261107-r2`。GPU 0 用于 mini，GPU 1/2 用于 sglang Qwen/GPT-OSS；
端口分别 28771/28772，日志和输出目录分离。BUS-2 GitHub 拉取超时后，通过已推送提交
的 Git bundle 执行 `git pull --ff-only`，没有编辑远端源码。

## 数值结果

两种模型的 warmup targets 均为 `[0,1,2,3,4,5,6,7,8]`，正式请求首个补算 raw token
分别为 Qwen 10479、GPT-OSS 9669。正常缓存条件下所有 warmup 实际都选中 mask-free
Extend；历史 input id 与绝对位置逐项匹配。无操作及所有 Drop+Reposition 参考 logits
与 `fb24883` 时逐元素完全一致，修复只影响 Drop 分消息路径。

Qwen Drop 对照：

| mini 参考 | max abs | mean abs | p99 abs | 原门禁 |
| --- | ---: | ---: | ---: | --- |
| 修复前 `fb24883` | 2.4375 | 0.174954 | 0.714417 | FAIL |
| 修复后 `7505600` | 0.59375 | 0.073974 | 0.25 | PASS，包括逐 token 检查 |

Qwen 整套仍有两项失败：`drop_repos-fixed` 和 `drop_repos-consecutive` 的
`per_token_mean[5] = 0.116310`，超过原上限 `0.083301`。这两项与修复前相同，
不能将 Drop 项通过描述为整套通过。
Qwen Drop 自由生成仍不逐 token 一致；上表 PASS 仅指相同强制前缀下的原数值门禁。

GPT-OSS 修复后的 Drop 自由生成与 sglang 8/8 tokens 相同；固定前缀 max/mean/p99 为
2.410156 / 0.079832 / 0.65625。修复前对应指标为 2.8125 / 0.185938 / 0.754883，
但新旧 mini 从第 5 个生成 token 开始分歧，完整 8 行不是完全相同的前缀条件；
每次 sglang/mini 对照内部均使用各自当次参考的相同固定前缀。
新 Drop 的 aggregate max/mean 通过原上限，p99 和部分逐 token 项仍失败，尤其
第 3 个 token max 2.410156、mean 0.294260。不能因自由输出一致就判定数值通过。
只取新旧共同前缀对应的前 5 行，sglang logits 在两次运行中逐元素相同；
对 mini 的 mean 差值由 0.208705 降为 0.108147，max 由 2.5 降为 2.410156，
p99 由 0.8125 降为 0.75。这一子集是严格相同前缀条件下的修复前后对照。

GPT-OSS 整套剩余 6 项失败：Drop 的 p99、`per_token_max[2]`、
`per_token_max[7]`、`per_token_mean[2]`，以及既有 Drop+Reposition hot 的
`per_token_max[6]` 和 `per_token_mean[6]`。两种模型完整配对测试各返回 1 failed，
全部 HTTP、完整输入 id、固定输出 token、logits 形状检查均通过。

本轮不修 sglang 原生无操作 SWA decode 的 127/128 窗口差异，也不改变模型级误差
门禁；它仍使 GPT-OSS 的无操作校准包含语义差异。Drop+Reposition 原有超限项仍需
独立处理。这次完成的是 mini 历史 stage 修复及修复后对照，不是整个迁移的最终数值验收。

额外的 Qwen 缓存清空实验在 9 条 warmup 和正式请求 dispatch 前调用原生
`Runner.clear()`，共 10 次，保留其他生产路径。正式请求随后执行 8192+2290 tokens
的 mask 恢复，8 个输出 token 与默认 mini mask 完全一致，全部 logits 也逐元素相同
（max/mean 均为 0）。这验证了缓存丢失保护，未用手写 attention 替代模型。
该产物标记为 `diagnostic-stage-eviction`，与正式参考分开保存。

关键产物：`mini-{qwen,gpt20}.json(.pt)`、`reference-changes.json`、
`qwen-staged/comparison.json`、`qwen-staged/gate-failures.json`、
`gpt20-staged/comparison.json`、`gpt20-staged/gate-failures.json`、
`mini-qwen-eviction.json(.pt)`、`eviction-diagnostic.json`、`unit.log`。
`comparison-summary.json` 保存同前缀修复前后指标、完整失败项及非固定输出是否一致。
