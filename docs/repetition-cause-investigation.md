# Drop 后重复生成的因果定位（PLAN-CS-20260907-R9）

## 结论

本轮已确定重复生成的直接原因，不是 JSON 解析、采样器、CUDA Graph、Radix 命中、页表
别名或注意力后端本身，也不能归类为“模型自身随机异常”。当前 Drop 实现删除目标 token
对应的 KV 行，但继续复用在删除前已经计算好的后续 token KV。后续 token 在更早层曾经
看见目标消息，因此这些“幸存 KV”仍间接包含已删消息；删除或复制目标 KV 行不能消除该
依赖。只有用最终存活 token 重新前向并重算所有层 KV，才能得到与逻辑 active context
一致的状态。

这是一项系统语义问题：物理 KV 集合已经符合最终 keep mask，但幸存 KV 的内容并不等于
在该 keep mask 下重新计算的内容。本计划只负责诊断，不改变生产 Drop 语义或实现；生产
修复必须另行批准范围。

## 可复核的数据流

1. tokenizer 用完整 chat template 生成 canonical token 流，并把最终被删范围置为
   `keep_mask=False`；存活 token 保留原绝对位置（`TokenizeManager._chat_tokenize`，
   `python/minisgl/tokenizer/tokenize.py:1625-1667,1710-1715,1801-1826`）。
2. mask 流在 Drop 发生前仍允许后续幸存 token 看见尚未到期的前缀；注意力 segment 根据
   `visible_until > query_raw` 选择可见 KV（`build_context_attention_segments`，
   `python/minisgl/attention/base.py:290-293,333-360`）。所以某个消息自己的 KV 行被删前，
   它的信息已经进入随后 token 的隐藏状态和更深层 KV。
3. 最终 mask Prefill 后，`Scheduler._compact_context_after_prefill` 只按 keep mask 重排
   page table、token pool 与请求元数据；它没有重新执行幸存 token 的模型前向
   （`python/minisgl/scheduler/scheduler.py:348-418`）。
4. 独立 staged reference 也只释放被删页并压紧幸存页，不重算幸存页
   （`StagedReferenceState.finish_segment`，
   `python/minisgl/scheduler/staged_reference.py:113-153`）。因此 mask/staged 一致并不能
   排除这项共同语义缺陷。

## 冻结输入与对照

诊断适配器是 `tests/contextual/reference_runtime_adapter.py`。它使用 Transformers
`DynamicCache`，不导入 mini-sglang scheduler、Radix、Drop compiler 或注意力后端。
全部实验固定：

- 模型：`AgenticQwen-30B-A3B`，BF16，greedy；
- tokenizer、chat template、config 和权重索引文件哈希；
- canonical token IDs、完整模板边界、Drop trigger、raw position；
- Transformers 4.57.1、PyTorch 2.9.1+cu128、A800 80GB、eager attention；
- Git production 基线 `6da145c`，其生产代码与计划基线 `a888596` 相同。

短用例的首条 user 消息恰为 raw `[0, 12)`；Drop 后活动 prompt 为 raw `[12, 36)`。
以下干预均复用同一模型实例：

| 对照 | 结果 | 排除或证明 |
|---|---|---|
| 无 Drop | `The number mentioned was 42.` | 冻结输入基线正常 |
| 仅删除 raw `[0,12)` KV | 连续生成 `The` | 复现故障 |
| 把所有幸存 KV 复制到新分配缓存 | 与故障逐 token 完全相同 | 排除旧页残留、页表别名和物理布局 |
| 从相同幸存 token、相同原绝对位置重算全部 KV | 与无 Drop 逐 token 完全相同 | 打开/关闭故障的因果干预 |
| 更长会话删除首条 user | 删除/复制失败，重算恢复 | 非单一长度偶然现象 |
| 删除中间 user，首段保留 | 三种路径都正常 | 删除操作本身并不必然失败 |
| 保留 system、删除首条 user | 三种路径都正常 | 存在未污染的早期锚点时不触发该样例 |
| 连续位置的“assistant + 当前问题”，无 Drop | 正常 | 排除孤立 assistant 文本本身 |
| 仅当前问题，无 Drop | 正常 | 排除问题文本本身 |

24 个最小保留干预中，单独保留 raw 1（`user`）或 raw 2（换行）会恢复正确答案；单独
保留 raw 5 或 raw 7 仍重复。这只作为敏感性证据，不能替代“重算全部幸存 KV”的因果
判据。

## 实测指纹与硬判定

独立 runtime 的实际环境为 Python 3.11.15、PyTorch 2.9.1+cu128、Transformers 4.57.1、
CUDA 12.8、NVIDIA A800-SXM4-80GB，attention=`eager`。它使用
`transformers.DynamicCache` 直接执行同一模型，不导入 mini-sglang scheduler、Radix、Drop
compiler 或 attention backend。模型 manifest 同时冻结 chat template、config、generation
config、tokenizer 与权重索引 SHA256；适配器记录这些字段的代码位于
`tests/contextual/reference_runtime_adapter.py:326-449`。

最终判定器在当前测试代码上复查了 11 个硬门，全部为 true：canonical prompt 相同；短、长
Drop 都重复；复制全部幸存 KV 后两者仍逐 token 保持故障；从相同幸存 token IDs 与相同原
绝对位置完整重算后两者都恢复 no-drop 基线；中间 Drop、保留 system、连续 orphan
assistant、question-only 四组控制均正常。判定结果只能因此输出
`stale_survivor_kv_after_drop`（`diagnose_reference`，
`tests/contextual/repetition_cause_runner.py:48-139`）。

冻结产物位于 `InfiniAI-BUS-2:/share/wangruoxi/local/`，没有纳入仓库：

| 文件 | SHA256 |
|---|---|
| `r9_reference_a888596_eager.json` | `4bc47675eafa3717e662f3dbbcba9e960d7db81e581d59723c405284c3f1db4a` |
| `r9_reference_retention_a888596_eager.json` | `e46500bfe4976074c1cfcde94f2cfe99dafec86ee86e002a6dc71640e7dcdcd7` |
| `r9_reference_orphan_a888596_eager.json` | `35f18602cac59f64d9f58b47797672f8a2bb7429f93d21ede0212bb9874f33df` |
| `r9_reference_rebuild_a888596_eager.json` | `4657729e602f7655a4791dc9120e59d05ed95ce98d4018ce52dcf66f5ece9c40` |
| `r9_diagnosis_a888596.json` | `c2c1fa61daee1fe2d88bb3d1e0d9ffd8d0d8eb3cce78d984ad930a98b2dc48cd` |

主短用例的 no-drop 与 rebuild token 均为
`[785,1372,9733,572,220,19,17,13,151645]`，文本为
`The number mentioned was 42.<|im_end|>`；Drop 与 copied 则逐 token 相同并连续输出 `The`。
这同时满足“故障在独立系统复现”和“只改变幸存 KV 是否重算即可关闭故障”两个条件，因此
不能把本故障归为模型本身问题。

## 运行与判定

独立模型实验：

```bash
python tests/contextual/reference_runtime_adapter.py \
  --model /path/to/AgenticQwen-30B-A3B \
  --output /tmp/r9-reference.json \
  --attention eager --max-tokens 32
```

冻结结果判定：

```bash
python tests/contextual/repetition_cause_runner.py \
  --reference /tmp/r9-reference.json \
  --retention-reference /tmp/r9-retention-reference.json \
  --output /tmp/r9-diagnosis.json
```

判定器只有在以下硬门同时成立时才输出 `stale_survivor_kv_after_drop`：原缓存重复、复制
缓存仍逐 token 相同、完整重算恢复无 Drop 输出、较长用例同样恢复，以及中段删除、保留
system、连续孤立 assistant、question-only 对照均通过。算子级说明性回放位于
`tests/attention/test_repetition_operator_replay.py`。

## 边界与后续

- 本结论精确解释本轮短用例的机械重复，并揭示所有“只删 KV 行、不重算后续幸存 KV”
  路径共有的状态不一致。
- BCP 249/802 的输出还可能包含模型策略层面的复述，不能仅凭本短用例宣称每一种自然语言
  重复都由同一机制导致；必须逐请求使用相同因果判据。
- 一个语义严格的修复需要在 Drop 后重算所有受已删上下文影响、但仍存活的后继 token
  KV，或明确把公开语义降级为近似 KV 擦除。前者影响 Prefill 成本、Radix 复用和调度，
  不在 R9 的诊断范围内。
