# 当前状态 snapshot（2026-08-05）

这是便于新窗口恢复的静态 snapshot，不是“仓库仍然如此”的保证。每次 startup、resume
或 compact 后都先动态运行：

```bash
git branch --show-current
git rev-parse --short HEAD
git status --short --branch
git diff --stat
git log --oneline -8
```

事实冲突时以 working tree 为准，其次才是本文、会话 handoff 和旧 transcript。

## Snapshot 基线

- 本地仓库：`/Users/x.puppet/Desktop/Undergraduate/Project/Contextual system/mini-sglang`
- 分支：`System-test`
- HEAD：`b43e5c8`（`Sort token Drop test imports`），当时与
  `origin/System-test` 一致。
- 最近 Context System 提交（旧到新）：
  - `fbd9010 Implement token-position Drop backend`
  - `08af742 Fix Radix bool mask FFI dtype`
  - `a9ba6e6 Flatten token Drop range wire metadata`
  - `f489533 Format token Drop implementation`
  - `e5052bd Release Radix markers on invalid commit boundary`
  - `1b8c03a Isolate marker cleanup regression test`
  - `b43e5c8 Sort token Drop test imports`

## 当时的 dirty working tree

共有 17 个 tracked dirty 文件，diff stat 为 775 insertions / 95 deletions：

```text
pyproject.toml
python/minisgl/attention/base.py
python/minisgl/attention/fa.py
python/minisgl/attention/fi.py
python/minisgl/engine/engine.py
python/minisgl/engine/graph.py
python/minisgl/layers/attention.py
python/minisgl/layers/linear.py
python/minisgl/message/tokenizer.py
python/minisgl/models/config.py
python/minisgl/models/register.py
python/minisgl/models/weight.py
python/minisgl/scheduler/scheduler.py
python/minisgl/server/api_server.py
python/minisgl/server/args.py
python/minisgl/tokenizer/detokenize.py
python/minisgl/tokenizer/tokenize.py
```

这些修改是续接 `PLAN-CS-20260804-R3` 的 GPT-OSS/Context System 原型；不得被本次
Codex handoff 配置任务覆盖、restore、stash 或格式化。

主要 untracked GPT-OSS 实现与 tests：

```text
python/minisgl/models/gpt_oss.py
python/minisgl/moe/mxfp4.py
tests/attention/test_gpt_oss_attention.py
tests/models/test_gpt_oss_config.py
tests/models/test_gpt_oss_e2e.py
tests/models/test_gpt_oss_weight.py
tests/moe/test_gpt_oss_mxfp4.py
tests/server/test_cuda_graph_staged.py
```

另有尚未纳入 Git 的项目 `AGENTS.md`、5 个自定义 agent、5 个 Context System skill 及
checkpoint scripts；它们是主代理待 checkpoint 的项目配置，不是可清理噪声。`.DS_Store`
属于纯 OS 噪声。

## 原型覆盖与验证状态

working tree 原型覆盖 GPT-OSS 模型注册/配置、Harmony chat 与 stop/output、MXFP4 packed
weight sharding/runtime、attention sinks 与 sliding window、分阶段 CUDA graph capture，
同时携带现有 delta-marker/staged Context System。详细续接边界见会话 handoff。

验证尚未完成。续接状态只记录了 `compileall` 与 diff/scope 检查通过；尚未运行新加的
focused `pytest`，也尚未在 `InfiniAI-BUS` 的 `rosetta` 环境做 CUDA、TP 或端到端验证。
因此不得声称 GPT-OSS 或 FA3 Context System mask 已受支持，也不得把本 snapshot 当成
`PASSED` checkpoint。

## 固定同步流程

本地编辑 -> 精确暂存本任务文件 -> commit -> push `origin/System-test` ->
`InfiniAI-BUS:/share/wangruoxi/repo/mini-sglang` 执行
`git pull --ff-only origin System-test` -> 激活 conda `rosetta` -> 运行相称的远端测试。
远端只 pull/test，不直接改源码。
