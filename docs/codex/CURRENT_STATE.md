# 当前状态 snapshot（2026-08-18）

这是恢复新窗口时使用的静态 snapshot，不替代动态检查。每次 startup、resume 或 compact 后
仍须先运行：

```bash
git branch --show-current
git rev-parse HEAD
git status --short --branch
git diff --stat
git log --oneline -8
```

事实冲突时按 working tree > 本文 > handoff > 旧 transcript 的顺序处理。

## Snapshot 基线

- 本地仓库：`/Users/x.puppet/Desktop/Undergraduate/Project/Contextual system/mini-sglang`
- 分支：`System-test`
- R3 实现提交：
  - `c4526a5 feat(context): add selectable token-position drop rules`
  - `bd513a4 fix(context): retain and map Harmony thinking components`
- R5 loader/ratio 实现提交：
  - `0080608 fix(context): support tied weights and honest cache ratios`
- 上述实现提交已 push 到 `origin/System-test`，并在
  `InfiniAI-BUS:/share/wangruoxi/repo/mini-sglang` fast-forward 拉取。包含本 snapshot 的后续
  文档 checkpoint 应以动态 `git rev-parse HEAD` 为准。

远端测试前原有的下列 untracked 文件保持原样，没有移动、清理或提交：

```text
contextual_readme.md
python/minisgl/CODEX_TASK_CONTEXT_SYSTEM.md
python/minisgl/INSTALL.md
python/minisgl/MANIFEST.md
python/minisgl/docs/
tests/core/test_contextual_prefill.py
tests/kernel/test_context_mask.py
```

## PLAN-CS-20260818-R3 已实现范围

公开请求统一使用 `drop_rule`，并仅暴露三个规则类：

- `MessageDropRule` / `message_drop`：字段继续叫 `drop_messages`，旧顶层
  `drop_message` 在入口转换为该规则；没有 `schedule` 字段。
- `TextDropRule` / `text_drop`：`drop_messages` 与历史 `messages` 等长、同序、同 role；
  `content` 支持 `null | str | list[str]`。重复子串默认取最靠前 occurrence，也可完整提供
  occurrence 编号。UTF-8 Aho-Corasick AOT CPU kernel 返回允许重叠的字符区间；只有完整落在
  selector 内的 token 被删除，跨边界 token 保留，完整 content 覆盖提升为 message owner
  ranges。
- `ThinkingDropRule` / `thinking_drop`：不开启时不修改 tokenizer；开启后仅从结构化
  `reasoning_content` 或唯一 leading `<think>` block 取 thinking。Qwen 历史过滤模板使用
  request-local guard adapter，未知模板 fail closed；Harmony 仅在该规则开启时关闭
  `auto_drop_analysis` 并逐 component 映射 analysis 内容 token。

三条规则统一编译成绝对 token-position delta wire，继续复用 delta-marker Radix、绝对
position、mask/staged warmup 与 cache 生命周期。最终生成请求的 `cache_hit_ratio` 已贯通
Scheduler -> Detokenizer -> Frontend：非流式响应顶层返回，流式只在 terminal chunk 返回。

## PLAN-CS-20260818-R5 已实现范围

- tied-weight checkpoint 可以同时包含 `model.embed_tokens.weight` 与冗余
  `lm_head.weight`。`Engine._load_weight_state_dict` 对模型模板内键继续按模板 dtype 转换，
  保护 GPT-OSS packed `uint8`；模板外键原样交给严格模型加载器，已有 tied
  `ParallelLMHead` 消费冗余 lm-head 键，真正未知键仍由顶层拒绝。入口见
  `python/minisgl/engine/engine.py:159-177` 与
  `python/minisgl/layers/embedding.py:59-85`；顶层剩余键检查见
  `python/minisgl/layers/base.py:32-53`。
- 公开 `cache_hit_ratio = cached_active_len / full_matchable_prefix_len`；分母是 Drop 前全部
  Radix 可匹配真实 token，不含最后一个强制 Prefill token 与 virtual marker。内部
  `cache_reuse_ratio = cached_active_len / active_matchable_prefix_len` 只供 warmup/fallback
  使用。长度计算和公式见 `python/minisgl/scheduler/cache.py:121-152`、
  `python/minisgl/scheduler/prefill.py:23-40`；Scheduler 分流见
  `python/minisgl/scheduler/scheduler.py:225-255`。

核心入口见：

- `python/minisgl/tokenizer/drop_rules.py:60`、`:138`、`:288`；
- `python/minisgl/kernel/text_match.py:127` 与
  `python/minisgl/kernel/csrc/src/text_match.cpp:37`；
- `python/minisgl/tokenizer/thinking_template.py:86`；
- `python/minisgl/tokenizer/tokenize.py:700`、`:1045`、`:1159`；
- `python/minisgl/server/api_server.py:161`、`:755`、`:953`、`:1161`。

## 已完成验证

本地只做只读/静态验证：所有修改 Python 文件 `compileall` 通过，`git diff --check` 通过；
R3 修改范围未超出获批的 23 个路径，R5 代码修改未超出获批的 7 个源码/测试路径。`ruff`
在本地和远端专用环境都未安装，因此没有声称 lint 通过。

远端使用 `/share/wangruoxi/.conda/envs/minisgl-gpt-oss-r4`，pytest 因该环境没有 coverage
插件而显式使用 `-o addopts=`：

- 首轮聚焦测试：48 passed / 1 failed；唯一失败暴露了 Harmony analysis token 不能依赖旧
  owner 猜测，随后改为 retention config + component token cursor。
- Harmony 修复定向复测：23 passed。
- Context/Radix/mask/API/GPT-OSS 扩展回归：67 passed。
- R5 tied loader/ratio 聚焦回归：19 passed；扩展 model/cache/scheduler/server 回归：70
  passed。

真实模型与服务验证：

- Qwen3-1.7B 在 GPU 7 上实际生成 C2C `math_followup` 首轮 assistant：
  `15 + 27 = 42.`。在完整四条历史中，`MessageDropRule {3:[1,2]}` 与等价完整-content
  `TextDropRule` 的 full IDs、keep mask、active IDs、absolute positions 和 delta metadata
  全部相等；事件位置为 `55`，Drop 绝对区间为 `[11,42)`。
- DeepSeek-R1-Distill-Qwen-1.5B 单卡 mini-sglang HTTP 服务中，Message/Text 两请求均返回
  200、相同输出；流式 ratio 只出现在第 4 个 terminal chunk。
  不具备已识别 history-retention guard 的该模型对 `thinking_drop` 返回预期 HTTP 400
  `thinking_history_not_preservable`。
- 原始 `/share/public/public_models/Qwen3-1.7B` checkpoint 实际同时含
  `model.embed_tokens.weight` 与 `lm_head.weight`。R5 后 GPU 7 直接加载该目录成功，日志到达
  `Model weights are ready` 和 Uvicorn startup，真实 chat completion 返回 HTTP 200；不再需要
  `model-view-untied` workaround。
- 同一 Qwen3 四消息实验中，无 Drop 冷/热请求的公开 ratio 为 `0.0 -> 1.0`；先缓存完整
  prompt，再以 `MessageDropRule` 删除历史 user+assistant 后，冷/热 Drop 请求均为
  `25 / 56 = 0.44642857142857145`。25 是命中的 active token，56 是 Drop 前 matchable
  token；内部复用率为 `25 / 25 = 1.0`，不会误触发 staged `< 0.95` fallback。

## 当前限制与后续检查

- partial `TextDropRule` 的 canonical 字符 offset 映射要求 fast Hugging Face tokenizer；
  GPT-OSS Harmony 当前正向覆盖的是完整-message Text promotion 与 Thinking component 映射。
- Thinking retention 只自动适配已验证的 Qwen guard；未知会删除历史 thinking 的模板必须
  新增有后置校验的专用 adapter，不能猜测普通 prose。
- AOT matcher caps：16 MiB source、4096 patterns、1 MiB pattern bytes、1,000,000 matches。
- 完成任何后续修改后仍执行：精确暂存 -> commit -> push -> 远端 `git pull --ff-only` ->
  相称 Linux/CUDA 测试；远端源码只 pull/test，不直接编辑。
