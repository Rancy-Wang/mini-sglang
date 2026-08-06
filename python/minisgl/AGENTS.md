# mini-sglang Context System：Codex 规则

本文件作用于 `python/minisgl` 子树。所有主代理、自定义 agent 和项目 skill 都必须
遵守仓库根目录 `AGENTS.md` 的本地开发、push、远端 pull、远端测试和 checkpoint
协议，并遵守本文件的专项门禁。

## 事实来源与工作边界

- 当前工作树代码是唯一事实来源。用户描述用于定位线索，不能替代代码核实。
- 新窗口以及 resume/compact 后先重读仓库的 `docs/codex/PROJECT_CONTEXT.md`、
  `docs/codex/CURRENT_STATE.md` 和会话 handoff，再运行 `git status --short --branch`、
  `git diff --stat` 和近期 `git log`。动态 working tree 优先于所有 snapshot/摘要。
- 先追踪端到端数据流，再修改实现；不要凭函数名、注释、单个 diff 或搜索命中下结论。
- 保留已有修改，不顺手重构、格式化无关文件、升级依赖或改变未获批的公开接口。
- 解释使用直白中文。代码事实引用 `仓库相对路径:起始行-结束行`，并写出关键符号；
  最终引用须基于当前工作树重新核对。
- 除远端 `git pull --ff-only` 和测试命令外，不在
  `/share/wangruoxi/repo/mini-sglang` 直接创建或编辑源码。

## 必须使用的 skills

根据任务显式读取并执行：

1. `$context-system-change-gate`：任何 Context System 写操作；
2. `$context-system-code-audit`：机制审计、bug 定位和修改前调查；
3. `$context-system-documentation`：撰写或更新技术说明；
4. `$context-system-validation`：修改后的独立验证；
5. `$system-test-checkpoint`：本轮有文件修改时的提交、push、远端同步与测试。

未自动加载时，读取 `.agents/skills/<skill-name>/SKILL.md`。

## 修改批准门禁

Context System 任务按以下状态推进：

`READ_ONLY_DISCOVERY -> PLAN_PENDING_APPROVAL -> APPROVED -> IMPLEMENTING -> VERIFYING -> PASSED | FAILED_PENDING_APPROVAL`

获得本轮明确批准前，只允许读取、搜索、查看 Git 状态/diff、运行保证不改源码的
检查，以及输出计划。禁止创建、编辑、删除或重命名任何仓库文件。

每个修改计划使用唯一编号 `PLAN-CS-YYYYMMDD-RN`，并包含：

- 目标和非目标；
- 已核实的实现、路径、行号和符号；
- 允许修改的精确文件、符号/章节、原因和预期行为；
- 关键不变量；
- 风险、兼容性和不破坏已有工作的回滚方法；
- 本地与远端验证命令、场景和通过标准；
- 明确的批准请求。

只有用户在计划之后明确批准该 Plan ID 和修改范围才算批准。最初任务描述、沉默、
“继续”、工具授权或过去会话的泛化授权都不算批准。

一轮等于“实现已批准计划 + 完整验证 + 独立复核”，最多 5 轮。验证失败后可在同一
已批准范围内根据证据修复和复测；扩大文件或语义范围必须重新规划并请求批准。第 5
轮仍失败则停止并给出人工接管信息。

## Agent 编排

复杂 Context System 任务按需使用：

- `context_code_mapper`：只读追踪代码、调用链和精确引用；
- `context_architect`：只读整合数据流、不变量、风险和计划；
- `context_implementer`：只在收到批准的 Plan ID 与文件边界后单线程写代码；
- `context_verifier`：独立运行检查与测试，不修源码；
- `context_reviewer`：只读审查 diff、引用、回归风险和完成度。

只读调查可拆为四条独立证据链：chat template/message 边界；Drop/Radix/page
释放；full-active/position/check_integrity；match ratio/逐 message prefill。
主代理必须回到源码抽查子代理结论。不得让多个写 agent 并行修改同一工作树。

## Context System 不变量

除非获批计划明确改变语义，否则保持：

- 无 Drop Message 的行为与 mini-sglang 基线一致。
- message id 在请求解析、模板转换、缓存匹配、测试和文档中含义一致。
- 删除 active KV 不改变幸存 token 的原始绝对位置。
- full 与 active token/KV 元数据不混用；每次转换明确输入、输出和所有权。
- Radix key 能区分 token 文本相同但 drop history 不同的上下文。
- drop 同时核实 table manager 映射与 cache manager page 生命周期，避免悬空、
  重复释放和泄漏。
- message 首 token 以完整 chat template 的真实 tokenization 为准，不假设逐消息
  tokenize 后可直接拼接。
- 低匹配率 fallback 的分子、分母、阈值、比较符和触发点必须从当前代码核实。
- `check_integrity` 的调整不能掩盖 page/index 错配。

## 完成条件

只有获批范围内的修改完成、验证通过或环境阻塞已准确列出、独立 reviewer 无阻塞
问题、checkpoint 已 push、服务器已 fast-forward pull 且适用远端测试完成时，才
声称 `PASSED`。最终报告轮次、修改摘要、验证证据、checkpoint 和剩余风险。
