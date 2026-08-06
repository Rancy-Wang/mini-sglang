# mini-sglang：本地开发与远端验证规则

本文件作用于整个仓库。更具体的 `python/minisgl/AGENTS.md` 会为 Context
System 代码补充更严格的规则。

## 固定仓库与分支

- 本地仓库：`/Users/x.puppet/Desktop/Undergraduate/Project/Contextual system/mini-sglang`
- GitHub 远程：`origin = git@github.com:Rancy-Wang/mini-sglang.git`
- 工作分支：`System-test`
- 测试服务器：SSH 主机别名 `InfiniAI-BUS`
- 服务器仓库：`/share/wangruoxi/repo/mini-sglang`

开始任务时先核对 `git branch --show-current`、`git remote -v`、`git status
--short --branch`。只在本地 `System-test` 上编辑源码、测试、文档和 Codex
配置。不要通过 SSH 直接编辑远端仓库；远端副本只用于 `git pull` 后的 Linux/CUDA
测试与实验。

每个新窗口以及 resume/compact 后，读取 `docs/codex/PROJECT_CONTEXT.md`、
`docs/codex/CURRENT_STATE.md` 和
`docs/codex/HANDOFF_019fafc9-22dc-7c80-90a0-746297fe72eb.md`。这些文件用于恢复
上下文，不替代动态检查；冲突时以当前 working tree 为准，其次是 `CURRENT_STATE`
snapshot、handoff，最后才是 transcript、粘贴文本或旧说明。恢复上下文不能代替新的
Plan ID、轮次和文件范围批准。

## 修改规则

- 保留本地和远端已有修改。禁止 `git reset --hard`、`git clean`、覆盖式
  checkout/restore、自动 stash、历史重写或无条件强推。
- 不要使用 `git add -A` 或提交与当前任务无关的文件；逐个暂存本轮实际修改。
- 修改前先读取适用的 `AGENTS.md` 和相关 `.agents/skills/*/SKILL.md`。
- 涉及 `python/minisgl` 的 Context System、Throwaway Context、Drop Message、
  Radix/KV Cache、chat template、position 或 prefill 时，必须遵守
  `python/minisgl/AGENTS.md`。
- 优先在本地完成静态检查和可运行的轻量测试，再创建 checkpoint commit。
  macOS 不支持本项目的 Linux/CUDA 运行时，因此 GPU、内核和端到端测试必须在
  `InfiniAI-BUS` 完成。

## Checkpoint 与远端测试

只要本轮产生了仓库修改，就必须使用 `$system-test-checkpoint`：

1. 复核 diff，逐文件暂存本轮修改并在 `System-test` 创建描述清晰的 commit。
2. 将该 commit push 到 `origin/System-test`。
3. SSH 到 `InfiniAI-BUS`，在服务器仓库执行 `git pull --ff-only origin
   System-test`。
4. 核对本地、GitHub 和远端 HEAD 完全一致，再运行与本轮改动相称的远端测试。
5. 若测试失败，继续在本地修复、提交、push、远端 pull 和复测；不要直接修远端。

每次对话原则上形成一个逻辑 checkpoint。提交前合并本轮尚未提交的相关修改，但
不得为了压成单一 commit 而重写已经 push 的共享历史。若本轮需要多个追加修复
commit，以最终通过验证的远端 HEAD 作为本次 checkpoint。

如果本地或远端的已有修改阻止安全 pull/测试，不得清理或覆盖它们；报告准确文件和
阻塞命令。纯只读对话不创建空 commit，使用当前 HEAD 声明 `NO-CHANGE
checkpoint`。

最终回复必须包含：checkpoint 类型、commit hash 与标题、push 状态、远端 pull
状态、测试命令与结果、未解决风险。

## 自定义 agents 与 skills

- 项目 skills 位于 `python/minisgl/.agents/skills/`。
- 项目自定义 agents 位于 `python/minisgl/.codex/agents/`。
- 从仓库根目录启动且任务涉及 `python/minisgl` 时，先手动读取该目录的
  `AGENTS.md` 及所需 skill；从 `python/minisgl` 作为项目窗口启动时，Codex 会
  直接发现这些配置。
- 只有用户明确要求，或适用的 `AGENTS.md`/skill 明确要求时才启动子代理。并行
  agent 只承担互不依赖的只读工作；写源码保持单线程。
