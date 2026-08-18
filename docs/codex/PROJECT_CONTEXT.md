# mini-sglang Context System 项目上下文

本文解释 2026-08-18 `System-test` 上的 Context System。它是机制导航，不是修改授权；动态
事实和批准范围仍按 `AGENTS.md`、change gate 与 `CURRENT_STATE.md` 检查。默认组合是
`radix_drop_key_mode="delta-marker"` 与 `contextual_prefill_mode="mask"`
（`python/minisgl/scheduler/config.py:16`，`SchedulerConfig`）。

## 一、统一 Drop Rule 输入

公开 `drop_rule.type` 只选择三个类：

- `MessageDropRule` / `message_drop`：按完整 message owner 删除，字段名为
  `drop_messages`，不使用 `schedule`；
- `TextDropRule` / `text_drop`：按与原始 `messages` 等长、同序、同 role 的 content
  selector 删除；
- `ThinkingDropRule` / `thinking_drop`：保留结构化 thinking 文本进入完整 Radix token
  流，但删除其 KV。

三个类和校验分别始于 `python/minisgl/tokenizer/drop_rules.py:60`、`:138`、`:288`，统一
解析在同文件 `:335`。旧顶层 `drop_message` 只作为 `MessageDropRule` 的兼容入口；新旧字段
并存会拒绝。HTTP canonicalization 在 `python/minisgl/server/api_server.py:161`，canonical
wire 通过 `python/minisgl/message/tokenizer.py:79` 进入 tokenizer。

## 二、完整模板、owner 与 query epoch

消息边界不能通过“每条消息独立 tokenize 后拼接”推断。`TokenizeManager` 对每个完整
conversation prefix 重新应用 chat template；稳定前缀/后缀用于保守 owner attribution，只有
稳定前缀继承旧 query epoch。实现见 `python/minisgl/tokenizer/tokenize.py:934`
（`_merge_owner_track`）和同文件 `:979`（`_round_by_round_no_gen`）。

普通 Hugging Face Drop 请求还会对完整 Jinja render 插入临时 provenance marker，再将
character owner 通过 fast-tokenizer `offset_mapping` 投影回 token。canonical token IDs 必须
与 `apply_chat_template(tokenize=True)` 完全相同；实现见
`python/minisgl/tokenizer/template_provenance.py:206`。这使 role header、BOS/EOS、分隔符、
generation prompt、模板重写和重复 content 都从完整模板事实确定，而不是靠文本猜 owner。

最终保留两条不同轴：

- owner 回答“该 token 属于哪条 message”；
- query epoch 回答“该 token 在哪次 prefix render 首次进入或被重写”。

owner 的连续区间在 `python/minisgl/tokenizer/tokenize.py:596` 构造。`full_input_ids` 保留完整
模板轴，`input_ids` 是最终 active 轴，`true_positions` 保留 active token 的完整绝对位置，
不会因 Drop gap 重新编号；字段组装在同文件 `:1159` 的 `_chat_tokenize`。

## 三、TextDropRule 的 raw text 到 token 映射

每条 selector 必须是对应原始 `messages[i].content` 的大小写敏感精确子串，不做 Unicode
normalization。`content` 支持 `null | str | list[str]`；空形态是 no-op。重复匹配允许重叠，
默认取从左到右第 1 次；一旦提供 `occurrence`，`list[str]` 的每段都必须提供对应正整数。

匹配器是 UTF-8 Aho-Corasick：Python AOT wrapper/reference/fallback 在
`python/minisgl/kernel/text_match.py:127`，稀疏 trie、failure/output-link C++ FFI 在
`python/minisgl/kernel/csrc/src/text_match.cpp:37`。复杂度是输入、patterns 与实际输出之和；
输入上限用于阻止异常资源消耗。

raw character span 先在该 owner 的 canonical rendered content 中唯一定位，再通过 token
offset 映射。只有完全包含在 selector span 中的 token 才 Drop；跨边界 token 保留，若没有
任何完整 token 则请求失败。selector 并集完整覆盖 raw content 时提升为完整 message owner
ranges，因此与相同目标的 `MessageDropRule` 精确等价。规则在最新 user Prefill 完成后生效；
编译分支见 `python/minisgl/tokenizer/tokenize.py:1045`。

## 四、ThinkingDropRule 的 retention 与定位

规则不开启时不做 retention probe、不覆盖 chat template，也不改变模型自己的 thinking 删除
行为。开启后只接受 assistant `reasoning_content` 或 content 开头唯一、完整、非嵌套的
`<think>...</think>`；两种来源不能同时出现，也不从普通 prose 推测。

Hugging Face 模板按需运行 capability probe，并以 tokenizer 身份、模板 SHA 和 tools variant
缓存。原生可保留的模板不改；已识别 Qwen guard 使用 request-local Jinja adapter，要求
reasoning 精确出现一次且无 reasoning 的输出不变；未知过滤模板返回
`thinking_history_not_preservable`。入口见
`python/minisgl/tokenizer/thinking_template.py:86`。

GPT-OSS 使用 Harmony token stream。只有 Thinking rule 开启时才设置
`RenderConversationConfig(auto_drop_analysis=False)`；随后逐 component 渲染并校验其 token
前缀与 conversation render 一致，记录 analysis 内容的绝对 token cursor。Drop 只覆盖内容，
channel/recipient/protocol token 保留；规则未开启时 Harmony 默认 `auto_drop_analysis` 不变。
实现见 `python/minisgl/tokenizer/tokenize.py:137`、`:273` 和 `:1122`。

每段 thinking 的事件位于对应 assistant message 末尾：assistant final 在同轮仍可见 thinking，
后续 user/tool/assistant generation 不可见。

## 五、统一绝对 position delta 与 Radix key

三种规则都先产生绝对半开 token ranges，再由
`python/minisgl/tokenizer/tokenize.py:700`（`_build_position_range_drop_plan`）排序、合并、
去除已生效重复区间。事件插入点是 `bisect_right(query_epoch, event_n)`；代码拒绝 token 尚未
计算就被隐藏的 range，并生成 `full_token_visible_until` 供 mask warmup 使用。

Scheduler 在每个事件边界插入 virtual delta marker：相同 canonical delta 复用同一负
signed-int64 key，不同 Drop history 进入不同 Radix 分支。marker 没有 KV page，真实 token
保留 key/token 双向映射。相关入口是：

- `python/minisgl/scheduler/radix_delta.py:57`，`DeltaMarkerRegistry`；
- 同文件 `:151`，`DeltaRadixLayout`；
- 同文件 `:223`，`key_prefix_len_for_token_boundary`；
- 同文件 `:269`，`inject_delta_markers`。

Scheduler 建立/释放 request refs 见 `python/minisgl/scheduler/scheduler.py:269` 与 `:407`。
delta-marker 要求最终 `page_size=1`，且不兼容要求非 unit page 的 TRTLLM；启动校验始于同文件
`:57`。

## 六、full match、active KV 与提交

Radix 先匹配完整 key 轴，移除 virtual marker 的 `-1` page 后得到 `full_match_indices`，再按
`prefix_keep_mask` 过滤为 `active_match_indices`。入口分别见
`python/minisgl/scheduler/cache.py:202`（full match）、`:154`（active match）和 `:143`
（`matchable_prefix_lens`）。该函数同时返回 Drop 前 full matchable 长度和 Drop 后 active
matchable 长度；最后一个强制 Prefill token 与 virtual marker 都不进入分母
（`python/minisgl/scheduler/cache.py:121-152`）。

`python/minisgl/scheduler/prefill.py:23-40` 的 `_calculate_cache_ratios` 明确产生两个值：

- 公开 `cache_hit_ratio = cached_active_len / full_matchable_prefix_len`，表示“命中的未 Drop
  token / Drop 前全部可匹配真实 token”；
- 内部 `cache_reuse_ratio = cached_active_len / active_matchable_prefix_len`，表示幸存上下文
  的复用完整度。

两者空分母都定义为 `1.0`，且强制检查
`cached <= active_matchable <= full_matchable`。

请求结束时，`python/minisgl/scheduler/cache.py:473` 的 `_cache_finished_delta_req` 用 active
KV slot 与 absolute position 重建 full-token prefix；virtual page 固定为 `-1`，真实 hole
截断可提交前缀，slot 冲突直接报错。Radix 生命周期与完整性检查分别见
`python/minisgl/kvcache/radix_cache.py:581`（insert）、`:640`（prune）、`:715`（evict）和
`:863`（integrity）。不能通过放宽 integrity 检查掩盖 page/index 错配。

## 七、mask warmup、staged 回退与 API ratio

有当前生效 Drop 的请求默认发送一次 mask warmup，携带完整 token 轴和
`full_token_visible_until`。入口是
`python/minisgl/server/api_server.py:755`（`run_contextual_warmup`）。后端从同一可见性语义
生成原生布局：

- 通用 segments/batch：`python/minisgl/attention/base.py:102`、`:174`；
- FlashInfer metadata：`python/minisgl/attention/fi.py:330`；
- FlashAttention 能力与 metadata：`python/minisgl/attention/fa.py:65`、`:133`、`:195`。

不支持的实际 Prefill backend 在启动期失败，不会退化为普通 causal attention。FA4 mask
Prefill 仍限制每批一个 mask 请求，GPT-OSS sinks/sliding-window 与该 mask 组合未启用。

`staged` 只作兼容/排障：warmup 使用内部 `cache_reuse_ratio`，首次
`cache_reuse_ratio >= 0.95` 即结束，严格低于 `0.95` 才发送逐 message prefix；因此 Drop
比例本身不会制造虚假的低命中 fallback。Scheduler 发送该内部值见
`python/minisgl/scheduler/scheduler.py:225-233`，比较点见
`python/minisgl/server/api_server.py:799-803`。commit boundary 通过
`python/minisgl/scheduler/radix_delta.py:223` 映射到包含边界 marker 的 key prefix。

最终用户生成请求的 `cache_hit_ratio` 只在 terminal reply 传播
（`python/minisgl/scheduler/scheduler.py:255`）：非流式位于 OpenAI response 顶层
（`python/minisgl/server/api_server.py:1161`），流式只位于最后一个 finish chunk
（同文件 `:953`）。内部 warmup ratio 不暴露给用户。

## 八、tied-weight 启动兼容

tied 模型的运行时模板不重复登记 `lm_head.weight`，但合法 checkpoint 可以同时保存
embedding 与 lm-head 两个别名。`Engine._load_weight_state_dict` 只对模板内键采用模板 dtype；
模板外键保持 checkpoint dtype 并继续传给模型加载器
（`python/minisgl/engine/engine.py:159-177`）。tied `ParallelLMHead` 消费冗余 weight/bias，
非 tied 模型仍按普通参数加载，其他未知键仍会留给顶层严格检查
（`python/minisgl/layers/embedding.py:59-85`、`python/minisgl/layers/base.py:32-53`）。不能
回退为统一 `self.dtype` 转换，因为 GPT-OSS packed MXFP4 权重需要保持模板指定的
`uint8`。

## 九、CUDA Graph 默认策略与回退

`cuda_graph_max_bs=None` 表示默认开启，而不是关闭。所有模型（包括 GPT-OSS）统一根据启动前
可用 GPU 显存生成 capture shapes：显存大于 80 GiB 时最大 batch size 为 256，否则为 160；
显式 `0` 才产生空列表并关闭。列表生成见
`python/minisgl/engine/graph.py:47-66`（`_determine_cuda_graph_bs`），公开参数及关闭别名见
`python/minisgl/server/args.py:149-165`。

`GraphRunner` 对每个 shape 分别捕获，并在 TP 模式下通过 CPU process group 汇总所有 rank 的
成功状态。只有所有 rank 均成功的 graph 才进入 `graph_map`；该映射是可 replay shape 的唯一
事实来源，失败 shape 只回退 eager，不影响其他 shape。捕获和 TP 汇总见
`python/minisgl/engine/graph.py:118-166`，Engine 传入通信组见
`python/minisgl/engine/engine.py:117-129`。Decode batch 向上 padding 到最小可用 shape 后才允许
replay；Prefill 始终保持原 batch size，见 `python/minisgl/engine/graph.py:168-186`。

## 十、操作不变量与验证边界

必须保持：无 Drop 基线不变；public message 顺序与模板 provenance 对齐；绝对 position 不
压缩；full/active metadata 不混用；不同 Drop history 不碰撞；request/tree marker refs 与
page 生命周期闭合；fallback 阈值、比较符和命中率公式不漂移。

本地只在 `System-test` 编辑。获批修改须先 commit/push，再到
`InfiniAI-BUS:/share/wangruoxi/repo/mini-sglang` 执行 `git pull --ff-only origin System-test`，
使用 `/share/wangruoxi/.conda/envs/minisgl-gpt-oss-r4` 做 Linux/CUDA 验证；远端源码不直接改。
