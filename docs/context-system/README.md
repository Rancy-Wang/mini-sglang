# Drop Message Context System

Mini-SGLang can remove selected earlier chat messages from later attention while
preserving the original token timeline and reusable KV cache. This operation is
called **Drop**.

## Messages and token ownership

Public message IDs are the zero-based indices of the submitted `messages` array.
Internal normalization, including an injected tool description, does not change
these public IDs.

Messages are not tokenized independently. The model's chat template renders the
whole conversation, so delimiters, role headers, generation prompts, and even
earlier text may change when a new message is appended. Mini-SGLang therefore
builds two tracks over the final token stream:

- **Owner:** the public message, or synthetic next-assistant prompt, that owns each
  rendered token. Drop removes message-owned tokens by owner ID.
- **Query epoch:** the prefix-construction step at which each token enters the
  conversation. Message `M_i` uses epoch `i`; the final generation prompt uses
  epoch `N = len(messages)`. A Drop event uses this track to determine its exact
  activation position.

`MessageDropRule` 的分割流程只对完整 conversation 做一次真正的 tokenizer encode：

1. 保持用户提交顺序，并在 request normalization 后保留 public message ID。
2. 用模型原生 chat template 做 render-only：一次得到带 generation prompt 的 canonical
   文本，另一次不带 generation prompt，只用于确定最后的 synthetic assistant 边界。这两次
   都是 Jinja 字符串渲染，不调用 tokenizer encode。
3. 系统编译一个 request-local 的 traced Jinja template，在每个 output node 前后临时写入带
   message object ID 的随机 nonce marker。marker 只存在于这份旁路 render，解析 owner 后
   立即删除；删除后的文本必须逐字符等于 canonical 文本，否则 fail closed。
4. canonical 文本以 `add_special_tokens=False` 和 `offset_mapping` 调用 tokenizer **一次**。
   `<|im_start|>`、`<|im_end|>`、BOS/EOS、role header 和模型的其他原生控制 token 都来自
   原生模板文本并按原 tokenizer 规则编码。
5. character owner 经 offset 投影为 token owner；generation prompt 使用 synthetic owner
   `N = len(messages)`。MessageDrop 的 query epoch 直接由这个单调 owner 轴得到，不再对
   `messages[:i]` 做逐前缀 tokenize。

这些 marker 不是 `AddedToken`，不会写入 tokenizer vocabulary，也不会送进模型，所以无需
训练或修改 embedding；`full_input_ids` 与未插 marker 时完全相同，所有 token 的 absolute
position 也不变。tokenizer 调用次数与 message 数量无关，8、32、128 条 message 都是一次。

### Detailed boundary rules

The owner track and query-epoch track answer different questions but, for
`MessageDropRule`, both come from the one canonical stream:

- **Canonical owner track:** output rendered while a message object is active uses
  that message's public ID. Static text before the first traceable message belongs
  to `M0`; later unowned template text follows the preceding owner. The final
  generation prompt belongs to synthetic owner `A_N`.
- **Character-to-token projection:** if one token crosses a message boundary, its
  first character decides the owner—therefore a token that starts in the previous
  message is forcibly assigned to that previous message. A zero-width token also
  inherits the previous token's owner. This never expands a Drop into the next
  message.
- **Query epoch:** owner IDs must be monotonic and are used directly as epochs. A
  template that reorders messages is rejected because it cannot express ordered
  Drop events safely.
- **Drop ranges:** consecutive tokens with the same owner form a half-open
  `[start, end)` range. One message may own several non-contiguous ranges. Dropping
  that message removes every one of those ranges.

Only text emitted by the chat template enters either track. An omitted JSON field
or an empty message that renders nothing owns no tokens. Ownership is based on
rendered provenance rather than content matching, so repeated message text does
not make the boundary ambiguous.

### Qwen example

Qwen's
[ChatML-style template](https://huggingface.co/Qwen/Qwen-tokenizer/blob/main/tokenizer_config.json)
emits this block for each message:

```text
<|im_start|>{role}\n{content}<|im_end|>\n
```

With a generation prompt enabled, it then appends:

```text
<|im_start|>assistant\n
```

Consider this request:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is 12 + 30?"},
    {"role": "assistant", "content": "42."},
    {"role": "user", "content": "Then multiply it by 3."}
  ]
}
```

Using `\n` below to display each actual newline, the canonical render is divided
as follows:

```text
owner  epoch  rendered span
M0     0      <|im_start|>system\nYou are a helpful assistant<|im_end|>\n
M1     1      <|im_start|>user\nWhat is 12 + 30?<|im_end|>\n
M2     2      <|im_start|>assistant\n42.<|im_end|>\n
M3     3      <|im_start|>user\nThen multiply it by 3.<|im_end|>\n
A4     4      <|im_start|>assistant\n
```

Each `M0`-`M3` range includes its role header, exact content, end marker, and
trailing newline. `A4` contains only the final generation prompt and is not a
submitted message.

![Exact Qwen owner and query-epoch boundaries for four messages](assets/qwen-message-token-ownership.png)

The append effect is easiest to see one step earlier. With only `M0` and `M1`, the
render ends with generation prompt `A2`:

```text
A2  <|im_start|>assistant\n
```

When the assistant reply is appended as public message `M2`, the complete
conversation is rendered again. The same visible assistant header is now the
start of `M2`, followed by `42.<|im_end|>\n`, and a new `A3` generation prompt is
added. `A2` and `M2` both use numeric owner ID `2` by design. Appending `M3` then
produces the final `A4` prompt shown above.

This is why a new message cannot be represented by independently tokenizing only
its content and appending those tokens. Mini-SGLang instead renders the complete
conversation, traces the template's native message loop, and encodes that one
canonical stream once.

### GPT-OSS Harmony boundary

GPT-OSS 不走 Hugging Face Jinja marker 路径。`MessageDropRule` 调用
`render_conversation_for_completion` 一次，并直接读取 Harmony 原生 `<|start|>` 与
`<|end|>`/`<|call|>`/`<|return|>` protocol boundaries。Harmony 会把多条
system/developer 指令合并到一个 component；系统在该 component 内按 UTF-8 byte 精确寻找原始
指令，再用“token 的第一个 byte 决定 owner”投影。这样既保留 Harmony 原生控制 token，也不
需要逐 message prefix render。

## 三种 Drop Rule

公开接口统一使用 `drop_rule.type` 判别规则。三个公开 Python 类分别是
`MessageDropRule`、`TextDropRule` 和 `ThinkingDropRule`。旧顶层 `drop_message`
仍可兼容输入，并在入口转换成 `MessageDropRule`；同一请求同时给出新旧接口会返回
HTTP 400。

### MessageDropRule：按完整 message 删除

接口名是 `message_drop`，字段仍叫 `drop_messages`，不使用 `schedule`：

```json
"drop_rule": {
  "type": "message_drop",
  "drop_messages": {"3": [1, 2]}
}
```

事件 `n` 在 message `n` 完成后生效；目标 epoch `t` 的已删除集合是所有 `n < t`
事件的并集。每个 message ID 对应完整模板 provenance owner，因此 role header、content、
结束符和属于该 message 的模板分隔符一起删除。事件和 ID 必须是非负 signed-int64，事件
不能引用未来 message；尚未发生的未来事件可以保留在请求中。

### TextDropRule：按原始 content 子串删除

接口名是 `text_drop`。`drop_messages` 必须与 `messages` 等长、同序且逐项 role 相同；
用户不提供 message ID。某项不删除时，`content` 可写 `null`、`""`、`[]` 或
`[""]`：

```json
"drop_rule": {
  "type": "text_drop",
  "drop_messages": [
    {"role": "system", "content": null},
    {"role": "user", "content": "temporary code"},
    {"role": "assistant", "content": ["Noted", "briefly"]},
    {"role": "user", "content": ""}
  ]
}
```

`content` 可为 `str` 或 `list[str]`，从而在一条 message 内选择不连续片段。每个非空
selector 必须是对应原始 `messages[i].content` 的大小写敏感精确子串，不做 Unicode
normalization；否则返回 HTTP 400。匹配使用 UTF-8 Aho-Corasick O(n) AOT CPU kernel，
同时保留等价 Python reference/fallback 和输入、pattern、输出数量上限。

重复子串默认选择从左到右第 1 次、允许重叠。用户也可提供 `occurrence`：单个字符串配
正整数，字符串列表必须配等长 `list[int]`，不能只给部分 selector 编号。例如：

```json
{"role": "user", "content": ["aba", "answer"], "occurrence": [2, 1]}
```

原始字符区间先映射到完整 chat-template render 中该 message 的唯一 content span，再通过
fast tokenizer 的 canonical `offset_mapping` 映射到绝对 token 区间。只有完全包含在字符
区间内的 token 才删除；横跨边界的 token 保留。如果 selector 没有覆盖任何完整 token，
请求失败而不是扩大删除范围。若一条 message 的 selector 并集恰好覆盖完整原始 content，
系统提升为完整 owner ranges，使它与相同 message ID 的 `MessageDropRule` 精确等价。该规则
固定在最新 user Prefill 完成后生效。

### ThinkingDropRule：保留文本、删除 thinking KV

接口名是 `thinking_drop`：

```json
"drop_rule": {"type": "thinking_drop"}
```

未开启时完全沿用模型 tokenizer/chat template，不做能力探测、模板覆盖或 thinking 解析。
开启后，thinking 必须来自 assistant 的 `reasoning_content`，或 content 中唯一、完整、非嵌套
且位于开头的 `<think>...</think>`。两种来源不得同时提供；双来源、标签残缺或有歧义时均
返回 HTTP 400，系统不会从普通 prose 猜测 thinking。

系统仅在该规则开启时做惰性 retention probe，并按 tokenizer 身份、模板 SHA、tools variant
缓存结果。原生保留历史 thinking 的模板不修改；已识别的 Qwen 历史过滤 guard 使用请求级
Jinja adapter，私有 `preserve_thinking_history` 只对当前请求生效，且必须通过“实际 reasoning
出现一次、无 reasoning 输出不变”的后置校验。未知的过滤模板 fail closed，返回
`thinking_history_not_preservable`。GPT-OSS 仅在该规则开启时用
`RenderConversationConfig(auto_drop_analysis=False)` 保留原生 Harmony `analysis` component，
逐 component 校验 token cursor，只删除 analysis 内容 token；channel/recipient/protocol
token 保留。规则未开启时仍沿用 Harmony 默认的 `auto_drop_analysis` 行为。

每段 thinking 的 Drop event 位于对应 assistant message 末尾。因此 assistant final token 可见
同轮 thinking，后续 user/tool/assistant-generation 不可见。冷缓存 `mask` warmup 仍输入完整
token timeline，以 `full_token_visible_until` 保证先计算 thinking/final，再在事件处删除其
active KV。

### Token and cache behavior

Each newly effective event is compiled to sorted, merged, half-open absolute token
ranges. The surviving tokens keep their original absolute positions; they are not
renumbered after gaps are removed.

The `delta-marker` Radix mode inserts one virtual negative marker at each event
boundary. The marker identifies the event's newly dropped position ranges but owns
no KV page. Consequently, identical token text with different Drop histories
branches to different Radix paths. Cache matching first follows the full token and
marker path, then filters matched KV entries to the positions active for the
request.

Drop changes later visibility. It does not retroactively recompute surviving KV
states from epochs at or before the event, or expand the model's maximum absolute
context length. Later message tokens are prefilled under the new visibility.

## Starting the server

默认启动已经启用 Radix Drop key、`mask` prefill、自动 attention backend 和 CUDA Graph。
例如卡 7 的 GPT-OSS 可以直接启动，不需要额外 Context 参数：

```bash
CUDA_VISIBLE_DEVICES=7 python -m minisgl \
  --model /share/wangruoxi/models/gpt-oss-20b \
  --port 8000
```

公开端口始终严格使用 `--port`。内部 distributed 端口从 loopback 动态选择，不再假定
`public_port + 1`，因此其他进程占用 8001 不会阻止公开 8000 启动。系统在创建 worker 前检查
公开端口；正常退出、Ctrl-C、SIGTERM 与启动异常都会关闭 frontend ZMQ、停止 scheduler 和
tokenizer/detokenizer，按 terminate → bounded join → kill fallback 回收子进程及启动队列。

需要显式写出 Drop 配置时，等价命令是：

```bash
python -m minisgl \
  --model Qwen/Qwen3-0.6B \
  --cache-type radix \
  --page-size 1 \
  --radix-drop-key-mode delta-marker \
  --contextual-prefill-mode mask \
  --attn fi \
  --port 1919
```

CUDA Graph 对所有模型默认开启，并按 `main` 的 capture shapes、过程和输出捕获；显式
`--disable-cuda-graph` 或最大 batch size `0` 才关闭。

Relevant options:

| Option | Values and behavior |
| --- | --- |
| `--radix-drop-key-mode` | `delta-marker` is the default and the only mode that accepts a non-empty Drop Rule. `bitmask` and `symbol` are legacy no-Drop modes. |
| `--page-size` | Must be `1` with `delta-marker`. TRTLLM attention is incompatible because it requires non-unit pages. |
| `--contextual-prefill-mode` | `mask` (default) or `staged`. The old `flashinfer-mask` and `flashattention-mask` names are deprecated CLI aliases for `mask`. |
| `--attn` / `--attention-backend` | In `mask` mode, the selected Prefill backend compiles its native exact representation. FlashInfer and FlashAttention are supported; unsupported backends fail during startup. |
| `--cache-type` | Use `radix` to reuse full token and Drop-history prefixes. |
| `--request-timeout` | Seconds without a reply before timeout; default `300`, or `0` to disable. |

`mask` sends one full-token-timeline warmup carrying the backend-neutral
`full_token_visible_until` metadata. FlashInfer compiles it into exact active-KV
segments. FlashAttention uses the FA3 segmented adapter on SM80/SM90 or the FA4
`mask_mod` adapter on SM100/SM110. The Scheduler validates the actual Prefill
backend instead of requiring the frontend mode to name that backend.

`staged` remains available for compatibility and diagnosis. It first measures the
internal active-prefix cache reuse. A `cache_reuse_ratio` greater than or equal to
`0.95` ends warmup; only a strictly lower ratio triggers per-message prefix
warmups. This internal value is separate from the user-visible `cache_hit_ratio`.

Known limitations remain backend-specific. Context-mask Prefill requires
`page_size=1`. FA4 mask Prefill is restricted to one request per batch, and its
GPT-OSS sinks/sliding-window combination is not enabled. A backend or GPU build
without the required adapter is rejected instead of silently running ordinary
causal attention.

For tied embeddings, a checkpoint may legally contain both
`model.embed_tokens.weight` and a redundant `lm_head.weight`. The loader preserves
template-specific dtypes for known model tensors (including GPT-OSS packed
`uint8`), while passing template-external keys to the strict model loader. The
tied LM head consumes its redundant alias; unrelated unknown keys remain errors.

## Chat API

Drop is exposed through the OpenAI-style endpoint:

```text
POST /v1/chat/completions
```

Minimal request:

```bash
curl http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "system", "content": "Answer briefly."},
      {"role": "user", "content": "My temporary code is 4815."},
      {"role": "assistant", "content": "Noted."},
      {"role": "user", "content": "Continue without using the temporary code."}
    ],
    "drop_rule": {
      "type": "message_drop",
      "drop_messages": {"2": [1]}
    },
    "max_tokens": 64,
    "stream": false
  }'
```

The `messages` entries support these roles and fields:

```text
role: system | user | assistant | tool | function
content: string | null
reasoning_content: string | null
name: string | null
tool_call_id: string | null
tool_calls: array | null
```

The backend applies `max_tokens`, `temperature`, `top_k`, `top_p`, `ignore_eos`,
`stop`, and `stream`. The chat path also supports `enable_thinking`,
`reasoning_effort`, `tools`, and `tool_choice`. The schema currently accepts `n`,
`presence_penalty`, and `frequency_penalty`, but does not apply them to backend
sampling. `message_drop.drop_messages` 的 JSON object key 在 wire 上是字符串，并解析为
整数事件 ID。

Drop Rule 只支持 chat `messages`，不支持 plain `prompt`。结构、子串、occurrence、thinking
来源或模板能力校验失败均返回 HTTP 400。非流式成功响应在顶层返回
`cache_hit_ratio`；流式响应只在最后一个带 `finish_reason` 的 SSE chunk 顶层返回该字段。
它是最终用户生成请求的 `cached_active_len / full_matchable_prefix_len`：分子是命中的未
Drop token，分母是 Drop 前全部可参与 Radix 匹配的真实 token，不包含最后一个强制 Prefill
token 和 virtual marker；空分母为 1.0。内部 warmup 使用
`cached_active_len / active_matchable_prefix_len`，不会暴露为用户结果。

## R10 validation and performance snapshot

R10 的 MessageDrop 边界实现已经在 Qwen/Harmony 单元测试和真实服务中验证：8、32、128 条
messages 都只触发一次 canonical tokenizer encode，且 encoded IDs 与原生 chat template
完全一致。GPU 服务验证覆盖了默认启动 `gpt-oss-20b`、Qwen3 tied、Qwen3 untied、Qwen3 MoE
和 `gpt-oss-120b --tp 4`；这些模型都默认进入 CUDA Graph capture，MessageDrop 请求返回
HTTP 200，并在 SIGTERM 后完成 scheduler shutdown 与 worker 回收。

在 tau2-airline 的一个长上下文问题上，Qwen3-8B、默认设置、流式 `max_tokens=64` 的 median
结果如下；System-test 使用 `MessageDropRule {"6": [2, 3]}`，main 不带 Drop：

| System | Scenario | E2E ms | TTFT ms | Decode ms | cache_hit_ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| `origin/main` | sequential cold | 908.53 | 212.54 | 696.04 | n/a |
| `System-test mask` | sequential cold | 984.02 | 282.27 | 701.68 | 0.9693 |
| `System-test staged` | sequential cold | 1167.78 | 466.12 | 701.26 | 0.9686 |
| `origin/main` | batch size 3 | 1345.22 | 415.26 | 932.48 | n/a |
| `System-test mask` | batch size 3 | 1443.87 | 709.21 | 736.49 | 0.9693 |
| `System-test staged` | batch size 3 | 1841.47 | 1090.54 | 741.35 | 0.9686 |

CPU tokenizer microbench 也验证了单次 encode 的常数倍开销：8/32/128 条 messages 下，
MessageDrop provenance 相对原生 `apply_chat_template` 的 median 分别约为
`3.49x`、`3.45x`、`3.67x`，同时保持 exact IDs。

## Implementation map

| Concern | Source |
| --- | --- |
| Public schema and validation | [`drop_rules.py`](../../python/minisgl/tokenizer/drop_rules.py#L60), [`api_server.py`](../../python/minisgl/server/api_server.py#L161) |
| MessageDrop one-encode ownership | [`tokenize.py`](../../python/minisgl/tokenizer/tokenize.py#L1396), [`template_provenance.py`](../../python/minisgl/tokenizer/template_provenance.py#L24), [`template_provenance.py`](../../python/minisgl/tokenizer/template_provenance.py#L206) |
| GPT-OSS native Harmony ownership | [`tokenize.py`](../../python/minisgl/tokenizer/tokenize.py#L375) |
| Text matcher | [`text_match.py`](../../python/minisgl/kernel/text_match.py#L127), [`text_match.cpp`](../../python/minisgl/kernel/csrc/src/text_match.cpp#L37) |
| Thinking retention adapter | [`thinking_template.py`](../../python/minisgl/tokenizer/thinking_template.py#L86), [`tokenize.py`](../../python/minisgl/tokenizer/tokenize.py#L273) |
| Position-range Drop compilation and absolute positions | [`tokenize.py`](../../python/minisgl/tokenizer/tokenize.py#L923), [`tokenize.py`](../../python/minisgl/tokenizer/tokenize.py#L1268), [`tokenize.py`](../../python/minisgl/tokenizer/tokenize.py#L1602) |
| Virtual delta markers | [`radix_delta.py`](../../python/minisgl/scheduler/radix_delta.py#L12-L261) |
| Full-path cache match and active-KV filtering | [`cache.py`](../../python/minisgl/scheduler/cache.py#L121-L200) |
| Dynamic internal port and worker lifecycle | [`args.py`](../../python/minisgl/server/args.py#L14), [`launch.py`](../../python/minisgl/server/launch.py#L23), [`launch.py`](../../python/minisgl/server/launch.py#L81), [`api_server.py`](../../python/minisgl/server/api_server.py#L1280) |
| Startup and backend constraints | [`args.py`](../../python/minisgl/server/args.py#L58), [`scheduler.py`](../../python/minisgl/scheduler/scheduler.py#L55-L131) |
| Contextual warmup | [`api_server.py`](../../python/minisgl/server/api_server.py#L755), [`prefill.py`](../../python/minisgl/scheduler/prefill.py#L67) |
| Cache-hit ratio response propagation | [`scheduler.py`](../../python/minisgl/scheduler/scheduler.py#L255), [`tokenizer/server.py`](../../python/minisgl/tokenizer/server.py#L121), [`api_server.py`](../../python/minisgl/server/api_server.py#L953) |
| Tied-weight checkpoint loading | [`engine.py`](../../python/minisgl/engine/engine.py#L159-L177), [`embedding.py`](../../python/minisgl/layers/embedding.py#L59-L85), [`base.py`](../../python/minisgl/layers/base.py#L32-L53) |
