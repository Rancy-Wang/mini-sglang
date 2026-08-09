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

The general partitioning flow is:

1. Keep the submitted messages in order and preserve their public IDs through
   request normalization.
2. For each message index `i`, render the complete prefix `messages[:i+1]` with
   the selected chat template and without a generation prompt. Never tokenize
   messages independently and concatenate the results.
3. Compare each prefix render with the previous one by exact token ID. The longest
   common prefix identifies the part whose history is unchanged; a non-overlapping
   longest common suffix is also recorded for conservative owner attribution.
4. Render the complete message list again with the generation prompt. This is the
   canonical full token stream, and the prompt uses the synthetic next-assistant
   ID `N`, where `N = len(messages)`.
5. Derive final message ownership from the canonical render when exact rendered
   provenance is available, tokenize that render once, and project the rendered
   ownership spans onto token offsets.

### Detailed boundary rules

The owner track and query-epoch track answer different questions and are built
with different rules:

- **Provisional owner track during prefix comparison:** exact stable prefix and
  suffix tokens retain their previous owners. Tokens in the rewritten middle are
  assigned to the newly appended message. This is a conservative fallback for a
  template that rewrites earlier output.
- **Query-epoch track:** only the exact stable prefix retains its earlier epoch.
  Every token from the first changed position onward receives the new message
  epoch, including a suffix whose token values happen to match again. This keeps
  the epoch sequence monotonic.
- **Canonical owner track:** rendered output traced to a message object uses that
  message's public ID. Static text before the first traceable message belongs to
  `M0`; unowned template text after that follows the preceding owner. The final
  generation prompt belongs to synthetic owner `A_N`.
- **Character-to-token projection:** the canonical render is tokenized once with
  character offsets. If one token crosses an ownership boundary, its first
  character decides the owner. A zero-width token inherits the previous token's
  owner.
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
its content and appending those tokens. The template may extend or rewrite its
previous suffix, so Mini-SGLang re-renders the conversation and recomputes the
boundary tracks from the resulting full stream.

## Drop semantics

`drop_message` is a schedule:

```text
drop_message[event_message_id] = [message_id, ...]
```

Event `n` takes effect **after message `n`**. It affects later messages and the
following generation prompt, not message `n` itself. At target epoch `t`, the
active dropped set is:

```text
union(drop_message[n] for every n < t)
```

Rules:

- Event IDs and message IDs are non-negative signed 64-bit integers.
- An event may only drop messages whose IDs are less than or equal to the event
  ID; it cannot drop a future message.
- Events are cumulative. Repeating an already dropped ID has no additional
  effect.
- Future events are accepted by validation. An event whose ID is not earlier than
  the current target epoch has no effect on the current output.
- An empty or omitted `drop_message` follows the normal no-Drop path.

Example:

```json
{
  "messages": [
    {"role": "system", "content": "Answer briefly."},
    {"role": "user", "content": "My temporary code is 4815."},
    {"role": "assistant", "content": "Noted."},
    {"role": "user", "content": "Continue without using the temporary code."}
  ],
  "drop_message": {"2": [1]}
}
```

Message `1` is visible while message `2` is produced. After message `2`, its
rendered token ranges are hidden from message `3` and from the next assistant
generation.

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

A conservative Drop-enabled configuration is:

```bash
python -m minisgl \
  --model Qwen/Qwen3-0.6B \
  --cache-type radix \
  --page-size 1 \
  --radix-drop-key-mode delta-marker \
  --contextual-prefill-mode staged \
  --attn fa \
  --port 1919
```

Relevant options:

| Option | Values and behavior |
| --- | --- |
| `--radix-drop-key-mode` | `delta-marker` is the default and the only mode that accepts a non-empty `drop_message`. `bitmask` and `symbol` are legacy no-Drop modes. |
| `--page-size` | Must be `1` with `delta-marker`. TRTLLM attention is incompatible because it requires non-unit pages. |
| `--contextual-prefill-mode` | `staged` (default), `flashinfer-mask`, or `flashattention-mask`. |
| `--attn` / `--attention-backend` | Mask modes require the matching prefill backend. FlashAttention mask prefill supports FA3 on SM80/SM90 and FA4 on SM100/SM110. |
| `--cache-type` | Use `radix` to reuse full token and Drop-history prefixes. |
| `--request-timeout` | Seconds without a reply before timeout; default `300`, or `0` to disable. |

`staged` first measures the active-prefix cache hit. If it is below `0.95`, the
server warms the conversation by message prefixes. The two mask modes instead
prefill the full token timeline under the exact Drop visibility mask.

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
    "drop_message": {"2": [1]},
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
sampling. JSON object keys in `drop_message` are strings on the wire and are
parsed as integer event IDs.

`drop_message` is supported only with chat `messages`, not with a plain `prompt`.
Invalid schedules return HTTP `400`. Responses use the existing OpenAI-compatible
chat completion or server-sent event format.

## Implementation map

| Concern | Source |
| --- | --- |
| Public schema and validation | [`api_server.py`](../../python/minisgl/server/api_server.py#L86-L145), [`OpenAICompletionRequest`](../../python/minisgl/server/api_server.py#L584-L617) |
| Template-independent epochs and token ownership | [`tokenize.py`](../../python/minisgl/tokenizer/tokenize.py#L745-L965), [`template_provenance.py`](../../python/minisgl/tokenizer/template_provenance.py#L23-L272) |
| Position-range Drop compilation | [`tokenize.py`](../../python/minisgl/tokenizer/tokenize.py#L487-L628) |
| Virtual delta markers | [`radix_delta.py`](../../python/minisgl/scheduler/radix_delta.py#L12-L261) |
| Full-path cache match and active-KV filtering | [`cache.py`](../../python/minisgl/scheduler/cache.py#L103-L179) |
| Startup and backend constraints | [`args.py`](../../python/minisgl/server/args.py#L176-L265), [`scheduler.py`](../../python/minisgl/scheduler/scheduler.py#L55-L131) |
| Contextual warmup | [`api_server.py`](../../python/minisgl/server/api_server.py#L721-L777), [`prefill.py`](../../python/minisgl/scheduler/prefill.py#L47-L230) |
