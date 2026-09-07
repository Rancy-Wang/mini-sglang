# Qwen 工具调用 JSON 验证（R8）

## 修复边界

R8 不通过补引号、重试或事后改写模型文本来掩盖错误。对于
AgenticQwen/Qwen3（不含 Qwen3-Coder）的有效工具请求，tokenizer 生成可序列化的
grammar descriptor，描述可选工具、`auto` / `required` / 指定函数、结构标签版本和
generation prompt 是否已经打开 `<think>`。

无工具、`tool_choice=none`、warmup、模板降级 safe mode 及不兼容模型都不生成
descriptor。普通请求因此不会初始化或等待 grammar matcher。

采样器使用 XGrammar 的 Qwen3 structural tag，在 GPU logits 上只屏蔽当前请求语法状态
不允许的 token。编译结果按 tokenizer、词表、stop token 和完整 descriptor 缓存；
matcher 按 `Req` 对象隔离。CPU matcher 接受已经提交的 token，并为下一次采样准备
bitmask；批次换序按 `Req` 身份重新定位，完成、取消或 UID 复用时释放 matcher。

`ChunkedReq` 的 padding sample 不推进 matcher；普通 Prefill 产生的第一个正式生成
token 只推进一次。不存在全局 token 黑名单或硬编码 token ID。

依赖固定为 `xgrammar==0.2.5.post1`，该版本要求
`apache-tvm-ffi>=0.1.9`。grammar 要求完整 JSON schema、合法转义、Unicode、嵌套
对象/数组以及闭合 tool marker 后才允许 stop token。若 `max_tokens` 先耗尽，服务仍
返回 `finish_reason=length`，parser 保留未完成原文，不把它伪造成完整
`tool_calls`。

## Parser 保真不变量

`_QwenToolStream` 同时服务完整响应与流式响应。只有完整、可解析且名称已知的调用才会
进入 `tool_calls`。非法、未知或未完成的块保留在 `content`，并产生内部
`ToolParseDiagnostic`。

因此，生成期 grammar 负责防止支持范围内的新非法 JSON；parser 仍负责忠实表示模型已经
生成的内容。这两个职责互不替代。

## 已知 BCP 故障结果

2026-09-07 在 AgenticQwen-30B-A3B、A800 80GB、BF16、TP1、生产 FlashAttention 和
开启 CUDA Graph 的环境中验证。生产 grammar 修复位于
`50cef9ac0cd443b6e9620f8746fc13b937017ad9`；最终 HTTP 验证脚本位于
`e5d3bb0506605df4a6d418fb76392f832ccf0c81`。冻结输入 gzip SHA256 为
`cfd6f2ed4246cab31c6bb0729b2b63e48bcd02bef372745faafae0d0e114ff8c`。

| 检查 | 结果 |
|---|---|
| 旧 logits 首次拒绝点 | 822 在 step 32 拒绝原 token 11248；844 在 step 30 拒绝原 token 11248 |
| 已知失败请求 | 822、844 都生成标准 JSON 可解码的 `search` 参数，原缺闭合引号错误未再出现 |
| grammar 与 graph | 两个 case 共用一个编译结果；无工具控制不初始化 grammar；CUDA Graph 保持开启 |
| 真实 HTTP | cold priming、相同 exact-Radix 热路径 full 和热路径 stream 均产生可解码对象 |
| 工具选择 | `required` 和指定函数均通过 |
| 截断 | `max_tokens=1` 返回 `finish_reason=length` 和原始未完成块，没有伪造调用 |

最终 HTTP full/stream 对照为 PASS。一次诊断中 cold 与 hot 的 844 查询文字不同，但都
是合法 JSON；这属于不同 prefill/Radix 数值路径，不能当作流式组装差异。

## 当前可复用验证

CPU 单元测试覆盖 grammar 编译共享、matcher 隔离、批次换序、取消/UID 复用、无工具零
初始化、完整/流式 parser 一致性、Unicode、转义、嵌套 JSON、多调用和长度截断：

```bash
python -B -m pytest -o addopts= -p no:cacheprovider -q \
  tests/engine/test_tool_grammar.py \
  tests/server/test_tool_json_generation.py \
  tests/server/test_tool_protocol.py
```

真实模型复测入口保留在
`tests/contextual/mask_staged_runner.py` 的 `MINISGL_R8_SUITE=tool_json_generation`
路径。该入口验证默认 mask 生产路径，不再依赖曾用于 R7/R9 对照的私有 staged-reference
生产脚手架。

本结论只保证 grammar 支持范围内的 AgenticQwen/Qwen3 有效工具请求具有结构合法性；它
不保证参数事实正确，也不把截断、不兼容模型或 safe-mode 请求伪装成成功调用。
