# OpenAI API 中转站综合检测器

这是一个针对 OpenAI Chat Completions 兼容接口的轻量检测工具。它保留了原项目的模型路由一致性检查，并增加能力、协议、计费和长上下文探针。

## 可以检测什么

程序开始前默认会先调用 `GET /v1/models` 做认证和模型预检查。若返回 401/403/404，会在发送其他探针前直接给出原因并停止，避免无效请求和额外费用。若服务不实现 `/v1/models`，可使用 `--skip-models-check` 跳过这一步。

| 检测项 | 内容 |
| --- | --- |
| `basic_request` | 接口是否能完成最小 Chat Completions 请求，检查 `choices/message` 结构 |
| `model_consistency` | 返回的 `model` 是否与请求模型完全一致 |
| `protocol` | `id`、`object`、`choices`、`index`、`finish_reason` 等字段是否符合基本协议 |
| `token_usage` | `prompt_tokens`、`completion_tokens`、`total_tokens` 是否完整且数值自洽 |
| `function_calling` | `tools`、`tool_choice`、`tool_calls` 和 JSON 参数是否正确透传 |
| `structured_output` | `response_format=json_schema` 是否返回符合 strict schema 的 JSON |
| `streaming` | SSE `data` 事件、增量内容和 `[DONE]` 结束事件是否正常 |
| `long_context` | full 模式下，在较长输入中插入校验词并验证模型能否找回 |

这类检测能够发现中转站换模型、剥离高级能力、包装错误响应或篡改 usage 字段等问题。OpenAI 协议本身没有 Claude thinking signature 那样的加密身份凭证，因此结果是协议和行为证据，不是数学意义上的官方身份认证。

若服务端返回 `HTTP 503` 且明确说明 `No available channel`，报告会标为“不可用”而不是“能力失败”：该模型不评分，也不计入可用模型的平均分，但本次检测仍会返回非零退出码，提醒你它目前不能正常使用。

## 安装

```cmd
python -m pip install -r requirements.txt
```

## 快速使用

API Key 建议放在环境变量中，不要写入脚本、配置文件或日志：

```cmd
set API_KEY=你的真实API_KEY
python model_route_consistency.py --api-url "https://example.com/v1/chat/completions" --model "gpt-4o" --mode quick
```

PowerShell：

```powershell
$env:API_KEY = "你的真实API_KEY"
python .\model_route_consistency.py --api-url "https://example.com/v1/chat/completions" --model "gpt-4o" --mode standard
```

## 三档模式

```cmd
python model_route_consistency.py --api-url URL --model gpt-4o --mode quick
python model_route_consistency.py --api-url URL --model gpt-4o --mode standard
python model_route_consistency.py --api-url URL --model gpt-4o --mode full --context-chars 12000
```

- `quick`：基础请求 + 模型一致性，成本最低。
- `standard`：增加协议、Token 用量、Function Calling、结构化输出和流式响应。
- `full`：在 standard 基础上增加长上下文找回测试；输入越长，消耗的 Token 越多。

多个模型可以重复使用 `--model`，也可以逗号分隔；`--rounds 2` 会对每个模型完整执行两轮。

## JSON 报告和 CI

```cmd
python model_route_consistency.py --api-url URL --model gpt-4o --mode standard --output-format json -o report.json --no-color
```

JSON 报告包含每项检测的状态、HTTP 状态码、耗时、错误说明、模型分数和总体 verdict。退出码：`0` 全部通过，`1` 存在失败，`130` 用户中断。

## 配置文件

复制 `config.example.json` 为 `config.json` 后可集中保存 URL、模型、模式、超时、重试和输出设置：

```cmd
set API_KEY=你的真实API_KEY
python model_route_consistency.py --config config.json
```

命令行参数优先于配置文件。配置文件不包含 API Key；日志只记录检测摘要，不记录认证头或完整响应正文。

## 其他选项

- `--timeout 30`：单次请求超时秒数。
- `--retries 1`：连接错误及 429/5xx 的重试次数；可能导致上游重复计费，请按需设置为 `0`。
- `--fail-fast`：第一次失败后停止。
- `--no-env-proxy`：忽略系统代理环境变量。
- `--skip-models-check`：跳过开始前的 `/v1/models` 预检查。
- `--no-verify`：关闭 SSL 校验，仅建议用于临时排查证书问题。

## 安全提醒

只检测自己拥有或明确获授权的 API。不要把 API Key 提交到 Git、截图或问题报告中；full 模式和多轮测试会产生实际 API 用量。

## 测试

项目内置不依赖外部 API 的单元测试：

```cmd
python -m unittest discover -s tests -v
```
