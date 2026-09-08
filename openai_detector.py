#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI Chat Completions 中转站综合检测引擎。

本模块只依赖 ``requests``，不保存 API Key，也不记录上游响应正文。检测器
使用公开的 OpenAI Chat Completions 协议构造探针，并把每项检查转换成可供
CLI、CI 或 Web 前端消费的结构化结果。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import requests


RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
MAX_ERROR_LENGTH = 240

# 权重只用于报告分数，不代表官方认证或模型能力的绝对结论。
CHECK_WEIGHTS: dict[str, float] = {
    "basic_request": 20.0,
    "model_consistency": 20.0,
    "protocol": 15.0,
    "token_usage": 10.0,
    "function_calling": 15.0,
    "structured_output": 10.0,
    "streaming": 10.0,
    "long_context": 10.0,
}

MODE_CHECKS: dict[str, tuple[str, ...]] = {
    "quick": ("basic_request", "model_consistency"),
    "standard": (
        "basic_request",
        "model_consistency",
        "protocol",
        "token_usage",
        "function_calling",
        "structured_output",
        "streaming",
    ),
    "full": (
        "basic_request",
        "model_consistency",
        "protocol",
        "token_usage",
        "function_calling",
        "structured_output",
        "streaming",
        "long_context",
    ),
}


@dataclass
class ProbeResponse:
    """一次 HTTP 探针的最小安全结果。"""

    status_code: Optional[int] = None
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    elapsed_ms: Optional[float] = None
    stream_events: list[dict[str, Any]] = field(default_factory=list)
    stream_done: bool = False


@dataclass
class CheckResult:
    """一个检测器的结果。"""

    name: str
    category: str
    passed: bool
    status: str = "pass"  # pass / fail / skip / unavailable
    message: str = ""
    round_number: int = 1
    elapsed_ms: Optional[float] = None
    status_code: Optional[int] = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelSummary:
    model: str
    score: Optional[float]
    verdict: str
    passed: int
    failed: int
    skipped: int
    unavailable: int
    total: int
    checks: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _truncate(value: Any, limit: int = MAX_ERROR_LENGTH) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_transient_status(status_code: Optional[int]) -> bool:
    return status_code in RETRY_STATUS_CODES


def classify_http_error(status_code: int, data: Any = None, body: str = "") -> str:
    """把常见 HTTP 错误转换成用户可操作的中文诊断。"""
    message = ""
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or error.get("type") or "")
        elif error:
            message = str(error)
    if not message:
        message = _truncate(body)
    suffix = f" {_truncate(message)}" if message else ""
    if status_code == 401:
        return f"认证失败（HTTP 401）：API Key 无效、已过期，或不是该服务签发的令牌。{suffix}"
    if status_code == 403:
        return f"权限不足（HTTP 403）：API Key 有效，但没有访问该接口或模型的权限。{suffix}"
    if status_code == 404:
        return f"接口或模型不存在（HTTP 404）：请检查 Base URL 是否包含 /v1。{suffix}"
    if status_code == 429:
        return f"请求过于频繁或额度不足（HTTP 429）。{suffix}"
    if status_code == 503 and "no available channel" in message.lower():
        return f"当前不可用（HTTP 503）：服务端没有可用通道。{suffix}"
    return f"HTTP {status_code}{(': ' + _truncate(message)) if message else ''}"


def _sleep_before_retry(backoff: float, attempt: int) -> None:
    if backoff > 0:
        time.sleep(backoff * (2**attempt))


class OpenAIProbeClient:
    """封装请求、超时、代理和有限重试。"""

    def __init__(
        self,
        session: requests.Session,
        api_url: str,
        api_key: str,
        *,
        proxies: Optional[dict[str, str]] = None,
        verify: bool = True,
        timeout: float = 30.0,
        retries: int = 0,
        retry_backoff: float = 0.5,
    ) -> None:
        self.session = session
        self.api_url = api_url
        self.api_key = api_key
        self.proxies = proxies or None
        self.verify = verify
        self.timeout = timeout
        self.retries = max(0, retries)
        self.retry_backoff = max(0.0, retry_backoff)

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "miniature-fiesta-openai-detector/1.0",
        }

    def list_models(self) -> ProbeResponse:
        """执行 GET /models 预检查。"""
        models_url = self.api_url.rstrip("/")
        if models_url.endswith("/chat/completions"):
            models_url = models_url[: -len("/chat/completions")] + "/models"
        elif not models_url.endswith("/models"):
            models_url += "/models"
        started = time.perf_counter()
        response: Optional[requests.Response] = None
        try:
            response = self.session.get(models_url, headers=self.headers, proxies=self.proxies, verify=self.verify, timeout=self.timeout)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            try:
                data = response.json()
            except ValueError as exc:
                return ProbeResponse(status_code=response.status_code, error=f"返回内容不是有效 JSON：{_truncate(exc)}", elapsed_ms=elapsed_ms)
            if not 200 <= response.status_code < 300:
                return ProbeResponse(status_code=response.status_code, data=data if isinstance(data, dict) else None, error=classify_http_error(response.status_code, data, response.text), elapsed_ms=elapsed_ms)
            if not isinstance(data, dict):
                return ProbeResponse(status_code=response.status_code, error="/v1/models 返回 JSON 不是对象。", elapsed_ms=elapsed_ms)
            return ProbeResponse(status_code=response.status_code, data=data, elapsed_ms=elapsed_ms)
        except requests.RequestException as exc:
            return ProbeResponse(error=f"预检查请求失败：{_truncate(str(exc).replace(models_url, '<MODELS_URL>'))}")
        finally:
            if response is not None:
                response.close()

    def post(self, payload: dict[str, Any], *, stream: bool = False) -> ProbeResponse:
        """发起 POST；错误消息只保留状态和截断后的响应片段。"""
        last_error: Optional[str] = None
        last_status: Optional[int] = None

        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            response: Optional[requests.Response] = None
            try:
                request_kwargs = {
                    "headers": self.headers,
                    "json": payload,
                    "proxies": self.proxies,
                    "verify": self.verify,
                    "timeout": self.timeout,
                }
                # 兼容旧版测试替身和 requests 风格客户端：只有流式请求才传 stream。
                if stream:
                    request_kwargs["stream"] = True
                response = self.session.post(self.api_url, **request_kwargs)
                last_status = response.status_code
                elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

                if not 200 <= response.status_code < 300:
                    try:
                        error_data = response.json()
                    except ValueError:
                        error_data = None
                    last_error = classify_http_error(response.status_code, error_data, response.text)
                    if _is_transient_status(response.status_code) and attempt < self.retries:
                        retry_after = response.headers.get("Retry-After")
                        try:
                            delay = max(0.0, min(float(retry_after), 30.0)) if retry_after else 0.0
                        except (TypeError, ValueError):
                            delay = 0.0
                        if delay:
                            time.sleep(delay)
                        else:
                            _sleep_before_retry(self.retry_backoff, attempt)
                        if stream:
                            response.close()
                        continue
                    if stream:
                        response.close()
                    return ProbeResponse(
                        status_code=response.status_code,
                        error=last_error,
                        elapsed_ms=elapsed_ms,
                    )

                if stream:
                    return self._read_stream(response, elapsed_ms)

                try:
                    data = response.json()
                except ValueError as exc:
                    return ProbeResponse(
                        status_code=response.status_code,
                        error=f"返回内容不是有效 JSON：{_truncate(exc)}",
                        elapsed_ms=elapsed_ms,
                    )
                if not isinstance(data, dict):
                    return ProbeResponse(
                        status_code=response.status_code,
                        error="返回 JSON 不是对象。",
                        elapsed_ms=elapsed_ms,
                    )
                return ProbeResponse(
                    status_code=response.status_code,
                    data=data,
                    elapsed_ms=elapsed_ms,
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = f"请求失败：{_truncate(str(exc).replace(self.api_url, '<API_URL>'))}"
                if attempt < self.retries:
                    _sleep_before_retry(self.retry_backoff, attempt)
                    continue
                return ProbeResponse(
                    status_code=last_status,
                    error=last_error,
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
                )
            except requests.RequestException as exc:
                return ProbeResponse(
                    status_code=last_status,
                    error=f"请求失败：{_truncate(str(exc).replace(self.api_url, '<API_URL>'))}",
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
                )
            finally:
                # stream 模式由 _read_stream 读取完后关闭；非 stream 也显式释放连接。
                if response is not None and not stream:
                    response.close()

        return ProbeResponse(status_code=last_status, error=last_error or "请求失败。")

    @staticmethod
    def _read_stream(response: requests.Response, elapsed_ms: float) -> ProbeResponse:
        events: list[dict[str, Any]] = []
        saw_done = False
        malformed = 0
        try:
            for line in response.iter_lines(decode_unicode=True):
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                if not line or not line.startswith("data:"):
                    continue
                item = line[5:].strip()
                if item == "[DONE]":
                    saw_done = True
                    continue
                try:
                    parsed = json.loads(item)
                except (TypeError, ValueError):
                    malformed += 1
                    continue
                if isinstance(parsed, dict):
                    events.append(parsed)
        except requests.RequestException as exc:
            return ProbeResponse(
                status_code=response.status_code,
                error=f"读取流式响应失败：{_truncate(exc)}",
                elapsed_ms=elapsed_ms,
                stream_events=events,
                stream_done=saw_done,
            )
        finally:
            response.close()

        error = None
        if not events:
            error = "流式响应没有有效 data 事件。"
        elif not saw_done:
            error = "流式响应缺少 [DONE] 结束事件。"
        elif malformed:
            error = f"流式响应包含 {malformed} 个无法解析的事件。"
        return ProbeResponse(
            status_code=response.status_code,
            error=error,
            elapsed_ms=elapsed_ms,
            stream_events=events,
            stream_done=saw_done,
        )


def _check(
    name: str,
    category: str,
    passed: bool,
    message: str,
    *,
    round_number: int,
    response: Optional[ProbeResponse] = None,
    details: Optional[dict[str, Any]] = None,
) -> CheckResult:
    return CheckResult(
        name=name,
        category=category,
        passed=passed,
        status="pass" if passed else "fail",
        message=message,
        round_number=round_number,
        elapsed_ms=response.elapsed_ms if response else None,
        status_code=response.status_code if response else None,
        details=details or {},
    )


def _skip(name: str, category: str, message: str, round_number: int) -> CheckResult:
    return CheckResult(
        name=name,
        category=category,
        passed=False,
        status="skip",
        message=message,
        round_number=round_number,
    )


def _unavailable(
    name: str, category: str, message: str, round_number: int, response: Optional[ProbeResponse] = None
) -> CheckResult:
    return CheckResult(
        name=name,
        category=category,
        passed=False,
        status="unavailable",
        message=message,
        round_number=round_number,
        elapsed_ms=response.elapsed_ms if response else None,
        status_code=response.status_code if response else None,
        details={"availability": "unavailable"},
    )


def _message(data: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not isinstance(data, dict):
        return None
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    value = choices[0].get("message")
    return value if isinstance(value, dict) else None


def _message_text(message: Optional[dict[str, Any]]) -> str:
    if not message:
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                pieces.append(item["text"])
        return "".join(pieces)
    return ""


def _error_result(name: str, category: str, response: ProbeResponse, round_number: int) -> CheckResult:
    result = _check(
        name,
        category,
        False,
        response.error or "请求失败。",
        round_number=round_number,
        response=response,
    )
    if response.status_code == 503 and response.error and "当前不可用" in response.error:
        result.status = "unavailable"
        result.message = "当前不可用：服务端没有可用通道。"
        result.details["availability"] = "unavailable"
    return result


def check_basic(response: ProbeResponse, model: str, round_number: int) -> CheckResult:
    if response.error:
        return _error_result("basic_request", "基础", response, round_number)
    data = response.data
    if not isinstance(data, dict):
        return _check("basic_request", "基础", False, "响应不是 JSON 对象。", round_number=round_number, response=response)
    choices = data.get("choices")
    message = _message(data)
    valid = (
        isinstance(choices, list)
        and bool(choices)
        and message is not None
        and message.get("role") == "assistant"
        and (bool(_message_text(message)) or isinstance(message.get("tool_calls"), list))
    )
    return _check(
        "basic_request",
        "基础",
        valid,
        "基础 Chat Completions 请求成功。" if valid else "响应缺少有效 choices/message 结构。",
        round_number=round_number,
        response=response,
        details={"object": data.get("object"), "choice_count": len(choices) if isinstance(choices, list) else 0},
    )


def check_model_consistency(response: ProbeResponse, model: str, round_number: int) -> CheckResult:
    if response.error:
        return _error_result("model_consistency", "真伪", response, round_number)
    actual_value = response.data.get("model") if response.data else None
    actual = str(actual_value) if actual_value is not None else None
    return _check(
        "model_consistency",
        "真伪",
        actual == model,
        "返回模型与请求一致。" if actual == model else f"模型不一致：期望 {model}，实际 {actual or '<缺失>'}。",
        round_number=round_number,
        response=response,
        details={"expected": model, "actual": actual},
    )


def check_protocol(response: ProbeResponse, round_number: int) -> CheckResult:
    if response.error:
        return _error_result("protocol", "协议", response, round_number)
    data = response.data or {}
    choices = data.get("choices")
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    problems: list[str] = []
    if not isinstance(data.get("id"), str) or not data["id"]:
        problems.append("缺少字符串 id")
    if data.get("object") not in (None, "chat.completion"):
        problems.append("object 不是 chat.completion")
    if not isinstance(choices, list) or not choices:
        problems.append("choices 为空")
    if first is not None:
        if not isinstance(first.get("index"), int):
            problems.append("choice.index 不是整数")
        if first.get("finish_reason") is not None and not isinstance(first.get("finish_reason"), str):
            problems.append("finish_reason 类型错误")
        if not isinstance(first.get("message"), dict):
            problems.append("缺少 message")
    passed = not problems
    return _check(
        "protocol",
        "协议",
        passed,
        "响应字段符合 Chat Completions 基本协议。" if passed else "；".join(problems),
        round_number=round_number,
        response=response,
        details={"problems": problems},
    )


def check_token_usage(response: ProbeResponse, round_number: int) -> CheckResult:
    if response.error:
        return _error_result("token_usage", "计费", response, round_number)
    usage = (response.data or {}).get("usage")
    if not isinstance(usage, dict):
        return _check("token_usage", "计费", False, "响应缺少 usage 对象。", round_number=round_number, response=response)
    required = ("prompt_tokens", "completion_tokens", "total_tokens")
    problems: list[str] = []
    values: dict[str, int] = {}
    for key in required:
        value = usage.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            problems.append(f"{key} 不是非负整数")
        else:
            values[key] = value
    if len(values) == 3 and values["total_tokens"] != values["prompt_tokens"] + values["completion_tokens"]:
        problems.append("total_tokens 不等于 prompt_tokens + completion_tokens")
    return _check(
        "token_usage",
        "计费",
        not problems,
        "usage 字段完整且数值自洽。" if not problems else "；".join(problems),
        round_number=round_number,
        response=response,
        details={"usage": usage},
    )


def check_function_calling(response: ProbeResponse, round_number: int) -> CheckResult:
    if response.error:
        return _error_result("function_calling", "能力", response, round_number)
    message = _message(response.data)
    calls = message.get("tool_calls") if message else None
    if not isinstance(calls, list) or not calls:
        return _check(
            "function_calling",
            "能力",
            False,
            "未返回 tool_calls，Function Calling 可能不受支持。",
            round_number=round_number,
            response=response,
        )
    call = calls[0] if isinstance(calls[0], dict) else {}
    function = call.get("function") if isinstance(call, dict) else None
    problems: list[str] = []
    if call.get("type") != "function":
        problems.append("tool_call.type 不是 function")
    if not isinstance(call.get("id"), str) or not call["id"]:
        problems.append("缺少 tool_call.id")
    if not isinstance(function, dict) or function.get("name") != "lookup_weather":
        problems.append("函数名称不匹配")
    arguments: Any = function.get("arguments") if isinstance(function, dict) else None
    try:
        parsed_arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
    except (TypeError, ValueError):
        parsed_arguments = None
    if not isinstance(parsed_arguments, dict) or not isinstance(parsed_arguments.get("city"), str):
        problems.append("arguments 不是包含 city 字符串的 JSON 对象")
    return _check(
        "function_calling",
        "能力",
        not problems,
        "Function Calling 结构和参数有效。" if not problems else "；".join(problems),
        round_number=round_number,
        response=response,
        details={"tool_call_count": len(calls), "arguments": parsed_arguments},
    )


def check_structured_output(response: ProbeResponse, round_number: int) -> CheckResult:
    if response.error:
        return _error_result("structured_output", "能力", response, round_number)
    text = _message_text(_message(response.data)).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except (TypeError, ValueError) as exc:
        return _check(
            "structured_output",
            "能力",
            False,
            f"返回内容不是合法 JSON：{_truncate(exc)}",
            round_number=round_number,
            response=response,
        )
    problems: list[str] = []
    if not isinstance(value, dict):
        problems.append("结果不是 JSON 对象")
    else:
        if not isinstance(value.get("answer"), str) or not value["answer"].strip():
            problems.append("answer 不是非空字符串")
        confidence = value.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            problems.append("confidence 不是 0 到 1 之间的数字")
        extras = set(value) - {"answer", "confidence"}
        if extras:
            problems.append(f"出现 schema 未声明字段：{', '.join(sorted(extras))}")
    return _check(
        "structured_output",
        "能力",
        not problems,
        "strict JSON Schema 输出有效。" if not problems else "；".join(problems),
        round_number=round_number,
        response=response,
        details={"parsed": value},
    )


def _stream_text(events: Iterable[dict[str, Any]]) -> str:
    pieces: list[str] = []
    for event in events:
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            continue
        delta = choices[0].get("delta")
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        if isinstance(content, str):
            pieces.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    pieces.append(item["text"])
    return "".join(pieces)


def check_streaming(response: ProbeResponse, round_number: int) -> CheckResult:
    if response.error:
        return _error_result("streaming", "协议", response, round_number)
    events = response.stream_events
    problems: list[str] = []
    if not events:
        problems.append("没有有效事件")
    if not response.stream_done:
        problems.append("缺少 [DONE]")
    if events and not isinstance(events[0].get("id"), str):
        problems.append("首个事件缺少 id")
    if events and not isinstance(events[0].get("choices"), list):
        problems.append("首个事件缺少 choices")
    content = _stream_text(events)
    if not content.strip():
        problems.append("没有收到文本增量")
    return _check(
        "streaming",
        "协议",
        not problems,
        "SSE 流式事件序列有效。" if not problems else "；".join(problems),
        round_number=round_number,
        response=response,
        details={"event_count": len(events), "content_length": len(content)},
    )


def check_long_context(response: ProbeResponse, needle: str, round_number: int) -> CheckResult:
    if response.error:
        return _error_result("long_context", "能力", response, round_number)
    text = _message_text(_message(response.data))
    passed = needle.lower() in text.lower()
    return _check(
        "long_context",
        "能力",
        passed,
        "成功找回长文本中的校验词。" if passed else "未找回长文本中的校验词。",
        round_number=round_number,
        response=response,
        details={"needle": needle, "response_length": len(text)},
    )


def basic_payload(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "max_tokens": 8,
    }


def function_calling_payload(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "请调用 lookup_weather 查询北京天气，不要直接回答。"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "description": "查询城市天气",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "lookup_weather"}},
        "max_tokens": 64,
    }


def structured_output_payload(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "返回 answer 和 confidence 两个字段，answer 写 OK，confidence 写 1。"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer_schema",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["answer", "confidence"],
                    "additionalProperties": False,
                },
            },
        },
        "max_tokens": 32,
    }


def streaming_payload(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the word STREAM."}],
        "max_tokens": 16,
        "stream": True,
    }


def long_context_payload(model: str, context_chars: int) -> tuple[dict[str, Any], str]:
    needle = "MINIATURE_FIESTA_CONTEXT_CHECK_7F3A"
    filler = "This is neutral context used for a retrieval probe. It carries no instructions. "
    marker = "\nSPECIAL MARKER: " + needle + "\n"
    filler_budget = max(0, context_chars - len(marker))
    left_budget = filler_budget // 2
    right_budget = filler_budget - left_budget
    left = (filler * (left_budget // len(filler) + 1))[:left_budget]
    right = (filler * (right_budget // len(filler) + 1))[:right_budget]
    document = left + marker + right
    prompt = (
        "Read the document below. Return only the exact SPECIAL MARKER value.\n\n"
        f"DOCUMENT:\n{document}\n\nSPECIAL MARKER:"
    )
    return (
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
        },
        needle,
    )


def _selected_checks(mode: str) -> tuple[str, ...]:
    try:
        return MODE_CHECKS[mode]
    except KeyError as exc:
        raise ValueError(f"不支持的检测模式：{mode}") from exc


def run_model_round(
    client: OpenAIProbeClient,
    model: str,
    *,
    mode: str,
    round_number: int,
    context_chars: int = 12000,
    fail_fast: bool = False,
) -> list[CheckResult]:
    """执行一个模型的一轮检测。基础请求失败时跳过后续需要响应的探针。"""
    selected = _selected_checks(mode)
    checks: list[CheckResult] = []

    def skip_remaining(message: str) -> None:
        existing = {item.name for item in checks}
        for name in selected:
            if name not in existing:
                checks.append(_skip(name, "未执行", message, round_number))

    basic_response = client.post(basic_payload(model))
    basic_result = check_basic(basic_response, model, round_number)
    checks.append(basic_result)

    if "model_consistency" in selected:
        model_result = check_model_consistency(basic_response, model, round_number)
        checks.append(model_result)
        if fail_fast and not model_result.passed:
            skip_remaining("模型一致性失败，已按 fail-fast 停止后续探针。")
            return checks

    if not basic_result.passed:
        if basic_result.status == "unavailable":
            for name in selected:
                if name not in {"basic_request", "model_consistency"}:
                    checks.append(
                        _unavailable(name, "可用性", "当前不可用：基础请求返回服务端无可用通道。", round_number, basic_response)
                    )
            return checks
        for name in selected:
            if name not in {"basic_request", "model_consistency"}:
                checks.append(_skip(name, "未执行", "基础请求失败，未继续发送后续探针。", round_number))
        return checks

    if "protocol" in selected:
        protocol_result = check_protocol(basic_response, round_number)
        checks.append(protocol_result)
        if fail_fast and not protocol_result.passed:
            skip_remaining("协议检查失败，已按 fail-fast 停止后续探针。")
            return checks
    if "token_usage" in selected:
        usage_result = check_token_usage(basic_response, round_number)
        checks.append(usage_result)
        if fail_fast and not usage_result.passed:
            skip_remaining("Token 用量检查失败，已按 fail-fast 停止后续探针。")
            return checks

    if "function_calling" in selected:
        response = client.post(function_calling_payload(model))
        function_result = check_function_calling(response, round_number)
        checks.append(function_result)
        if fail_fast and not function_result.passed:
            skip_remaining("Function Calling 检查失败，已按 fail-fast 停止后续探针。")
            return checks
    if "structured_output" in selected:
        response = client.post(structured_output_payload(model))
        structured_result = check_structured_output(response, round_number)
        checks.append(structured_result)
        if fail_fast and not structured_result.passed:
            skip_remaining("结构化输出检查失败，已按 fail-fast 停止后续探针。")
            return checks
    if "streaming" in selected:
        response = client.post(streaming_payload(model), stream=True)
        streaming_result = check_streaming(response, round_number)
        checks.append(streaming_result)
        if fail_fast and not streaming_result.passed:
            skip_remaining("流式协议检查失败，已按 fail-fast 停止后续探针。")
            return checks
    if "long_context" in selected:
        payload, needle = long_context_payload(model, context_chars)
        response = client.post(payload)
        checks.append(check_long_context(response, needle, round_number))
    return checks


def summarize_model(model: str, checks: list[CheckResult]) -> ModelSummary:
    applicable = [item for item in checks if item.status not in {"skip", "unavailable"}]
    total_weight = sum(CHECK_WEIGHTS.get(item.name, 0.0) for item in applicable)
    earned_weight = sum(CHECK_WEIGHTS.get(item.name, 0.0) for item in applicable if item.passed)
    score = round(earned_weight / total_weight * 100, 2) if total_weight else 0.0
    passed = sum(item.passed for item in checks)
    failed = sum(item.status == "fail" for item in checks)
    skipped = sum(item.status == "skip" for item in checks)
    unavailable = sum(item.status == "unavailable" for item in checks)
    if unavailable and not applicable:
        verdict = "不可用"
        score = None
    elif score >= 85:
        verdict = "优秀"
    elif score >= 70:
        verdict = "通过"
    elif score >= 50:
        verdict = "基本合格"
    else:
        verdict = "未达标"
    # 基础请求、模型一致性和协议是核心项；核心失败时不应仅靠其他能力拉高结论。
    if any(item.name in {"basic_request", "model_consistency", "protocol"} and not item.passed for item in applicable):
        verdict = "未达标"
    return ModelSummary(
        model=model,
        score=score,
        verdict=verdict,
        passed=passed,
        failed=failed,
        skipped=skipped,
        unavailable=unavailable,
        total=len(checks),
        checks=[item.to_dict() for item in checks],
    )


def build_report(
    api_url: str,
    models: list[str],
    mode: str,
    rounds: int,
    all_checks: dict[str, list[CheckResult]],
    *,
    timeout: float,
    retries: int,
    context_chars: int,
) -> dict[str, Any]:
    summaries = [summarize_model(model, all_checks.get(model, [])) for model in models]
    total_checks = sum(summary.total for summary in summaries)
    total_passed = sum(summary.passed for summary in summaries)
    total_failed = sum(summary.failed for summary in summaries)
    total_skipped = sum(summary.skipped for summary in summaries)
    total_unavailable = sum(summary.unavailable for summary in summaries)
    scored_summaries = [summary for summary in summaries if summary.score is not None]
    return {
        "schema_version": "1.0",
        "detector_version": "0.2.0",
        "generated_at": _now_iso(),
        "api_url": api_url,
        "protocol": "openai-chat-completions",
        "mode": mode,
        "rounds": rounds,
        "timeout_seconds": timeout,
        "retries": retries,
        "context_chars": context_chars if mode == "full" else None,
        "evidence_notice": "结果基于协议与行为证据，不构成模型官方身份的加密证明。",
        "summary": {
            "models": len(models),
            "checks": total_checks,
            "passed": total_passed,
            "failed": total_failed,
            "skipped": total_skipped,
            "unavailable": total_unavailable,
            "available_models": sum(summary.unavailable == 0 for summary in summaries),
            "unavailable_models": sum(summary.unavailable > 0 for summary in summaries),
            "all_passed": total_failed == 0 and total_unavailable == 0 and total_checks > 0,
            "average_score": round(sum(item.score for item in scored_summaries) / len(scored_summaries), 2) if scored_summaries else None,
        },
        "models": [summary.to_dict() for summary in summaries],
    }
