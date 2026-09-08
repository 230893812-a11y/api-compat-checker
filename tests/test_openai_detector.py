import json
import unittest

import requests

from openai_detector import OpenAIProbeClient, build_report, classify_http_error, run_model_round, summarize_model
from model_route_consistency import resolve_api_key, validate_api_url


class FakeResponse:
    def __init__(self, data=None, *, lines=None, status_code=200):
        self._data = data
        self._lines = lines or []
        self.status_code = status_code
        self.headers = {}
        self.text = json.dumps(data, ensure_ascii=False) if data is not None else ""
        self.closed = False

    def json(self):
        return self._data

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)

    def close(self):
        self.closed = True


def completion(model, content="OK", *, usage=True):
    result = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
    if usage:
        result["usage"] = {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
    return result


class FakeSession:
    def __init__(self, *, actual_model=None, structured_content=None, invalid_usage=False):
        self.actual_model = actual_model
        self.structured_content = structured_content
        self.invalid_usage = invalid_usage
        self.calls = []

    def post(self, url, **kwargs):
        payload = kwargs["json"]
        self.calls.append(payload)
        model = self.actual_model or payload["model"]
        if payload.get("stream"):
            lines = [
                "data: "
                + json.dumps(
                    {
                        "id": "chatcmpl-stream",
                        "choices": [{"index": 0, "delta": {"content": "STREAM"}}],
                    }
                ),
                "data: "
                + json.dumps(
                    {
                        "id": "chatcmpl-stream",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                ),
                "data: [DONE]",
            ]
            return FakeResponse(lines=lines)
        if payload.get("tools"):
            data = completion(model, usage=not self.invalid_usage)
            data["choices"][0]["message"] = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_test",
                        "type": "function",
                        "function": {"name": "lookup_weather", "arguments": '{"city":"Beijing"}'},
                    }
                ],
            }
            data["choices"][0]["finish_reason"] = "tool_calls"
            return FakeResponse(data)
        if payload.get("response_format"):
            return FakeResponse(completion(model, self.structured_content or '{"answer":"OK","confidence":1}'))
        if "SPECIAL MARKER" in payload["messages"][0]["content"]:
            return FakeResponse(completion(model, "MINIATURE_FIESTA_CONTEXT_CHECK_7F3A"))
        data = completion(model, usage=not self.invalid_usage)
        if self.invalid_usage:
            data["usage"] = {"prompt_tokens": -1, "completion_tokens": "1", "total_tokens": 0}
        return FakeResponse(data)


class RetrySession:
    def __init__(self):
        self.calls = 0

    def post(self, url, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise requests.ConnectionError("temporary")
        return FakeResponse(completion(kwargs["json"]["model"]))


class UnavailableSession:
    def post(self, url, **kwargs):
        return FakeResponse(
            {"error": {"message": "No available channel for model gpt-test under group default"}},
            status_code=503,
        )


class ModelsSession:
    def __init__(self, response):
        self.response = response
        self.last_url = None

    def get(self, url, **kwargs):
        self.last_url = url
        return self.response


class OpenAIDetectorTests(unittest.TestCase):
    def test_api_key_normalizes_bearer_and_quotes(self):
        self.assertEqual(resolve_api_key('"Bearer sk-test"'), "sk-test")

    def test_api_key_rejects_non_ascii_text(self):
        with self.assertRaises(ValueError):
            resolve_api_key("这不是 API Key")

    def test_api_url_can_be_root_v1_or_full_endpoint(self):
        self.assertEqual(validate_api_url("https://example.test"), "https://example.test/v1/chat/completions")
        self.assertEqual(validate_api_url("https://example.test/v1"), "https://example.test/v1/chat/completions")
        self.assertEqual(validate_api_url("https://example.test/v1/chat/completions"), "https://example.test/v1/chat/completions")

    def test_full_mode_passes_against_valid_openai_shape(self):
        session = FakeSession()
        client = OpenAIProbeClient(session, "https://example.test/v1/chat/completions", "secret", retries=0)
        checks = run_model_round(client, "gpt-test", mode="full", round_number=1, context_chars=500)
        self.assertEqual(len(checks), 8)
        self.assertTrue(all(item.passed for item in checks), checks)
        self.assertEqual(summarize_model("gpt-test", checks).score, 100.0)

    def test_model_mismatch_is_reported(self):
        session = FakeSession(actual_model="gpt-mini")
        client = OpenAIProbeClient(session, "https://example.test/v1/chat/completions", "secret", retries=0)
        checks = run_model_round(client, "gpt-test", mode="quick", round_number=1)
        self.assertTrue(checks[0].passed)
        self.assertFalse(checks[1].passed)
        self.assertIn("不一致", checks[1].message)

    def test_invalid_structured_output_fails_only_that_capability(self):
        session = FakeSession(structured_content='{"answer":"OK"}')
        client = OpenAIProbeClient(session, "https://example.test/v1/chat/completions", "secret", retries=0)
        checks = run_model_round(client, "gpt-test", mode="standard", round_number=1)
        structured = next(item for item in checks if item.name == "structured_output")
        self.assertFalse(structured.passed)
        self.assertIn("confidence", structured.message)

    def test_invalid_usage_is_detected(self):
        session = FakeSession(invalid_usage=True)
        client = OpenAIProbeClient(session, "https://example.test/v1/chat/completions", "secret", retries=0)
        checks = run_model_round(client, "gpt-test", mode="standard", round_number=1)
        usage = next(item for item in checks if item.name == "token_usage")
        self.assertFalse(usage.passed)

    def test_fail_fast_skips_following_probes(self):
        session = FakeSession(actual_model="gpt-mini")
        client = OpenAIProbeClient(session, "https://example.test/v1/chat/completions", "secret", retries=0)
        checks = run_model_round(client, "gpt-test", mode="standard", round_number=1, fail_fast=True)
        self.assertEqual([payload.get("tools") is not None for payload in session.calls], [False])
        self.assertTrue(any(item.status == "skip" for item in checks))

    def test_transient_connection_error_is_retried(self):
        session = RetrySession()
        client = OpenAIProbeClient(session, "https://example.test/v1/chat/completions", "secret", retries=1, retry_backoff=0)
        response = client.post({"model": "gpt-test"})
        self.assertIsNone(response.error)
        self.assertEqual(session.calls, 2)

    def test_models_preflight_derives_models_url(self):
        session = ModelsSession(FakeResponse({"object": "list", "data": [{"id": "gpt-test"}]}))
        client = OpenAIProbeClient(session, "https://example.test/v1/chat/completions", "secret")
        response = client.list_models()
        self.assertIsNone(response.error)
        self.assertEqual(session.last_url, "https://example.test/v1/models")

    def test_models_preflight_classifies_invalid_token(self):
        session = ModelsSession(FakeResponse({"error": {"message": "Invalid token"}}, status_code=401))
        client = OpenAIProbeClient(session, "https://example.test/v1/chat/completions", "secret")
        response = client.list_models()
        self.assertIn("认证失败", response.error)
        self.assertIn("HTTP 401", response.error)

    def test_http_error_classification(self):
        self.assertIn("权限不足", classify_http_error(403))
        self.assertIn("接口或模型不存在", classify_http_error(404))
        self.assertIn("额度不足", classify_http_error(429))

    def test_no_available_channel_is_unavailable_and_not_scored(self):
        client = OpenAIProbeClient(UnavailableSession(), "https://example.test/v1/chat/completions", "secret", retries=0)
        checks = run_model_round(client, "gpt-test", mode="standard", round_number=1)
        self.assertTrue(all(item.status == "unavailable" for item in checks))
        summary = summarize_model("gpt-test", checks)
        self.assertEqual(summary.verdict, "不可用")
        self.assertIsNone(summary.score)
        report = build_report("https://example.test/v1/chat/completions", ["gpt-test"], "standard", 1, {"gpt-test": checks}, timeout=30, retries=0, context_chars=12000)
        self.assertEqual(report["summary"]["unavailable_models"], 1)
        self.assertIsNone(report["summary"]["average_score"])


if __name__ == "__main__":
    unittest.main()
