#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI 兼容 API 中转站检测 CLI。

保留了旧版的 ``--api-url``、``--model``、``--rounds`` 等参数，并增加
quick/standard/full 三档检测模式。API Key 优先从环境变量读取，绝不写入
日志或 JSON 报告。
"""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

from openai_detector import OpenAIProbeClient, MODE_CHECKS, basic_payload, build_report, run_model_round


TIMEOUT_SECONDS = 30.0
DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_RETRIES = 1
DEFAULT_RETRY_BACKOFF = 0.5


@dataclass
class TestResult:
    """兼容旧版调用者的单次模型路由结果。"""

    expected: str
    actual: Optional[str]
    passed: bool
    error: Optional[str] = None
    status_code: Optional[int] = None
    elapsed_ms: Optional[float] = None


class AnsiColor:
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"


def configure_console_encoding() -> None:
    """让 Windows 默认控制台也能输出中文及 ✅/❌ 等符号。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="检测 OpenAI Chat Completions 中转站的真实性、能力和协议兼容性。"
    )
    parser.add_argument(
        "--config",
        default=os.getenv("CONFIG_FILE"),
        help="JSON 配置文件路径；命令行参数优先于配置文件。",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="完整的 OpenAI 兼容 chat/completions 地址。",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API Key；也可使用 API_KEY 或 OPENAI_API_KEY 环境变量。",
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help="要测试的模型名称；可重复使用，也可用逗号分隔（默认 gpt-4o）。",
    )
    parser.add_argument(
        "--mode",
        choices=tuple(MODE_CHECKS),
        default=None,
        help="检测深度：quick（2 项）、standard（7 项）、full（含长上下文，8 项）。",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="每个模型完整检测次数（默认 1）。",
    )
    parser.add_argument(
        "--http-proxy",
        "--http_proxy",
        default=None,
        help="HTTP 代理；未指定时读取 HTTP_PROXY/http_proxy。",
    )
    parser.add_argument(
        "--https-proxy",
        "--https_proxy",
        default=None,
        help="HTTPS 代理；未指定时读取 HTTPS_PROXY/https_proxy。",
    )
    verify_group = parser.add_mutually_exclusive_group()
    verify_group.add_argument(
        "--no-verify",
        action="store_true",
        help="跳过 SSL 证书验证（不推荐）。",
    )
    verify_group.add_argument(
        "--verify",
        action="store_true",
        help="强制开启 SSL 证书验证。",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="单次请求超时时间（秒，默认 30）。",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=None,
        help="瞬时网络错误重试次数（默认 1；设为 0 可关闭）。",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=None,
        help="重试退避系数（秒，默认 0.5）。",
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=None,
        help="full 模式长上下文探针的字符数（默认 12000）。",
    )
    parser.add_argument(
        "--output-format",
        choices=("text", "json"),
        default=None,
        help="终端输出格式（默认 text）。",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="将完整 JSON 报告写入此文件；不影响终端输出格式。",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="首次检测失败后停止后续探针。",
    )
    parser.add_argument(
        "--no-env-proxy",
        action="store_true",
        help="忽略 requests 自动读取的系统代理环境变量。",
    )
    parser.add_argument(
        "--skip-models-check",
        action="store_true",
        help="跳过开始前的 GET /v1/models 认证和模型预检查。",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="可选日志文件路径；日志使用 UTF-8 编码，不记录 API Key。",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="关闭终端颜色。",
    )
    return parser


def load_config(config_file: Optional[str]) -> Dict[str, Any]:
    if not config_file:
        return {}
    path = Path(config_file).expanduser()
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"配置文件不存在：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"配置文件不是有效 JSON（第 {exc.lineno} 行）：{exc.msg}") from exc
    except OSError as exc:
        raise ValueError(f"无法读取配置文件：{exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("配置文件顶层必须是 JSON 对象。")
    return config


def _config_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    return config[key] if key in config else default


def resolve_api_key(explicit_key: Optional[str]) -> str:
    key = (explicit_key or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    # 用户常把文档中的 `Bearer xxx` 或引号一起复制进来；去掉包装后再发送。
    if len(key) >= 2 and key[0] == key[-1] and key[0] in {"\"", "'"}:
        key = key[1:-1].strip()
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    if key:
        try:
            key.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(
                "API Key 含有非 ASCII 字符。请只粘贴令牌本身，不要粘贴中文提示、全角引号或其他文本。"
            ) from exc
        return key
    if sys.stdin.isatty():
        key = getpass.getpass("请输入 API Key（输入不会回显）：").strip()
    if not key:
        raise ValueError("未提供 API Key，请使用 --api-key 或设置 API_KEY 环境变量。")
    return key


def resolve_models(model_args: Optional[Sequence[str]], configured: Any = None) -> List[str]:
    raw: List[Any] = list(model_args or [])
    if not raw and configured is not None:
        if isinstance(configured, str):
            raw = [configured]
        elif isinstance(configured, (list, tuple)):
            raw = list(configured)
        else:
            raise ValueError("配置项 models 必须是字符串或字符串数组。")
    values: List[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise ValueError("模型名称必须是字符串。")
        values.extend(part.strip() for part in item.split(",") if part.strip())
    if not values:
        values = [os.getenv("MODEL", "gpt-4o").strip() or "gpt-4o"]
    return list(dict.fromkeys(values))


def resolve_proxies(
    http_proxy: Optional[str], https_proxy: Optional[str], *, use_environment: bool = True
) -> Dict[str, str]:
    http_value = http_proxy
    https_value = https_proxy
    if use_environment:
        if http_value is None:
            http_value = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        if https_value is None:
            https_value = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    proxies: Dict[str, str] = {}
    if http_value and str(http_value).strip():
        proxies["http"] = str(http_value).strip()
    if https_value and str(https_value).strip():
        proxies["https"] = str(https_value).strip()
    return proxies


def validate_api_url(value: str) -> str:
    value = value.strip()
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("API 地址必须是包含主机名的 http:// 或 https:// URL。")
    path = parsed.path.rstrip("/")
    if path == "":
        path = "/v1/chat/completions"
    elif path == "/v1":
        path += "/chat/completions"
    elif path.endswith("/models"):
        raise ValueError("API 地址不能填写 /models，请填写服务根地址、/v1 或 /v1/chat/completions。")
    elif not path.endswith("/chat/completions"):
        path += "/chat/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))


def _parse_positive_float(value: Any, name: str, default: float) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数字。") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} 必须是大于 0 的有限数字。")
    return parsed


def _parse_non_negative_float(value: Any, name: str, default: float) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数字。") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{name} 必须是非负有限数字。")
    return parsed


def _parse_non_negative_int(value: Any, name: str, default: int) -> int:
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是整数。") from exc
    if parsed < 0:
        raise ValueError(f"{name} 不能小于 0。")
    return parsed


def _parse_rounds(value: Any, default: int = 1) -> int:
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("--rounds 必须是整数。") from exc
    if parsed < 1:
        raise ValueError("--rounds 必须是大于等于 1 的整数。")
    return parsed


def _parse_bool(value: Any, name: str, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} 必须是布尔值。")


def redact_url(value: str) -> str:
    """隐藏 URL 用户密码和常见 query-string 密钥。"""
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if parsed.port:
            host += f":{parsed.port}"
        netloc = f"***:***@{host}" if parsed.username or parsed.password else host
        secret_keys = {"api_key", "apikey", "access_token", "token", "key"}
        query = [
            (key, "***" if key.lower() in secret_keys else item)
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        ]
        return urlunsplit((parsed.scheme, netloc, parsed.path, urlencode(query), parsed.fragment))
    except ValueError:
        return "<URL>"


def redact_url_credentials(value: str) -> str:
    """旧版兼容函数：只隐藏代理 URL 的用户名和密码。"""
    try:
        parsed = urlsplit(value)
        if not parsed.username and not parsed.password:
            return value
        host = parsed.hostname or ""
        if parsed.port:
            host += f":{parsed.port}"
        return urlunsplit((parsed.scheme, "***:***@" + host, parsed.path, parsed.query, parsed.fragment))
    except ValueError:
        return "<代理地址>"


def configure_logger(log_file: Optional[str]) -> Optional[logging.Logger]:
    if not log_file:
        return None
    path = Path(log_file).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("model_route_consistency")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    return logger


def close_logger(logger: Optional[logging.Logger]) -> None:
    if not logger:
        return
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def emit(
    message: str,
    *,
    color: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    no_color: bool = False,
) -> None:
    if color and not no_color and "NO_COLOR" not in os.environ:
        print(f"{color}{message}{AnsiColor.RESET}")
    else:
        print(message)
    if logger:
        logger.info(message)


def print_result(
    result: TestResult,
    *,
    model: str,
    round_number: int,
    logger: Optional[logging.Logger],
    no_color: bool,
) -> None:
    """旧版兼容输出函数。新模式使用 print_check_result。"""
    prefix = f"[模型 {model} | 第 {round_number} 轮] "
    if result.error:
        emit(prefix + f"❌ {result.error}", color=AnsiColor.RED, logger=logger, no_color=no_color)
    elif result.passed:
        emit(prefix + "✅ 100分 官方一致", color=AnsiColor.GREEN, logger=logger, no_color=no_color)
    else:
        actual = result.actual if result.actual is not None else "<缺失>"
        emit(
            prefix + f"❌ 路由异常：期望 {result.expected}，实际返回 {actual}",
            color=AnsiColor.RED,
            logger=logger,
            no_color=no_color,
        )


def check_model_route(
    session: requests.Session,
    api_url: str,
    api_key: str,
    model: str,
    *,
    proxies: Optional[Dict[str, str]],
    verify: bool,
    timeout: float,
) -> TestResult:
    """旧版函数兼容层：只执行一次基础请求和 model 字段检查。"""
    client = OpenAIProbeClient(
        session,
        api_url,
        api_key,
        proxies=proxies,
        verify=verify,
        timeout=timeout,
        retries=0,
    )
    response = client.post(basic_payload(model))
    if response.error:
        return TestResult(model, None, False, response.error, response.status_code, response.elapsed_ms)
    actual_value = response.data.get("model") if response.data else None
    actual = str(actual_value) if actual_value is not None else None
    return TestResult(model, actual, actual == model, None, response.status_code, response.elapsed_ms)


def print_check_result(check: Any, model: str, *, logger: Optional[logging.Logger], no_color: bool) -> None:
    if check.status == "skip":
        icon, color = "⏭", AnsiColor.YELLOW
    elif check.status == "unavailable":
        icon, color = "⚠️", AnsiColor.YELLOW
    elif check.passed:
        icon, color = "✅", AnsiColor.GREEN
    else:
        icon, color = "❌", AnsiColor.RED
    elapsed = f" [{check.elapsed_ms:.0f} ms]" if check.elapsed_ms is not None else ""
    message = f"[模型 {model} | 第 {check.round_number} 轮 | {check.name}] {icon} {check.message}{elapsed}"
    emit(message, color=color, logger=logger, no_color=no_color)


def _write_report(path_value: str, report: dict[str, Any]) -> None:
    path = Path(path_value).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_console_encoding()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        api_url = validate_api_url(
            str(args.api_url if args.api_url is not None else _config_value(config, "api_url") or os.getenv("API_URL") or DEFAULT_API_URL)
        )
        models = resolve_models(args.models, _config_value(config, "models"))
        mode = str(args.mode or _config_value(config, "mode", "quick")).lower()
        if mode not in MODE_CHECKS:
            raise ValueError("mode 必须是 quick、standard 或 full。")
        rounds = _parse_rounds(args.rounds if args.rounds is not None else _config_value(config, "rounds", os.getenv("ROUNDS")))
        timeout = _parse_positive_float(args.timeout if args.timeout is not None else _config_value(config, "timeout", os.getenv("API_TIMEOUT")), "--timeout", TIMEOUT_SECONDS)
        retries = _parse_non_negative_int(args.retries if args.retries is not None else _config_value(config, "retries", os.getenv("API_RETRIES")), "--retries", DEFAULT_RETRIES)
        retry_backoff = _parse_non_negative_float(args.retry_backoff if args.retry_backoff is not None else _config_value(config, "retry_backoff", os.getenv("API_RETRY_BACKOFF")), "--retry-backoff", DEFAULT_RETRY_BACKOFF)
        context_chars = _parse_non_negative_int(args.context_chars if args.context_chars is not None else _config_value(config, "context_chars", 12000), "--context-chars", 12000)
        if mode == "full" and context_chars < 100:
            raise ValueError("--context-chars 在 full 模式下必须至少为 100。")
        output_format = str(args.output_format or _config_value(config, "output_format", "text")).lower()
        if output_format not in {"text", "json"}:
            raise ValueError("output_format 必须是 text 或 json。")
        configured_verify = _parse_bool(_config_value(config, "verify_ssl"), "verify_ssl", True)
        verify = True if args.verify else False if args.no_verify else configured_verify
        no_color = args.no_color or _parse_bool(_config_value(config, "no_color"), "no_color", False)
        fail_fast = args.fail_fast or _parse_bool(_config_value(config, "fail_fast"), "fail_fast", False)
        no_env_proxy = args.no_env_proxy or _parse_bool(_config_value(config, "no_env_proxy"), "no_env_proxy", False)
        http_proxy = args.http_proxy if args.http_proxy is not None else _config_value(config, "http_proxy")
        https_proxy = args.https_proxy if args.https_proxy is not None else _config_value(config, "https_proxy")
        proxies = resolve_proxies(http_proxy, https_proxy, use_environment=not no_env_proxy)
        log_file = args.log_file if args.log_file is not None else _config_value(config, "log_file") or os.getenv("LOG_FILE")
        if log_file and args.config and not Path(str(log_file)).expanduser().is_absolute():
            log_file = str(Path(args.config).expanduser().parent / str(log_file))
    except ValueError as exc:
        parser.error(str(exc))

    try:
        api_key = resolve_api_key(args.api_key)
    except ValueError as exc:
        parser.error(str(exc))

    logger = configure_logger(log_file)
    session = requests.Session()
    session.trust_env = not no_env_proxy
    client = OpenAIProbeClient(
        session,
        api_url,
        api_key,
        proxies=proxies,
        verify=verify,
        timeout=timeout,
        retries=retries,
        retry_backoff=retry_backoff,
    )
    all_checks: dict[str, list[Any]] = {model: [] for model in models}
    interrupted = False
    stopped_early = False
    preflight_info: dict[str, Any] = {"performed": False, "passed": None}

    try:
        if output_format == "text":
            emit(f"API 地址：{redact_url(api_url)}", logger=logger, no_color=no_color)
            emit(f"检测模式：{mode}；模型：{', '.join(models)}；每个模型 {rounds} 轮", logger=logger, no_color=no_color)
            emit(f"超时：{timeout:g} 秒；重试：{retries} 次", logger=logger, no_color=no_color)
            if proxies:
                safe_proxies = {key: redact_url(value) for key, value in proxies.items()}
                emit(f"代理：{safe_proxies}", logger=logger, no_color=no_color)
            if not verify:
                emit("⚠️ 已关闭 SSL 证书验证（verify=False）", color=AnsiColor.YELLOW, logger=logger, no_color=no_color)

        if not args.skip_models_check:
            preflight_info["performed"] = True
            preflight = client.list_models()
            if preflight.error:
                message = f"预检查失败：{preflight.error}"
                if output_format == "text":
                    emit(message, color=AnsiColor.RED, logger=logger, no_color=no_color)
                report = build_report(redact_url(api_url), models, mode, rounds, all_checks, timeout=timeout, retries=retries, context_chars=context_chars)
                report["preflight"] = {"passed": False, "status_code": preflight.status_code, "error": preflight.error}
                report["interrupted"] = False
                report["stopped_early"] = True
                if args.output:
                    _write_report(args.output, report)
                if output_format == "json":
                    print(json.dumps(report, ensure_ascii=False, indent=2))
                close_logger(logger)
                return 1
            returned_models = preflight.data.get("data", []) if preflight.data else []
            available = {item.get("id") for item in returned_models if isinstance(item, dict) and isinstance(item.get("id"), str)}
            unknown_models = [model for model in models if available and model not in available]
            preflight_info.update({"passed": True, "status_code": preflight.status_code, "available_model_count": len(available), "requested_models_not_listed": unknown_models})
            if output_format == "text":
                if available:
                    emit(f"认证预检查通过：服务返回 {len(available)} 个模型。", logger=logger, no_color=no_color)
                    if unknown_models:
                        emit(f"⚠️ 以下模型不在 /v1/models 列表中，仍将继续探测：{', '.join(unknown_models)}", color=AnsiColor.YELLOW, logger=logger, no_color=no_color)
                else:
                    emit("认证预检查通过，但服务未返回可用模型列表；继续执行探针。", color=AnsiColor.YELLOW, logger=logger, no_color=no_color)

        for round_number in range(1, rounds + 1):
            for model in models:
                checks = run_model_round(
                    client,
                    model,
                    mode=mode,
                    round_number=round_number,
                    context_chars=context_chars,
                    fail_fast=fail_fast,
                )
                all_checks[model].extend(checks)
                if output_format == "text":
                    for check in checks:
                        print_check_result(check, model, logger=logger, no_color=no_color)
                if fail_fast and any(item.status == "fail" for item in checks):
                    stopped_early = True
                    break
            if stopped_early:
                break
    except KeyboardInterrupt:
        interrupted = True
        if output_format == "text":
            emit("❌ 用户中断测试。", color=AnsiColor.RED, logger=logger, no_color=no_color)
    finally:
        session.close()

    report = build_report(
        redact_url(api_url),
        models,
        mode,
        rounds,
        all_checks,
        timeout=timeout,
        retries=retries,
        context_chars=context_chars,
    )
    report["interrupted"] = interrupted
    report["stopped_early"] = stopped_early
    report["preflight"] = preflight_info
    if args.output:
        try:
            _write_report(args.output, report)
        except OSError as exc:
            emit(f"❌ 无法写入报告文件：{exc}", color=AnsiColor.RED, logger=logger, no_color=no_color)
            close_logger(logger)
            return 1

    if output_format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for summary in report["models"]:
            score_display = f"{summary['score']:.2f}/100" if summary["score"] is not None else "不评分"
            emit(
                f"模型 {summary['model']}：{score_display}，{summary['verdict']}，"
                f"通过 {summary['passed']}，失败 {summary['failed']}，不可用 {summary['unavailable']}，跳过 {summary['skipped']}",
                color=AnsiColor.GREEN if summary["failed"] == 0 and summary["unavailable"] == 0 else AnsiColor.YELLOW if summary["unavailable"] else AnsiColor.RED,
                logger=logger,
                no_color=no_color,
            )
        overall = report["summary"]
        average_display = f"{overall['average_score']:.2f}" if overall["average_score"] is not None else "无（没有可评分模型）"
        emit(
            f"汇总：{overall['passed']}/{overall['checks']} 项通过，失败 {overall['failed']}，不可用 {overall['unavailable']}，平均分（仅可用模型） {average_display}",
            color=AnsiColor.GREEN if overall["failed"] == 0 and overall["unavailable"] == 0 and not stopped_early and not interrupted else AnsiColor.YELLOW if overall["failed"] == 0 and overall["unavailable"] else AnsiColor.RED,
            logger=logger,
            no_color=no_color,
        )

    close_logger(logger)
    if interrupted:
        return 130
    return 0 if report["summary"]["all_passed"] and not stopped_early else 1


if __name__ == "__main__":
    raise SystemExit(main())
