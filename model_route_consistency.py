#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检测 OpenAI 兼容 API 中转服务的模型路由一致性。

示例：
    python model_route_consistency.py \
        --api-url https://example.com/v1/chat/completions \
        --api-key sk-xxx \
        --model gpt-4o

API Key 也可以放在 API_KEY 或 OPENAI_API_KEY 环境变量中，避免直接写在命令行历史里。
"""

from __future__ import annotations

import argparse
import getpass
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

import requests


TIMEOUT_SECONDS = 30
DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"


@dataclass
class TestResult:
    """一次模型路由检查的结果。"""

    expected: str
    actual: Optional[str]
    passed: bool
    error: Optional[str] = None


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
                # 某些被替换的输出流不支持 reconfigure，保留其原设置即可。
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="检查 API 返回的 model 字段是否与请求模型完全一致。"
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("API_URL", DEFAULT_API_URL),
        help="完整的 OpenAI 兼容 chat/completions 地址（默认读取 API_URL 环境变量）。",
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
        "--rounds",
        type=int,
        default=1,
        help="每个模型测试次数（默认 1）。",
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
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="跳过 SSL 证书验证（等价于 requests 的 verify=False）。",
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


def resolve_api_key(explicit_key: Optional[str]) -> str:
    key = (explicit_key or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if key:
        return key

    # 命令行交互时隐藏输入；非交互环境下给出明确提示。
    if sys.stdin.isatty():
        key = getpass.getpass("请输入 API Key（输入不会回显）：").strip()
    else:
        key = ""
    if not key:
        raise ValueError("未提供 API Key，请使用 --api-key 或设置 API_KEY 环境变量。")
    return key


def resolve_models(model_args: Optional[Sequence[str]]) -> List[str]:
    values: List[str] = []
    for item in model_args or []:
        values.extend(part.strip() for part in item.split(",") if part.strip())

    if not values:
        values = [os.getenv("MODEL", "gpt-4o").strip() or "gpt-4o"]

    # 保持输入顺序，同时去掉重复项。
    return list(dict.fromkeys(values))


def resolve_proxies(http_proxy: Optional[str], https_proxy: Optional[str]) -> Dict[str, str]:
    http_value = http_proxy or os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    https_value = https_proxy or os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    proxies: Dict[str, str] = {}
    if http_value:
        proxies["http"] = http_value
    if https_value:
        proxies["https"] = https_value
    return proxies


def redact_url_credentials(value: str) -> str:
    """隐藏代理 URL 中可能存在的用户名和密码，避免写入终端或日志。"""
    try:
        parsed = urlsplit(value)
        if not parsed.username and not parsed.password:
            return value
        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"
        return urlunsplit(
            (parsed.scheme, "***:***@" + host, parsed.path, parsed.query, parsed.fragment)
        )
    except ValueError:
        # 代理格式不标准时不影响请求本身；日志中只保留占位符。
        return "<代理地址>"


def configure_logger(log_file: Optional[str]) -> Optional[logging.Logger]:
    if not log_file:
        return None

    path = Path(log_file).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("model_route_consistency")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 避免同一进程重复调用 main 时添加多个 handler。
    if not logger.handlers:
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    return logger


def emit(
    message: str,
    *,
    color: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    no_color: bool = False,
) -> None:
    use_color = bool(color) and not no_color and "NO_COLOR" not in os.environ
    if use_color:
        print(f"{color}{message}{AnsiColor.RESET}")
    else:
        print(message)
    if logger:
        logger.info(message)


def check_model_route(
    session: requests.Session,
    api_url: str,
    api_key: str,
    model: str,
    *,
    proxies: Optional[Dict[str, str]],
    verify: bool,
    timeout: int,
) -> TestResult:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 1,
    }

    try:
        response = session.post(
            api_url,
            headers=headers,
            json=payload,
            proxies=proxies or None,
            verify=verify,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return TestResult(expected=model, actual=None, passed=False, error=f"请求失败：{exc}")
    except ValueError as exc:
        return TestResult(expected=model, actual=None, passed=False, error=f"返回内容不是有效 JSON：{exc}")

    if not isinstance(data, dict):
        return TestResult(expected=model, actual=None, passed=False, error="返回 JSON 不是对象。")

    actual_value = data.get("model")
    actual = str(actual_value) if actual_value is not None else None
    return TestResult(expected=model, actual=actual, passed=actual == model)


def print_result(
    result: TestResult,
    *,
    model: str,
    round_number: int,
    logger: Optional[logging.Logger],
    no_color: bool,
) -> None:
    prefix = f"[模型 {model} | 第 {round_number} 轮] "
    if result.error:
        emit(prefix + f"❌ {result.error}", color=AnsiColor.RED, logger=logger, no_color=no_color)
    elif result.passed:
        # 保留题目要求的固定成功文案。
        emit(prefix + "✅ 100分 官方一致", color=AnsiColor.GREEN, logger=logger, no_color=no_color)
    else:
        actual = result.actual if result.actual is not None else "<缺失>"
        # 保留题目要求的固定异常文案。
        emit(
            prefix + f"❌ 路由异常：期望 {result.expected}，实际返回 {actual}",
            color=AnsiColor.RED,
            logger=logger,
            no_color=no_color,
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_console_encoding()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.rounds < 1:
        parser.error("--rounds 必须是大于等于 1 的整数。")

    try:
        api_key = resolve_api_key(args.api_key)
    except ValueError as exc:
        parser.error(str(exc))

    models = resolve_models(args.models)
    proxies = resolve_proxies(args.http_proxy, args.https_proxy)
    verify = not args.no_verify
    logger = configure_logger(args.log_file)

    emit(f"API 地址：{args.api_url}", logger=logger, no_color=args.no_color)
    emit(f"测试模型：{', '.join(models)}；每个模型 {args.rounds} 轮", logger=logger, no_color=args.no_color)
    if proxies:
        safe_proxies = {kind: redact_url_credentials(value) for kind, value in proxies.items()}
        emit(f"代理：{safe_proxies}", logger=logger, no_color=args.no_color)
    if not verify:
        emit(
            "⚠️ 已关闭 SSL 证书验证（verify=False）",
            color=AnsiColor.YELLOW,
            logger=logger,
            no_color=args.no_color,
        )

    session = requests.Session()
    results: List[Tuple[str, TestResult]] = []
    try:
        for round_number in range(1, args.rounds + 1):
            for model in models:
                result = check_model_route(
                    session,
                    args.api_url,
                    api_key,
                    model,
                    proxies=proxies,
                    verify=verify,
                    timeout=TIMEOUT_SECONDS,
                )
                results.append((model, result))
                print_result(
                    result,
                    model=model,
                    round_number=round_number,
                    logger=logger,
                    no_color=args.no_color,
                )
    except KeyboardInterrupt:
        emit("❌ 用户中断测试。", color=AnsiColor.RED, logger=logger, no_color=args.no_color)
        return 130
    finally:
        session.close()

    total = len(results)
    passed = sum(1 for _, result in results if result.passed)
    rate = passed / total * 100 if total else 0.0
    summary = f"汇总：{passed}/{total} 通过，通过率 {rate:.2f}%"
    emit(
        summary,
        color=AnsiColor.GREEN if passed == total else AnsiColor.RED,
        logger=logger,
        no_color=args.no_color,
    )

    if logger:
        for handler in logger.handlers:
            handler.flush()
            handler.close()
        logger.handlers.clear()

    return 0 if total > 0 and passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
