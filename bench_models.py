#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""多模型横评 v4：推理 / 代码 / 指令 / 长上下文 / 计费 / 首字延迟 / 吞吐

输出风格与 model_route_consistency.py 一致：一行一项，带 ✅ / ❌ 和耗时。

用法：
    python bench_models.py --selftest    # 先验判分器，不联网
    $env:API_KEY = "sk-..."
    python bench_models.py               # 正式跑

    python bench_models.py --models gpt-6-sol,claude-opus-5-5
    python bench_models.py --dry-run     # 不联网，看输出样式
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass, field, asdict

import requests

DEFAULT_URL = os.environ.get("API_URL") or "https://api.openai.com/v1/chat/completions"

DEFAULT_MODELS = [
    "gpt-6-sol",
    "gpt-6.1-sol",
    "gpt-6-astra",
    "claude-sonnet-5-5",
    "claude-opus-5-5",
    "deepseek-v4.1-flash",
    "grok-4.7",
]

MATH_PROMPT = (
    "某商品进货价 80 元，标价时加价 50%，实际按标价八折出售。"
    "这个商品的利润率是多少？只输出一个整数百分比，例如 20。"
)
MATH_ANSWER = 20

CODE_PROMPT = (
    "用 Python 写一个函数 top_k(nums, k)：返回列表 nums 中第 k 大的数，"
    "重复元素按出现次数计算，例如 [3,1,4,1,5] 中第 3 大是 3。"
    "不要修改输入列表。若列表为空或 k 越界，返回 None 而不是抛异常。"
    "只输出代码，不要解释，不要 markdown 代码块标记。"
)

CODE_CASES = [
    ([3, 1, 4, 1, 5], 3, 3),
    ([1, 1, 1], 2, 1),
    ([5], 1, 5),
    ([2, 2, 3, 1], 2, 2),
    ([], 1, None),
]

INSTR_PROMPT = (
    "请输出恰好两行：第一行是一个四字成语；第二行是这个成语的意思，不超过十个字。"
    "全文不得出现「的」字，不要标点。"
)

STREAM_PROMPT = "用大约一百五十个字介绍你自己，中文回答。"

NEEDLE = "紫罗兰-7391"

# ---------- 颜色 ----------

USE_COLOR = True


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


DIM = lambda s: c(s, "90")
CYAN = lambda s: c(s, "96")
MAGENTA = lambda s: c(s, "95")
GREEN = lambda s: c(s, "92")
RED = lambda s: c(s, "91")
YELLOW = lambda s: c(s, "93")


def dwidth(text: str) -> int:
    plain = re.sub(r"\033\[[0-9;]*m", "", text)
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in plain)


def pad(text: str, width: int) -> str:
    return text + " " * max(0, width - dwidth(text))


def build_long_text() -> str:
    base = (
        "本章介绍数据整理的一般流程，包括采集、清洗、归档与复核四个环节。"
        "在采集阶段需要确认数据来源与授权范围，在清洗阶段需要记录每一次改动的原因，"
        "归档阶段要求统一命名并保留原始副本，复核阶段由另一名同事独立检查。"
    )
    target = 5200
    parts, idx = [], 0
    while sum(len(p) for p in parts) < target:
        idx += 1
        parts.append(f"第{idx}段。" + base)
    body = "".join(parts)
    cut = len(body) // 2
    return body[:cut] + f"\n\n记住这个暗号：{NEEDLE}\n\n" + body[cut:]


LONG_TEXT = build_long_text()


# ---------- 数据结构 ----------

@dataclass
class Probe:
    ok: bool = False
    message: str = ""
    elapsed: float = 0.0


@dataclass
class Row:
    model: str
    math: Probe = field(default_factory=Probe)
    code: Probe = field(default_factory=Probe)
    instr: Probe = field(default_factory=Probe)
    needle: Probe = field(default_factory=Probe)
    billing: Probe = field(default_factory=Probe)
    speed: Probe = field(default_factory=Probe)
    ttft: float = 0.0
    tps: float = 0.0
    prompt_tokens: int = 0
    error: str = ""
    interrupted: bool = False

    @property
    def score(self) -> int:
        return sum(1 for p in (self.math, self.code, self.instr, self.needle) if p.ok)

    @property
    def percent(self) -> float:
        return self.score / 4 * 100

    @property
    def verdict(self) -> str:
        if self.interrupted:
            return "中断"
        if self.score == 4:
            return "优秀"
        if self.score == 3:
            return "良好"
        if self.score == 2:
            return "及格"
        return "不合格"


def line(model: str, check: str, ok: bool, message: str, elapsed: float) -> str:
    icon = GREEN("✅") if ok else RED("❌")
    head = f"{DIM('[')}{DIM('模型 ')}{CYAN(model)}{DIM(' │ ')}{MAGENTA(pad(check, 4))}{DIM(']')}"
    return f"{head} {icon} {message} {DIM(f'[{elapsed * 1000:.0f} ms]')}"


# ---------- 请求 ----------

RETRY_CODES = {408, 409, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}


def brief(status: int, text: str) -> str:
    """把 HTML 错误页压成一句话，别让它刷屏。"""
    body = (text or "").lstrip()
    if body.startswith("<"):
        hints = {
            524: "网关等待源站超时",
            522: "连接源站超时",
            520: "源站返回未知错误",
            502: "网关错误",
            503: "服务暂不可用",
        }
        return f"HTTP {status}（{hints.get(status, '网关返回 HTML 错误页')}）"
    return f"HTTP {status}: {body[:100]}"


def post(url, key, model, messages, max_tokens, stream=False, timeout=180, retries=2):
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

    last = "请求失败"
    for attempt in range(retries + 1):
        started = time.perf_counter()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout, stream=stream)
        except requests.RequestException as exc:
            last = f"{type(exc).__name__}（第 {attempt + 1} 次）"
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(last) from exc

        if resp.status_code == 200:
            break

        last = brief(resp.status_code, resp.text)
        if resp.status_code in RETRY_CODES and attempt < retries:
            time.sleep(3 * (attempt + 1))
            continue
        raise RuntimeError(last)
    else:
        raise RuntimeError(last)

    if not stream:
        body = resp.json()
        elapsed = time.perf_counter() - started
        text = body["choices"][0]["message"].get("content") or ""
        return text, body.get("usage") or {}, elapsed, 0.0

    ttft, chunks, usage = 0.0, [], {}
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        data = raw[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        if obj.get("usage"):
            usage = obj["usage"]
        for choice in obj.get("choices") or []:
            piece = (choice.get("delta") or {}).get("content")
            if piece:
                if not ttft:
                    ttft = time.perf_counter() - started
                chunks.append(piece)
    total = time.perf_counter() - started
    return "".join(chunks), usage, total, ttft


# ---------- 判分器 ----------

def judge_math(text: str) -> tuple[bool, str]:
    clean = text.strip()
    numbers = re.findall(r"\d+", clean)
    if numbers and int(numbers[0]) == MATH_ANSWER:
        return True, f"利润率算对（{MATH_ANSWER}%）"
    return False, f"答成了 {clean[:30] or '空'}"


def judge_code(code: str, timeout: float = 15.0) -> tuple[bool, str]:
    if "def top_k" not in code:
        return False, "没有定义 top_k"
    harness = (
        code
        + "\n\nimport json\ncases = "
        + repr(CODE_CASES)
        + "\nout = []\n"
        "for nums, k, _ in cases:\n"
        "    try:\n"
        "        out.append(top_k(list(nums), k))\n"
        "    except Exception as exc:\n"
        "        out.append('ERR:' + type(exc).__name__)\n"
        "print(json.dumps(out, ensure_ascii=False))\n"
    )
    try:
        done = subprocess.run(
            [sys.executable, "-c", harness], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return False, f"执行超时（{timeout:.0f}s）"
    if done.returncode != 0:
        tail = (done.stderr.strip().splitlines() or ["?"])[-1]
        return False, "运行报错：" + tail[:50]
    try:
        got = json.loads(done.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return False, "输出无法解析"
    expect = [want for _, _, want in CODE_CASES]
    passed = sum(1 for a, b in zip(got, expect) if a == b)
    ok = passed == len(expect)
    return ok, f"{passed}/{len(expect)} 用例通过"


def judge_instr(text: str) -> tuple[bool, str]:
    rows = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    problems = []
    if len(rows) != 2:
        problems.append(f"{len(rows)} 行")
    else:
        if len(rows[0]) != 4:
            problems.append(f"首行 {len(rows[0])} 字")
        if len(rows[1]) > 10:
            problems.append(f"次行 {len(rows[1])} 字")
    if "的" in text:
        problems.append("出现「的」")
    return (not problems), ("、".join(problems) if problems else "两行 / 无「的」")


def judge_needle(text: str) -> tuple[bool, str]:
    if NEEDLE in text:
        return True, f"找回暗号 {NEEDLE}"
    snippet = text.strip()[:26].replace("\n", " ")
    return False, f"没找回：{snippet or '空'}"


# ---------- 单个模型 ----------

def run_model(url, key, model, emit) -> Row:
    row = Row(model=model)

    def step(name, fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            row.error = f"{type(exc).__name__}: {exc}"[:140]
            row.interrupted = True
            return None

    def do_math():
        text, _, el, _ = post(url, key, model, [{"role": "user", "content": MATH_PROMPT}], 200)
        ok, msg = judge_math(text)
        row.math = Probe(ok, msg, el)
        emit("推理", ok, msg, el)

    def do_code():
        text, _, el, _ = post(url, key, model, [{"role": "user", "content": CODE_PROMPT}], 800)
        code = re.sub(r"^```(?:python)?|```$", "", text.strip(), flags=re.M).strip()
        ok, msg = judge_code(code)
        row.code = Probe(ok, msg, el)
        emit("代码", ok, msg, el)

    def do_instr():
        text, _, el, _ = post(url, key, model, [{"role": "user", "content": INSTR_PROMPT}], 160)
        ok, msg = judge_instr(text)
        row.instr = Probe(ok, msg, el)
        emit("指令", ok, msg, el)

    def do_needle():
        text, _, el, _ = post(
            url, key, model,
            [{"role": "user", "content": LONG_TEXT + "\n\n请只输出文中出现的暗号，不要其他内容。"}],
            64,
        )
        ok, msg = judge_needle(text)
        row.needle = Probe(ok, msg, el)
        emit("长文", ok, msg, el)

    def do_billing():
        _, usage, el, _ = post(url, key, model, [{"role": "user", "content": "hi"}], 5)
        n = int((usage or {}).get("prompt_tokens") or 0)
        row.prompt_tokens = n
        ok = 0 < n <= 500
        msg = f"hi 计 {n} token" if ok else f"hi 竟计 {n} token，疑似虚报"
        row.billing = Probe(ok, msg, el)
        emit("计费", ok, msg, el)

    def do_speed():
        text, usage, total, ttft = post(
            url, key, model, [{"role": "user", "content": STREAM_PROMPT}], 300, stream=True
        )
        out = int((usage or {}).get("completion_tokens") or 0) or max(1, len(text) // 2)
        row.ttft, row.tps = ttft, out / total if total > 0 else 0.0
        ok = ttft > 0 and ttft <= 5.0
        msg = (
            f"首字 {ttft:.2f}s，{row.tps:.1f} t/s"
            if ok
            else f"首字 {ttft:.2f}s，太慢（{row.tps:.1f} t/s）"
        )
        row.speed = Probe(ok, msg, total)
        emit("速度", ok, msg, total)

    for fn in (do_math, do_code, do_instr, do_needle, do_billing, do_speed):
        step(fn.__name__, fn)
        if row.error:
            break
    return row


# ---------- 自检 ----------

def selftest() -> int:
    good_code = (
        "def top_k(nums, k):\n"
        "    if not nums or k < 1 or k > len(nums):\n"
        "        return None\n"
        "    return sorted(nums, reverse=True)[k-1]\n"
    )
    bad_code = "def top_k(nums, k):\n    return max(nums)\n"

    cases = [
        ("推理-答20应通过", judge_math("20")[0], True),
        ("推理-答25应失败", judge_math("25")[0], False),
        ("代码-正确实现应通过", judge_code(good_code)[0], True),
        ("代码-错误实现应失败", judge_code(bad_code)[0], False),
        ("指令-四字+十字应通过", judge_instr("一丝不苟\n做事认真细致")[0], True),
        ("指令-含「的」应失败", judge_instr("一丝不苟\n认真做事的样")[0], False),
        ("指令-首行五字应失败", judge_instr("一丝不苟啊\n做事认真")[0], False),
        ("指令-一行应失败", judge_instr("一丝不苟")[0], False),
        ("长文-含暗号应通过", judge_needle("紫罗兰-7391")[0], True),
        ("长文-不含暗号应失败", judge_needle("紫罗兰")[0], False),
    ]
    failed = 0
    for name, got, want in cases:
        ok = got == want
        failed += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {name}  (得到 {got}，期望 {want})")
    print(f"\n判分器自检：{len(cases) - failed}/{len(cases)} 通过")
    return 0 if failed == 0 else 1


def dry_run(emit_factory) -> list[Row]:
    fake = [
        ("deepseek-v4.1-flash", [1, 1, 1, 1], 1, 2.49, 42.3, 5),
        ("gpt-6-sol", [1, 1, 1, 1], 1, 3.44, 37.9, 46),
        ("gpt-6.1-sol", [1, 1, 1, 1], 1, 4.80, 23.0, 5),
        ("claude-sonnet-5-5", [1, 1, 1, 1], 1, 3.21, 35.4, 10),
        ("claude-opus-5-5", [1, 1, 1, 1], 1, 11.82, 23.4, 10),
        ("gpt-6-astra", [1, 1, 1, 1], 0, 3.15, 17.2, 18200),
        ("grok-4.7", [1, 1, 1, 0], 0, 40.97, 60.9, 3337),
    ]
    rows = []
    for name, flags, billok, ttft, tps, tok in fake:
        emit = emit_factory(name)
        row = Row(model=name, ttft=ttft, tps=tps, prompt_tokens=tok)
        specs = [
            ("math", "推理", "利润率算对（20%）", "答成了 25%", 3.4),
            ("code", "代码", "5/5 用例通过", "4/5 用例通过", 4.1),
            ("instr", "指令", "两行 / 无「的」", "次行 12 字", 2.2),
            ("needle", "长文", "找回暗号 紫罗兰-7391", "没找回：我不能按要求只输出那段暗号。", 1.8),
        ]
        for flag, (key, label, msg_pass, msg_fail, sec) in zip(flags, specs):
            ok = bool(flag)
            msg = msg_pass if ok else msg_fail
            setattr(row, key, Probe(ok, msg, sec))
            emit(label, ok, msg, sec)
        msg = f"hi 计 {tok} token" if billok else f"hi 竟计 {tok} token，疑似虚报"
        row.billing = Probe(bool(billok), msg, 0.6)
        emit("计费", bool(billok), msg, 0.6)
        sp_ok = ttft <= 5
        sp_msg = f"首字 {ttft:.2f}s，{tps:.1f} t/s"
        if not sp_ok:
            sp_msg += "，太慢"
        row.speed = Probe(sp_ok, sp_msg, 2.0)
        emit("速度", sp_ok, sp_msg, 2.0)
        rows.append(row)
    return rows


def main() -> int:
    global USE_COLOR
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default=DEFAULT_URL)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--out", default="bench_report")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    USE_COLOR = not args.no_color
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    def emit_factory(model):
        def emit(check, ok, message, elapsed):
            print(line(model, check, ok, message, elapsed))
        return emit

    print(f"接口：{args.api_url}")
    print(f"检测模式：横评；模型：{len(models)} 个；每模型 6 次请求")
    print("超时：180 秒；温度：0\n")

    if args.dry_run:
        rows = dry_run(emit_factory)
    else:
        key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            print("没有找到 API_KEY 环境变量。", file=sys.stderr)
            return 2
        rows = []
        for idx, model in enumerate(models, 1):
            print(DIM(f"── {idx}/{len(models)} {model} ──"))
            row = run_model(args.api_url, key, model, emit_factory(model))
            rows.append(row)
            if row.error:
                print(RED(f"   中断：{row.error}"))

    for row in rows:
        if row.interrupted:
            print(f"{'模型 ' + CYAN(row.model)}: 中断，已得 {row.score}/4 —— {row.error}")
            continue
        print(
            f"{'模型 ' + CYAN(row.model)}: {row.percent:.2f}/100，{row.verdict}，"
            f"通过 {row.score}，失败 {4 - row.score}，"
            f"首字 {row.ttft:.2f}s，速度 {row.tps:.1f} t/s"
        )
    print()

    done = [r for r in rows if not r.interrupted]
    dropped = [r for r in rows if r.interrupted]
    total = len(done) * 4
    passed = sum(r.score for r in done)
    avg = sum(r.percent for r in done) / len(done) if done else 0
    print(DIM("─" * 72))
    print(
        f"汇总：{len(rows)} 个模型（完成 {len(done)}，中断 {len(dropped)}），"
        f"{total} 项能力检测，通过 {passed}，失败 {total - passed}，平均分 {avg:.2f}"
    )

    dirty = [r for r in rows if r.prompt_tokens > 500]
    if dirty:
        print("计费异常：" + "、".join(f"{r.model}({r.prompt_tokens})" for r in dirty))
    clean = [r for r in rows if 0 < r.prompt_tokens <= 500]
    if clean:
        print("计费正常：" + "、".join(r.model for r in clean))

    winners = sorted(done, key=lambda x: (-x.score, x.ttft if x.ttft else 999))
    if winners:
        best = winners[0]
        print(f"综合最优：{best.model}（{best.score}/4，首字 {best.ttft:.2f}s，{best.tps:.1f} t/s）")

    payload = {"rows": [asdict(r) for r in rows]}
    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(DIM(f"已写入 {args.out}.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
