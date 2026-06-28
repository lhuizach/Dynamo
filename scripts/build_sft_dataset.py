"""
Build a balanced SFT dataset from multiple HuggingFace sources.
Targets: JavaScript, HTML/CSS, C, Python.
Output: dynamo-train/sft_data.jsonl

Usage:
    python scripts/build_sft_dataset.py
    python scripts/build_sft_dataset.py --per-lang 600 --output dynamo-train/sft_data.jsonl
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from collections import defaultdict


# ── language detection ────────────────────────────────────────────────────────

JS_SIGNALS  = ["function ", "const ", "let ", "var ", "=>", "document.", "console.", ".addEventListener", "async ", "await ", "require(", "module.exports", "querySelector", "addEventListener", "Promise", "JSON."]
HTML_SIGNALS = ["<html", "<div", "<span", "<p>", "<head>", "<body", "<!DOCTYPE", "<nav", "<form", "<input", "<button", "<ul>", "<li>", "<a href", "<img", "<style>", "<script>"]
CSS_SIGNALS  = ["{", "margin:", "padding:", "display:", "color:", "background:", "font-size:", "width:", "height:", "border:", "flex", "grid", "@media", "position:", "transform:"]
C_SIGNALS    = ["#include", "int main", "printf(", "malloc(", "free(", "typedef", "struct ", "void *", "NULL", "sizeof(", "->", "fprintf(", "fopen(", "return 0;", "#define"]
PY_SIGNALS   = ["def ", "import ", "class ", "print(", "if __name__", "self.", "lambda ", "with open", "raise ", "yield ", "async def", "from ", "isinstance(", "dict(", "list("]


def detect_lang(code: str) -> str:
    scores: dict[str, int] = {
        "javascript": sum(1 for s in JS_SIGNALS if s in code),
        "html":       sum(1 for s in HTML_SIGNALS if s in code),
        "css":        sum(1 for s in CSS_SIGNALS if s in code),
        "c":          sum(1 for s in C_SIGNALS if s in code),
        "python":     sum(1 for s in PY_SIGNALS if s in code),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "other"


# ── cleaning ──────────────────────────────────────────────────────────────────

def clean_code(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```\w*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def is_useful(code: str, min_lines: int) -> bool:
    lines = [l for l in code.splitlines() if l.strip()]
    return len(lines) >= min_lines and len(code) >= 50


def make_precise(instruction: str, inp: str) -> str:
    instr = instruction.strip()
    if inp and inp.strip():
        return f"{instr} Input: {inp.strip()}"
    return instr


def make_request(instruction: str) -> str:
    instr = instruction.strip()
    if len(instr) <= 80:
        return instr
    first = re.split(r"[.!?]", instr)[0].strip()
    return first if len(first) > 10 else instr[:80]


# ── sources ───────────────────────────────────────────────────────────────────

SOURCES = [
    # large multi-language alpaca-style dataset
    ("TokenBender/code_instructions_122k_alpaca_style", "train"),
    # original CodeAlpaca
    ("sahil2801/CodeAlpaca-20k", "train"),
    # Python-heavy but high quality
    ("iamtarun/python_code_instructions_18k_alpaca", "train"),
]

TARGET_LANGS = {"javascript", "html", "css", "c", "python"}


def process_row(row: dict) -> dict | None:
    instruction = (row.get("instruction") or row.get("prompt") or "").strip()
    inp         = (row.get("input") or "").strip()
    output      = (row.get("output") or row.get("response") or row.get("completion") or "").strip()

    if not instruction or not output:
        return None

    code = clean_code(output)
    if not is_useful(code, min_lines=3):
        return None

    lang = detect_lang(code)
    if lang not in TARGET_LANGS:
        return None

    # for CSS, also accept if instruction mentions CSS/style
    if lang == "other" and any(w in instruction.lower() for w in ["css", "style", "html", "navbar", "flexbox", "grid"]):
        lang = "css"

    return {
        "user_request": make_request(instruction),
        "precise_instruction": make_precise(instruction, inp),
        "code": code,
        "_lang": lang,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-lang", type=int, default=600, help="Max records per language")
    p.add_argument("--output", default="dynamo-train/sft_data.jsonl")
    p.add_argument("--min-lines", type=int, default=3)
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    buckets: dict[str, list[dict]] = defaultdict(list)
    targets = {lang: args.per_lang for lang in TARGET_LANGS}

    for ds_name, split in SOURCES:
        remaining = {k: v for k, v in targets.items() if len(buckets[k]) < v}
        if not remaining:
            break

        print(f"\nLoading {ds_name}...")
        try:
            dataset = load_dataset(ds_name, split=split, streaming=True)
        except Exception as e:
            print(f"  Skipped ({e})")
            continue

        for row in dataset:
            if not any(len(buckets[k]) < targets[k] for k in TARGET_LANGS):
                break
            record = process_row(row)
            if record is None:
                continue
            lang = record.pop("_lang")
            if len(buckets[lang]) < targets[lang]:
                buckets[lang].append(record)

        for lang in TARGET_LANGS:
            print(f"  {lang}: {len(buckets[lang])}")

    # write balanced output
    all_records: list[dict] = []
    for lang, records in buckets.items():
        all_records.extend(records)

    # interleave languages so the file isn't all one language at a time
    from itertools import zip_longest
    lists = list(buckets.values())
    interleaved = [r for group in zip_longest(*lists) for r in group if r is not None]

    with open(args.output, "w") as f:
        for record in interleaved:
            f.write(json.dumps(record) + "\n")

    total = len(interleaved)
    print(f"\nTotal: {total} records → {args.output}")
    print("\nBreakdown:")
    for lang, records in sorted(buckets.items()):
        print(f"  {lang:12s}: {len(records)}")

    print("\nSample records:")
    for lang in TARGET_LANGS:
        if buckets[lang]:
            r = buckets[lang][0]
            print(f"\n[{lang}] {r['user_request'][:60]}")


if __name__ == "__main__":
    main()
