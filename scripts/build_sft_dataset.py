"""
Download CodeAlpaca-20k from HuggingFace and convert to Dynamo SFT format.
Output: dynamo-train/sft_data.jsonl

Usage:
    python scripts/build_sft_dataset.py
    python scripts/build_sft_dataset.py --max-samples 500 --output dynamo-train/sft_data.jsonl
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys


def clean_code(text: str) -> str:
    text = text.strip()
    # strip markdown code fences if present
    text = re.sub(r"^```\w*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def is_code(text: str) -> bool:
    code_signals = ["def ", "class ", "import ", "for ", "while ", "if ", "return ", "print(", "=>", "function ", "const ", "var ", "let "]
    return any(s in text for s in code_signals)


def make_precise_instruction(instruction: str, inp: str) -> str:
    if inp and inp.strip():
        return f"{instruction.strip()} Input: {inp.strip()}"
    return instruction.strip()


def make_user_request(instruction: str) -> str:
    # shorten to a vague natural-language version
    instr = instruction.strip()
    # if already short enough, use as-is
    if len(instr) < 80:
        return instr
    # take first sentence
    first = re.split(r'[.!?]', instr)[0].strip()
    return first if len(first) > 10 else instr[:80]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-samples", type=int, default=2000)
    p.add_argument("--output", default="dynamo-train/sft_data.jsonl")
    p.add_argument("--min-code-lines", type=int, default=3)
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f"Downloading CodeAlpaca-20k...")
    dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

    written = 0
    skipped = 0

    with open(args.output, "w") as f:
        for row in dataset:
            if written >= args.max_samples:
                break

            instruction = row.get("instruction", "").strip()
            inp = row.get("input", "").strip()
            output = row.get("output", "").strip()

            if not instruction or not output:
                skipped += 1
                continue

            code = clean_code(output)

            # filter: must look like code and have enough lines
            if not is_code(code):
                skipped += 1
                continue
            if len(code.splitlines()) < args.min_code_lines:
                skipped += 1
                continue

            record = {
                "user_request": make_user_request(instruction),
                "precise_instruction": make_precise_instruction(instruction, inp),
                "code": code,
            }
            f.write(json.dumps(record) + "\n")
            written += 1

    print(f"Written {written} records to {args.output} ({skipped} skipped)")
    print(f"\nSample record:")
    with open(args.output) as f:
        print(json.dumps(json.loads(f.readline()), indent=2))


if __name__ == "__main__":
    main()
