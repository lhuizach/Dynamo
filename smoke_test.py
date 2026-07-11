#!/usr/bin/env python
"""
CPU smoke test — verifies the full pipeline with a tiny model.
No GPU or network access required.
Run: python smoke_test.py
"""
from __future__ import annotations
import json
import os
import sys
import tempfile

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from architecture.model import Dynamo, ModelConfig
from architecture.prompt import build_prompt, sanitize_router_output
from architecture.tokenizer import DynamoTokenizer


SAMPLE_FILE = "05_dataset_sample.jsonl"


def _build_mini_tokenizer(save_path: str, vocab_size: int = 512) -> None:
    texts = []
    with open(SAMPLE_FILE) as f:
        for line in f:
            r = json.loads(line)
            texts += [r["user_request"], r["precise_instruction"], r["code"]]

    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    special = ["<|pad|>", "<|eos|>", "<|bos|>", "<|unk|>", "<|sep|>"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer=trainer)
    tok.decoder = decoders.ByteLevel()
    tok.save(save_path)


def _build_sft_sample(
    record: dict,
    tokenizer: DynamoTokenizer,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt = build_prompt(
        user_request=record["user_request"],
        precise_instruction=record["precise_instruction"],
    )
    # encode separately so the prompt/code boundary is exact
    prompt_ids = tokenizer.encode(prompt)
    code_ids = tokenizer.encode(record["code"]) + [tokenizer.eos_id]
    full_ids = (prompt_ids + code_ids)[: max_len + 1]

    x = torch.tensor(full_ids[:-1], dtype=torch.long)
    y = torch.tensor(full_ids[1:], dtype=torch.long)

    mask = torch.zeros(len(y), dtype=torch.bool)
    code_start = min(max(0, len(prompt_ids) - 1), len(y))
    mask[code_start:] = True
    if not mask.any():  # prompt longer than window — train on everything
        mask[:] = True
    return x, y, mask


def _check_sanitizer() -> None:
    want = "Write a Python function named `retry` that retries a callable."
    cases = [
        want,
        f"Sure, here's the instruction: {want}",
        f"Instruction:\n{want}",
        f'"{want}"',
        f"```\n{want}\n```",
        f"{want}\n\nThis instruction covers the retry behaviour you asked for.",
        f"Write a Python function named `retry`\nthat retries a callable.",
    ]
    for raw in cases:
        got = sanitize_router_output(raw)
        assert got == want, f"sanitize failed:\n  raw={raw!r}\n  got={got!r}"
    assert sanitize_router_output("   ") == ""
    long = sanitize_router_output("word " * 1000)
    assert len(long) <= 1200 and not long.endswith(" ")


def main() -> None:
    tmpdir = tempfile.mkdtemp()
    tok_path = os.path.join(tmpdir, "mini.json")

    print("0/5  sanitizing router output edge cases...")
    _check_sanitizer()
    print("     preamble/fence/quote/multiline/truncation cases OK")

    print("1/5  building mini tokenizer from sample records (no network)...")
    _build_mini_tokenizer(tok_path)
    tokenizer = DynamoTokenizer(tok_path)
    print(f"     vocab_size={tokenizer.vocab_size}  pad_id={tokenizer.pad_id}  eos_id={tokenizer.eos_id}")

    print("2/5  instantiating tiny Dynamo model...")
    config = ModelConfig(
        dim=64,
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        ffn_dim=128,
        max_seq_len=1024,
        vocab_size=tokenizer.vocab_size,
    )
    model = Dynamo(config)
    params = sum(p.numel() for p in set(model.parameters()))
    print(f"     params={params:,}  ({params / 1e6:.3f}M)")

    print("3/5  forward pass + cross-entropy loss...")
    x = torch.randint(0, tokenizer.vocab_size, (1, 32))
    y = torch.randint(0, tokenizer.vocab_size, (1, 32))
    logits, loss = model(x, y)
    assert logits.shape == (1, 32, tokenizer.vocab_size), f"unexpected shape {logits.shape}"
    assert loss is not None and loss.item() > 0
    print(f"     logits {tuple(logits.shape)}  loss={loss.item():.4f}  OK")

    print("4/5  SFT step on sample record (prompt-masked loss, plain AdamW)...")
    with open(SAMPLE_FILE) as f:
        record = json.loads(f.readline())
    x_sft, y_sft, mask = _build_sft_sample(record, tokenizer, max_len=config.max_seq_len)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    logits_sft, _ = model(x_sft.unsqueeze(0))
    loss_sft = F.cross_entropy(logits_sft.squeeze(0)[mask], y_sft[mask])
    loss_sft.backward()
    opt.step()
    opt.zero_grad()
    print(f"     sft loss={loss_sft.item():.4f}  code tokens={mask.sum().item()}  OK")

    print("5/5  generate() and checkpoint round-trip...")
    model.eval()
    prompt_ids = tokenizer.encode("def retry")
    idx = torch.tensor([prompt_ids], dtype=torch.long)
    out = model.generate(idx, max_new_tokens=16, temperature=1.0)
    assert out.shape == (1, len(prompt_ids) + 16), f"unexpected shape {out.shape}"
    ckpt_path = os.path.join(tmpdir, "smoke.pt")
    torch.save({"config": config, "model": model.state_dict()}, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    reloaded = Dynamo(ckpt["config"])
    reloaded.load_state_dict(ckpt["model"])
    print(f"     generated {out.shape[1]} tokens  checkpoint saved + reloaded  OK")

    print("\nAll smoke tests passed.")
    print(f"(temp files in {tmpdir})")
    print()
    print("Note: the router (Qwen2.5-Coder-0.5B-Instruct) is NOT tested here.")
    print("To test the full pipeline end-to-end, run:")
    print("  python dynamo_cli.py --checkpoint dynamo/final.pt --tokenizer tokenizer/dynamo.json 'hello'")


if __name__ == "__main__":
    main()
