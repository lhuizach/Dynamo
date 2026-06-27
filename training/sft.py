from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from typing import List, Tuple

import torch
import torch.nn.functional as F
import bitsandbytes as bnb
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architecture.model import Dynamo, ModelConfig
from architecture.tokenizer import DynamoTokenizer


PROMPT_TEMPLATE = "User request: {user_request}\nRouter output: {precise_instruction}\n"


def load_records(data_dir: str) -> List[dict]:
    records: List[dict] = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def build_sample(
    record: dict,
    tokenizer: DynamoTokenizer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt = PROMPT_TEMPLATE.format(
        user_request=record["user_request"],
        precise_instruction=record["precise_instruction"],
    )
    full = prompt + record["code"]
    prompt_ids = tokenizer.encode(prompt)
    full_ids = tokenizer.encode(full) + [tokenizer.eos_id]

    x = torch.tensor(full_ids[:-1], dtype=torch.long)
    y = torch.tensor(full_ids[1:], dtype=torch.long)

    # mask prompt positions so loss is only computed on code tokens
    mask = torch.ones(len(y), dtype=torch.bool)
    mask[: max(0, len(prompt_ids) - 1)] = False

    return x, y, mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supervised fine-tuning for Dynamo")
    p.add_argument("--checkpoint", required=True, help="Path to pretrained .pt checkpoint")
    p.add_argument("--data", default="dynamo-train/", help="Directory with JSONL training files")
    p.add_argument("--output", default="dynamo/", help="Output directory for final checkpoint")
    p.add_argument("--tokenizer", default="tokenizer/dynamo.json")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config: ModelConfig = ckpt["config"]
    tokenizer = DynamoTokenizer(args.tokenizer)

    model = Dynamo(config).to(device)
    model.load_state_dict(ckpt["model"])

    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and not use_bf16))

    records = load_records(args.data)
    if not records:
        raise ValueError(f"No JSONL records found in {args.data}")

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for record in tqdm(records, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            x, y, mask = build_sample(record, tokenizer)
            x = x.unsqueeze(0).to(device)
            y = y.to(device)
            mask = mask.to(device)

            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
                logits, _ = model(x)
                logits = logits.squeeze(0)  # (T, vocab)
                loss = F.cross_entropy(logits[mask], y[mask])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg = total_loss / max(1, len(records))
        print(f"Epoch {epoch + 1}/{args.epochs} | avg loss {avg:.4f}")

    final = os.path.join(args.output, "final.pt")
    torch.save({"config": config, "model": model.state_dict()}, final)
    print(f"Saved SFT checkpoint → {final}")


if __name__ == "__main__":
    main(parse_args())
