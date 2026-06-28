from __future__ import annotations
import argparse
import glob
import json
import math
import os
import sys
import time
import urllib.request
from typing import Optional

from datasets import load_dataset as _load_dataset  # noqa: F401 — must import before torch on Windows
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architecture.model import Dynamo, ModelConfig
from architecture.tokenizer import DynamoTokenizer
from training.data import CodeDataset


def _push_gist(gist_id: str, token: str, content: str) -> None:
    try:
        data = json.dumps({"files": {"training_log.jsonl": {"content": content}}}).encode()
        req = urllib.request.Request(
            f"https://api.github.com/gists/{gist_id}",
            data=data, method="PATCH",
            headers={"Authorization": f"token {token}", "Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


def cosine_lr(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int,
) -> float:
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain Dynamo from scratch")
    p.add_argument("--tokenizer", default="tokenizer/dynamo.json")
    p.add_argument("--output", default="dynamo/")
    p.add_argument("--dim", type=int, default=2048)
    p.add_argument("--n-layers", type=int, default=16)
    p.add_argument("--n-heads", type=int, default=16)
    p.add_argument("--n-kv-heads", type=int, default=4)
    p.add_argument("--ffn-dim", type=int, default=5632)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--max-lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--languages", nargs="*", default=None)
    p.add_argument("--no-bnb", action="store_true", help="Use AdamW instead of 8-bit Adam (use if bitsandbytes crashes)")
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in output dir")
    p.add_argument("--gist-id", default=None, help="GitHub Gist ID to stream metrics to (for remote monitoring)")
    p.add_argument("--github-token", default=None, help="GitHub token with gist write permission")
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    tokenizer = DynamoTokenizer(args.tokenizer)
    config = ModelConfig(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        ffn_dim=args.ffn_dim,
        max_seq_len=args.seq_len,
        vocab_size=tokenizer.vocab_size,
    )

    model = Dynamo(config).to(device)
    for block in model.layers:
        block.use_checkpoint = True

    step = 0
    if args.resume:
        candidates = sorted(glob.glob(os.path.join(args.output, "checkpoint_[0-9]*.pt")))
        if candidates:
            ckpt_path = candidates[-1]
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"])
            step = ckpt["step"]
            print(f"Resumed from {ckpt_path} (step {step})")
        else:
            print("No checkpoint found — starting from scratch")

    if args.no_bnb:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr, betas=(0.9, 0.95))
        print("Using AdamW (--no-bnb)")
    else:
        import bitsandbytes as bnb
        optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.max_lr, betas=(0.9, 0.95))
        print("Using Adam8bit (bitsandbytes)")

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda" and not use_bf16))

    dataset = CodeDataset(tokenizer, args.seq_len, args.languages)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=(device == "cuda"),
        persistent_workers=True,
    )

    log_path = os.path.join(args.output, "training_log.jsonl")
    if not args.resume:
        open(log_path, "w").close()  # reset log on fresh run

    micro_step = step * args.grad_accum
    tokens_seen = 0
    loss_accum = 0.0
    t0 = time.time()
    optimizer.zero_grad()

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
            _, loss = model(x, y)
            loss = loss / args.grad_accum

        scaler.scale(loss).backward()
        loss_accum += loss.item()
        micro_step += 1
        tokens_seen += x.numel()

        if micro_step % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            lr = cosine_lr(step, args.max_lr, args.min_lr, args.warmup_steps, args.max_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            if step % args.log_every == 0:
                dt = time.time() - t0
                tok_per_sec = tokens_seen / dt if dt > 0 else 0.0
                print(
                    f"step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e} | {tok_per_sec:.0f} tok/s"
                )
                entry = json.dumps({
                    "step": step, "loss": round(loss_accum, 4),
                    "lr": lr, "tok_per_sec": round(tok_per_sec),
                    "max_steps": args.max_steps,
                })
                with open(log_path, "a") as f:
                    f.write(entry + "\n")
                if args.gist_id and args.github_token:
                    with open(log_path) as f:
                        _push_gist(args.gist_id, args.github_token, f.read())

            if step > 0 and step % args.save_every == 0:
                ckpt = os.path.join(args.output, f"checkpoint_{step:06d}.pt")
                torch.save({"step": step, "model": model.state_dict(), "config": config}, ckpt)

            loss_accum = 0.0
            step += 1
            if step >= args.max_steps:
                break

    final = os.path.join(args.output, "checkpoint_final.pt")
    torch.save({"step": step, "model": model.state_dict(), "config": config}, final)
    print(f"Saved final checkpoint → {final}")


if __name__ == "__main__":
    main(parse_args())
