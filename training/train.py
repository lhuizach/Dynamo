from __future__ import annotations
import argparse
import glob
import json
import math
import os
import shutil
import sys
import threading
import time
import urllib.request
from collections import deque
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


def _hf_upload_checkpoint(repo_id: str, token: str, local_path: str) -> threading.Thread:
    """Upload a checkpoint to HF Hub in a background thread. Returns the thread."""
    def _do_upload() -> None:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            fname = os.path.basename(local_path)
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=f"checkpoints/{fname}",
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )
            print(f"[HF Hub] Uploaded {fname} → {repo_id}")
            # prune old HF checkpoints — keep only 2 most recent
            all_files = list(api.list_repo_files(repo_id=repo_id, repo_type="model", token=token))
            old_ckpts = sorted(
                f for f in all_files
                if f.startswith("checkpoints/checkpoint_") and f.endswith(".pt")
            )[:-2]
            for old in old_ckpts:
                api.delete_file(path_in_repo=old, repo_id=repo_id, repo_type="model", token=token)
                print(f"[HF Hub] Deleted old checkpoint {old}")
        except Exception as e:
            print(f"[HF Hub] Upload failed: {e}")

    t = threading.Thread(target=_do_upload, daemon=True)
    t.start()
    return t


def _hf_download_latest_checkpoint(
    repo_id: str, token: str, output_dir: str, resume_step: Optional[int] = None,
) -> Optional[str]:
    """Download a checkpoint from HF Hub. If resume_step is given, fetch that exact
    step; otherwise fetch the latest. Returns local path or None."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.utils import RepositoryNotFoundError
        api = HfApi(token=token)
        try:
            all_files = list(api.list_repo_files(repo_id=repo_id, repo_type="model", token=token))
        except RepositoryNotFoundError:
            print(f"[HF Hub] Repo {repo_id!r} not found — will be created on first checkpoint save")
            return None
        ckpt_files = sorted(
            f for f in all_files
            if f.startswith("checkpoints/checkpoint_") and f.endswith(".pt")
        )
        if not ckpt_files:
            print(f"[HF Hub] No checkpoints found in {repo_id}")
            return None
        if resume_step is not None:
            wanted = f"checkpoints/checkpoint_{resume_step:06d}.pt"
            if wanted not in ckpt_files:
                print(f"[HF Hub] {wanted} not found in {repo_id}. Available: {ckpt_files}")
                return None
            latest = wanted
        else:
            latest = ckpt_files[-1]
        print(f"[HF Hub] Downloading {latest} from {repo_id} ...")
        dl_path = hf_hub_download(
            repo_id=repo_id,
            filename=latest,
            repo_type="model",
            token=token,
            local_dir=output_dir,
        )
        # hf_hub_download nests the file under output_dir/checkpoints/ — move it up one level
        dest = os.path.join(output_dir, os.path.basename(latest))
        if os.path.abspath(dl_path) != os.path.abspath(dest):
            shutil.move(dl_path, dest)
            subdir = os.path.join(output_dir, "checkpoints")
            if os.path.isdir(subdir) and not os.listdir(subdir):
                os.rmdir(subdir)
        print(f"[HF Hub] Saved to {dest}")
        return dest
    except Exception as e:
        print(f"[HF Hub] Download failed: {e}")
        return None


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
    p.add_argument("--no-grad-checkpoint", action="store_true", help="Disable activation checkpointing — faster but uses more VRAM")
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint (local or HF Hub)")
    p.add_argument("--resume-step", type=int, default=None, help="Resume from this exact step instead of the latest checkpoint (use to roll back past a corrupted save)")
    p.add_argument("--hf-repo", default=None, help="HuggingFace Hub repo for persistent checkpoint storage (e.g. username/dynamo-checkpoints)")
    p.add_argument("--hf-save-every", type=int, default=None, help="Upload to HF Hub every N steps (default: same as --save-every)")
    p.add_argument("--gist-id", default=None, help="GitHub Gist ID to stream metrics to (for remote monitoring)")
    p.add_argument("--github-token", default=None, help="GitHub token with gist write permission")
    p.add_argument("--wandb-project", default=None, help="Weights & Biases project name — streams a live dashboard (requires WANDB_API_KEY env var)")
    p.add_argument("--wandb-run-name", default=None, help="Optional W&B run name (default: auto-generated)")
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    print(f"Config: {vars(args)}")
    os.makedirs(args.output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    hf_save_every = args.hf_save_every if args.hf_save_every is not None else args.save_every

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        print(f"[W&B] Live dashboard: {wandb_run.url}")

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
    if not args.no_grad_checkpoint:
        for block in model.layers:
            block.use_checkpoint = True

    step = 0
    resumed_from_checkpoint = False
    if args.resume:
        if args.resume_step is not None:
            wanted = os.path.join(args.output, f"checkpoint_{args.resume_step:06d}.pt")
            ckpt_path: Optional[str] = wanted if os.path.exists(wanted) else None
            if ckpt_path:
                print(f"Found local checkpoint: {ckpt_path}")
            elif args.hf_repo and hf_token:
                ckpt_path = _hf_download_latest_checkpoint(
                    args.hf_repo, hf_token, args.output, resume_step=args.resume_step
                )
        else:
            candidates = sorted(glob.glob(os.path.join(args.output, "checkpoint_[0-9]*.pt")))
            if candidates:
                ckpt_path = candidates[-1]
                print(f"Found local checkpoint: {ckpt_path}")
            elif args.hf_repo and hf_token:
                ckpt_path = _hf_download_latest_checkpoint(args.hf_repo, hf_token, args.output)
            else:
                ckpt_path = None

        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            bad_tensors = [
                name for name, t in ckpt["model"].items()
                if torch.is_floating_point(t) and not torch.isfinite(t).all()
            ]
            if bad_tensors:
                raise RuntimeError(
                    f"Checkpoint {ckpt_path} contains NaN/Inf in {len(bad_tensors)} tensor(s) "
                    f"(e.g. {bad_tensors[0]}) — this checkpoint is corrupted. "
                    f"Re-run with --resume-step set to an earlier saved step."
                )
            model.load_state_dict(ckpt["model"])
            step = ckpt["step"]
            resumed_from_checkpoint = True
            print(f"Resumed from {ckpt_path} (step {step})")
        else:
            print("No checkpoint found — starting from scratch")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

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
    loader = DataLoader(dataset, batch_size=args.batch_size)
    data_iter = iter(loader)

    log_path = os.path.join(args.output, "training_log.jsonl")
    if not args.resume:
        open(log_path, "w").close()  # reset log on fresh run

    HEALTH_LOSS_THRESHOLD = 15.0

    if resumed_from_checkpoint:
        print("Validating resumed checkpoint against live data before training...")
        model.eval()
        health_losses = []
        with torch.no_grad():
            for check_i in range(5):
                x, y = next(data_iter)
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
                    _, health_loss = model(x, y)
                    health_loss = health_loss.mean()
                if not torch.isfinite(health_loss):
                    raise RuntimeError(
                        f"Resumed checkpoint {ckpt_path} produces non-finite loss on live data "
                        f"(check {check_i + 1}/5) — its weights are individually finite but the "
                        f"model itself is corrupted/unstable. Re-run with --resume-step set to an "
                        f"earlier checkpoint."
                    )
                health_losses.append(health_loss.item())
        avg_health_loss = sum(health_losses) / len(health_losses)
        if avg_health_loss > HEALTH_LOSS_THRESHOLD:
            raise RuntimeError(
                f"Resumed checkpoint {ckpt_path} has an average loss of {avg_health_loss:.2f} over "
                f"a 5-batch live check — far higher than a healthy training loss, meaning this "
                f"checkpoint has already diverged even though its weights are finite. Re-run with "
                f"--resume-step set to an earlier checkpoint."
            )
        model.train()
        print(f"Checkpoint looks healthy (avg loss {avg_health_loss:.2f}) — starting training.")

    micro_step = step * args.grad_accum
    tokens_seen = 0
    loss_accum = 0.0
    t0 = time.time()
    optimizer.zero_grad()

    upload_thread: Optional[threading.Thread] = None
    consecutive_nonfinite = 0
    MAX_CONSECUTIVE_NONFINITE = 20
    NONFINITE_WINDOW = 200
    NONFINITE_RATE_THRESHOLD = 0.15
    nonfinite_window: deque = deque(maxlen=NONFINITE_WINDOW)

    for x, y in data_iter:
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
            _, loss = model(x, y)
            loss = loss.mean() / args.grad_accum

        is_finite = torch.isfinite(loss)
        nonfinite_window.append(0 if is_finite else 1)
        if len(nonfinite_window) == NONFINITE_WINDOW:
            rate = sum(nonfinite_window) / NONFINITE_WINDOW
            if rate > NONFINITE_RATE_THRESHOLD:
                raise RuntimeError(
                    f"{rate:.0%} of the last {NONFINITE_WINDOW} micro-steps produced a non-finite "
                    f"loss at step {step} — the model is gradually diverging even without a single "
                    f"unbroken failure streak. Stop and re-run with --resume-step set to an earlier "
                    f"checkpoint (or restart from scratch)."
                )

        if not is_finite:
            consecutive_nonfinite += 1
            print(f"WARNING: non-finite loss at step {step} (micro_step {micro_step}) — skipping batch [{consecutive_nonfinite}/{MAX_CONSECUTIVE_NONFINITE}]")
            optimizer.zero_grad()
            if consecutive_nonfinite >= MAX_CONSECUTIVE_NONFINITE:
                raise RuntimeError(
                    f"{MAX_CONSECUTIVE_NONFINITE} consecutive non-finite losses at step {step} — "
                    f"the model weights are corrupted, not the data. Stop and re-run with "
                    f"--resume-step set to a checkpoint saved before this point."
                )
            continue
        consecutive_nonfinite = 0

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
                if wandb_run is not None:
                    wandb_run.log({"loss": loss_accum, "lr": lr, "tok_per_sec": tok_per_sec}, step=step)

            if step > 0 and step % args.save_every == 0:
                ckpt = os.path.join(args.output, f"checkpoint_{step:06d}.pt")
                state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save({"step": step, "model": state, "config": config}, ckpt)

                ckpt_is_healthy = all(
                    torch.isfinite(t).all() for t in state.values() if torch.is_floating_point(t)
                )
                if not ckpt_is_healthy:
                    print(
                        f"WARNING: checkpoint at step {step} contains NaN/Inf — NOT uploading to "
                        f"HF Hub to avoid poisoning the persisted checkpoint history."
                    )

                # Wait for any in-flight HF upload before removing old local files
                if upload_thread and upload_thread.is_alive():
                    print("[HF Hub] Waiting for previous upload to finish...")
                    upload_thread.join()

                # keep only the 2 most recent checkpoints to save disk space
                old = sorted(glob.glob(os.path.join(args.output, "checkpoint_[0-9]*.pt")))[:-2]
                for f in old:
                    os.remove(f)

                # Push to HF Hub for cross-session persistence
                if args.hf_repo and hf_token and step % hf_save_every == 0 and ckpt_is_healthy:
                    upload_thread = _hf_upload_checkpoint(args.hf_repo, hf_token, ckpt)

            loss_accum = 0.0
            step += 1
            if step >= args.max_steps:
                break

    # Ensure final HF upload completes before exit
    if upload_thread and upload_thread.is_alive():
        print("[HF Hub] Waiting for final upload to finish...")
        upload_thread.join()

    final = os.path.join(args.output, "checkpoint_final.pt")
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({"step": step, "model": state, "config": config}, final)
    print(f"Saved final checkpoint → {final}")

    if args.hf_repo and hf_token:
        print("[HF Hub] Uploading final checkpoint ...")
        t = _hf_upload_checkpoint(args.hf_repo, hf_token, final)
        t.join()

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main(parse_args())
