#!/usr/bin/env python
"""DDP smoke test — drives the real training loop multi-process on CPU.

No GPU, network, or HF token required: the gated dataset is replaced with a
synthetic corpus and the model is tiny. Exercises the pieces that only break
under torchrun: process-group setup, gradient sync with no_sync accumulation,
collective skip/health/divergence decisions, rank-0 checkpointing, and the
resume fast-forward with sharded data.

Run:
    python scripts/ddp_smoke_test.py            # orchestrates both phases
    torchrun --standalone --nproc_per_node=2 scripts/ddp_smoke_test.py --phase fresh --workdir DIR
"""
from __future__ import annotations
import argparse
import glob
import hashlib
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def _fake_corpus():
    from datasets import Dataset
    return Dataset.from_dict({
        "content": [f"def fn_{i}(x):\n    return x + {i}\n" * 4 for i in range(60)],
        "lang": ["python"] * 60,
    })


def _train_args(workdir: str, resume: bool, max_steps: int) -> argparse.Namespace:
    return argparse.Namespace(
        tokenizer=os.path.join(workdir, "mini.json"),
        output=os.path.join(workdir, "ckpts"),
        dim=64, n_layers=2, n_heads=2, n_kv_heads=2, ffn_dim=128,
        seq_len=64, batch_size=2, grad_accum=2,
        max_lr=1e-3, min_lr=1e-4, warmup_steps=2, max_steps=max_steps,
        log_every=1, save_every=3, languages=None,
        no_bnb=True, no_grad_checkpoint=False,
        resume=resume, resume_step=None,
        hf_repo=None, hf_save_every=None,
        gist_id=None, github_token=None,
        wandb_project=None, wandb_run_name=None,
    )


def run_phase(phase: str, workdir: str) -> None:
    import datasets
    datasets.load_dataset = lambda *a, **k: _fake_corpus()

    from training.train import main as train_main

    if phase == "fresh":
        train_main(_train_args(workdir, resume=False, max_steps=6))
        if int(os.environ.get("RANK", "0")) == 0:
            saved = glob.glob(os.path.join(workdir, "ckpts", "checkpoint_0*.pt"))
            assert saved, "fresh phase saved no periodic checkpoint"
    else:
        train_main(_train_args(workdir, resume=True, max_steps=10))
        if int(os.environ.get("RANK", "0")) == 0:
            import torch
            final = torch.load(
                os.path.join(workdir, "ckpts", "checkpoint_final.pt"),
                map_location="cpu", weights_only=False,
            )
            assert final["step"] == 10, f"resume did not reach step 10 (got {final['step']})"
            with open(os.path.join(workdir, "mini.json"), "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            assert final["tokenizer_sha256"] == sha, "tokenizer fingerprint missing/wrong in checkpoint"


def orchestrate() -> None:
    import tempfile
    workdir = tempfile.mkdtemp(prefix="ddp_smoke_")

    from smoke_test import _build_mini_tokenizer
    os.chdir(REPO_ROOT)  # _build_mini_tokenizer reads the sample jsonl relatively
    _build_mini_tokenizer(os.path.join(workdir, "mini.json"))

    for phase in ("fresh", "resume"):
        print(f"\n=== torchrun 2-process CPU phase: {phase} ===")
        subprocess.run(
            [
                "torchrun", "--standalone", "--nproc_per_node=2",
                os.path.abspath(__file__), "--phase", phase, "--workdir", workdir,
            ],
            check=True,
            cwd=REPO_ROOT,
            timeout=600,
        )
    print("\nDDP smoke test passed: fresh run, checkpointing, and sharded resume all work.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["fresh", "resume"], default=None)
    p.add_argument("--workdir", default=None)
    args = p.parse_args()

    if args.phase is None:
        orchestrate()
    else:
        run_phase(args.phase, args.workdir)
