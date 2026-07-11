# Dynamo

A compound coding AI with a two-stage pipeline:

1. **Router** — Qwen2.5-Coder-0.5B-Instruct (frozen) converts vague requests into precise technical instructions
2. **Dynamo** — custom 760M-parameter transformer trained from scratch generates code from those instructions

## Architecture

| Component | Value |
|-----------|-------|
| Dimensions | 2048 |
| Layers | 16 |
| Query heads | 16 |
| KV heads | 4 (GQA) |
| FFN dim | 5632 (SwiGLU) |
| Vocab size | 16384 (BPE) |
| Max seq len | 4096 |
| Positional encoding | RoPE |

## Quickstart

```bash
pip install -r requirements.txt
```

### Train tokenizer

```bash
python -m architecture.tokenizer
```

Downloads `bigcode/the-stack-smol` (~2.6GB) and fits the BPE vocabulary on
100k samples drawn evenly across all 30 languages. The dataset stores each
language as a contiguous 10k-row block, so sampling must be language-balanced
— fitting the vocab on the head of the raw stream produces a single-language
tokenizer that cripples pretraining on everything else.

> **Note:** retraining the tokenizer changes the meaning of every token ID.
> Checkpoints trained with a previous tokenizer cannot be resumed — training
> records the tokenizer's SHA-256 in each checkpoint and refuses mismatches.
> After retraining the tokenizer, start pretraining fresh (no `--resume`) and
> use an empty HF Hub checkpoint repo.

### Pretrain

```bash
# single GPU
python training/train.py --tokenizer tokenizer/dynamo.json --output dynamo/

# multi-GPU (DDP) — one process per GPU; each rank trains a disjoint shard
# of the data stream, so tokens per optimizer step scale with process count
torchrun --standalone --nproc_per_node=2 training/train.py \
  --tokenizer tokenizer/dynamo.json --output dynamo/
```

Keep the launch mode, `--batch-size`, and `--grad-accum` identical across
every session of one run — a resumed session uses them to compute how much
of the data stream previous sessions consumed. Mechanics are covered by
`python scripts/ddp_smoke_test.py` (2-process CPU run, no GPU/network).

### SFT

```bash
python training/sft.py \
  --checkpoint dynamo/checkpoint_final.pt \
  --data dynamo-train/ \
  --output dynamo/
```

### Inference

```bash
# single request
python dynamo_cli.py "make a retry function"

# interactive REPL with router output visible
python dynamo_cli.py --interactive --show-router
```

## File Map

```
architecture/      model definition (model.py), tokenizer training (tokenizer.py),
                   router↔model prompt contract and output sanitizer (prompt.py)
training/          pretraining loop (train.py), SFT (sft.py), dataset streaming (data.py)
inference/         two-stage router + dynamo pipeline (pipeline.py)
dynamo_cli.py      CLI entrypoint (argparse, single-shot and REPL modes)
dynamo/            trained model checkpoints — git-ignored, created after training
dynamo-train/      SFT training data in JSONL format — git-ignored
tokenizer/         trained tokenizer JSON — git-ignored, created after tokenizer training
03_modelfile.txt   legacy copy of the prompt template — the canonical
                   definition lives in architecture/prompt.py
05_dataset_sample.jsonl   example SFT records showing the expected data format
```
