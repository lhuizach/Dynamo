#!/usr/bin/env bash
set -euo pipefail

# 1. Train the tokenizer (streams 100k samples from bigcode/the-stack-smol)
python -m architecture.tokenizer

# 2. Pretrain Dynamo (~760M params, targets 8GB VRAM)
python training/train.py \
  --tokenizer tokenizer/dynamo.json \
  --output dynamo/ \
  --batch-size 1 \
  --grad-accum 16 \
  --max-steps 100000

# 3. SFT on dynamo-train/ data
python training/sft.py \
  --checkpoint dynamo/checkpoint_final.pt \
  --data dynamo-train/ \
  --output dynamo/ \
  --epochs 3

# 4. Run inference (interactive mode)
python dynamo_cli.py --interactive \
  --checkpoint dynamo/final.pt \
  --tokenizer tokenizer/dynamo.json \
  --show-router
