from __future__ import annotations
import os
import sys
from typing import Iterator, List, Optional, Tuple

import torch
from torch.utils.data import IterableDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architecture.tokenizer import DynamoTokenizer


class CodeDataset(IterableDataset):
    def __init__(
        self,
        tokenizer: DynamoTokenizer,
        seq_len: int,
        languages: Optional[List[str]] = None,
        num_samples: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.languages = languages
        self.num_samples = num_samples

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        from datasets import load_dataset
        # the-stack-smol stores its 30 languages as contiguous 10k-row blocks.
        # A streaming shuffle buffer smaller than or comparable to a block
        # (e.g. 10k) barely mixes across block boundaries, so the model still
        # sees long single-language runs — which shows up as loss drift
        # unrelated to any actual training instability. The dataset is only
        # ~2.6GB, so load it non-streaming and do a true full-permutation
        # shuffle instead of an approximate windowed one.
        dataset = load_dataset("bigcode/the-stack-smol", split="train").shuffle(seed=42)
        buffer: List[int] = []
        count = 0

        for sample in dataset:
            if self.num_samples is not None and count >= self.num_samples:
                break
            if self.languages is not None and sample.get("lang") not in self.languages:
                continue

            ids = self.tokenizer.encode(sample["content"]) + [self.tokenizer.eos_id]
            buffer.extend(ids)
            count += 1

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len + 1 :]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y
