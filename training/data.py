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
        skip_sequences: int = 0,
        base_seed: int = 42,
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.languages = languages
        self.num_samples = num_samples
        self.skip_sequences = skip_sequences
        self.base_seed = base_seed

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        from datasets import load_dataset
        # the-stack-smol stores its 30 languages as contiguous 10k-row blocks.
        # A streaming shuffle buffer smaller than or comparable to a block
        # (e.g. 10k) barely mixes across block boundaries, so the model still
        # sees long single-language runs — which shows up as loss drift
        # unrelated to any actual training instability. The dataset is only
        # ~2.6GB, so load it non-streaming and do a true full-permutation
        # shuffle instead of an approximate windowed one.
        dataset = load_dataset("bigcode/the-stack-smol", split="train")

        # The stream is deterministic given base_seed, so a resumed run can
        # fast-forward past everything an earlier session already trained on
        # by counting yielded sequences (skipping still pays the tokenizer
        # cost, but that is minutes, not GPU-hours). Each pass over the data
        # reshuffles with a new seed so multi-epoch training doesn't repeat
        # one fixed order. When num_samples is set the dataset is a single
        # bounded pass (used for quick experiments); otherwise it is
        # infinite and the training loop's --max-steps is the terminator.
        to_skip = self.skip_sequences
        epoch = 0
        while True:
            shuffled = dataset.shuffle(seed=self.base_seed + epoch)
            buffer: List[int] = []
            count = 0

            for sample in shuffled:
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
                    if to_skip > 0:
                        to_skip -= 1
                        continue
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    yield x, y

            if self.num_samples is not None:
                break
            epoch += 1
