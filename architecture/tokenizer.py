from __future__ import annotations
import os
from typing import List

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def train_tokenizer(
    output_path: str,
    vocab_size: int = 16384,
    num_samples: int = 100_000,
) -> None:
    from datasets import load_dataset
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    special_tokens = ["<|pad|>", "<|eos|>", "<|bos|>", "<|unk|>", "<|sep|>"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    dataset = load_dataset("bigcode/the-stack-smol", streaming=True, split="train")

    def text_iterator():
        count = 0
        for sample in dataset:
            if count >= num_samples:
                break
            yield sample["content"]
            count += 1

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer, length=num_samples)
    tokenizer.decoder = decoders.ByteLevel()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tokenizer.save(output_path)


class DynamoTokenizer:
    def __init__(self, path: str) -> None:
        self._tok = Tokenizer.from_file(path)

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._tok.token_to_id("<|pad|>")

    @property
    def eos_id(self) -> int:
        return self._tok.token_to_id("<|eos|>")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train Dynamo BPE tokenizer")
    p.add_argument("--output", default="tokenizer/dynamo.json")
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--num-samples", type=int, default=100_000)
    args = p.parse_args()

    print(f"Training tokenizer on {args.num_samples} samples → {args.output}")
    train_tokenizer(args.output, args.vocab_size, args.num_samples)

    tok = DynamoTokenizer(args.output)
    sample = "def hello_world():\n    print('Hello, world!')"
    ids = tok.encode(sample)
    decoded = tok.decode(ids)
    print(f"Vocab size: {tok.vocab_size}")
    print(f"pad_id={tok.pad_id}  eos_id={tok.eos_id}")
    print(f"Sample encode (first 10): {ids[:10]}")
    print(f"Round-trip match: {decoded == sample}")
