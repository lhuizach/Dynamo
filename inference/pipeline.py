from __future__ import annotations
import os
import sys
from typing import Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architecture.model import Dynamo, ModelConfig
from architecture.tokenizer import DynamoTokenizer


PROMPT_TEMPLATE = "User request: {user_request}\nRouter output: {precise_instruction}\n"

_ROUTER_SYSTEM = (
    "Convert the user's vague coding request into a single precise technical "
    "instruction a code model can follow exactly. Output only the instruction."
)


class RouterModel:
    _instance: Optional[RouterModel] = None

    def __new__(cls) -> RouterModel:
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._loaded = False
            cls._instance = obj
        return cls._instance

    def _load(self) -> None:
        if self._loaded:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._hf_tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-0.5B-Instruct"
        )
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self._hf_model.eval()
        self._loaded = True

    def translate(self, user_request: str) -> str:
        self._load()
        messages = [
            {"role": "system", "content": _ROUTER_SYSTEM},
            {"role": "user", "content": user_request},
        ]
        text = self._hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._hf_tokenizer(text, return_tensors="pt").to(self._hf_model.device)
        with torch.no_grad():
            output = self._hf_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                repetition_penalty=1.1,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        return self._hf_tokenizer.decode(generated, skip_special_tokens=True).strip()


class DynamoPipeline:
    def __init__(
        self,
        checkpoint: str,
        tokenizer_path: str,
        temperature: float = 0.2,
        max_new_tokens: int = 512,
    ) -> None:
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt = torch.load(checkpoint, map_location="cpu")
        config: ModelConfig = ckpt["config"]
        self.model = Dynamo(config).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.tokenizer = DynamoTokenizer(tokenizer_path)
        self._router = RouterModel()

    def run(self, user_request: str, show_router: bool = False) -> str:
        precise_instruction = self._router.translate(user_request)
        if show_router:
            print(f"[router] {precise_instruction}")

        prompt = PROMPT_TEMPLATE.format(
            user_request=user_request,
            precise_instruction=precise_instruction,
        )
        input_ids = self.tokenizer.encode(prompt)
        idx = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        output_ids = self.model.generate(
            idx,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        generated = output_ids[0][len(input_ids) :].tolist()
        return self.tokenizer.decode(generated)
