from __future__ import annotations

import re
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFLoraClient:
    def __init__(
        self,
        base_model_name: str,
        adapter_path: str,
        max_new_tokens: int = 256,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
        return text

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        decoded = self._strip_think_tags(decoded)
        decoded = self._strip_code_fences(decoded)
        return decoded