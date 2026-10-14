"""Optional offline Transformers backend for a bounded local GPU pilot."""

from __future__ import annotations

from pathlib import Path
from time import monotonic

from .runner import Generation


class LocalHFBackend:
    def __init__(self, snapshot: Path, device: str = "mps") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device != "mps" or not torch.backends.mps.is_available():
            raise ValueError("this pilot requires an available Apple Metal GPU")
        if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
            raise ValueError("model snapshot must be an existing local Hugging Face directory")
        self.device = device
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False,
            dtype=torch.float16, low_cpu_mem_usage=True,
        ).to(device).eval()

    def generate(self, prompt: str, max_new_tokens: int) -> Generation:
        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        encoded = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        inputs = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        started = monotonic()
        with self.torch.inference_mode():
            output = self.model.generate(
                inputs, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        self.torch.mps.synchronize()
        elapsed = monotonic() - started
        completion = output[0, inputs.shape[-1]:]
        return Generation(
            self.tokenizer.decode(completion, skip_special_tokens=True),
            int(inputs.shape[-1]), int(completion.shape[-1]), elapsed,
        )
