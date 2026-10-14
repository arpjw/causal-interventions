"""Pinned, offline vLLM adapter for CUDA development pilots."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from time import monotonic
from typing import Any

from .runner import Generation


class LocalVLLMBackend:
    def __init__(self, snapshot: Path, thinking: bool = False,
                 tensor_parallel_size: int = 1,
                 json_schema: dict[str, Any] | None = None) -> None:
        from vllm import LLM

        if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
            raise ValueError("model snapshot must be an existing local directory")
        if tensor_parallel_size < 1:
            raise ValueError("tensor parallel size must be positive")
        self.thinking = thinking
        self.json_schema = json_schema
        self.llm = LLM(
            model=str(snapshot), trust_remote_code=False,
            dtype="auto", max_model_len=8192 if thinking else 4096,
            gpu_memory_utilization=0.82,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, prompt: str, max_new_tokens: int) -> Generation:
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams

        chat_options = {"add_generation_prompt": True, "tokenize": False}
        if self.thinking:
            chat_options["enable_thinking"] = True
        else:
            chat_options["enable_thinking"] = False
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], **chat_options,
        )
        seed = int.from_bytes(sha256(prompt.encode()).digest()[:4], "big")
        started = monotonic()
        if not self.thinking:
            settings = SamplingParams(
                temperature=0.7, top_p=0.8, top_k=20,
                max_tokens=max_new_tokens, seed=seed,
                guided_decoding=(GuidedDecodingParams(json=self.json_schema)
                                 if self.json_schema is not None
                                 else GuidedDecodingParams(json_object=True)),
            )
            response = self.llm.generate([rendered], sampling_params=settings,
                                         use_tqdm=False)[0]
            completion = response.outputs[0]
            return Generation(completion.text, len(response.prompt_token_ids),
                              len(completion.token_ids), monotonic() - started)

        if max_new_tokens <= 256:
            raise ValueError("thinking mode needs a token cap above 256")
        thought_budget = min(1536, max_new_tokens - 256)
        thought_settings = SamplingParams(
            temperature=0.6, top_p=0.95, top_k=20,
            max_tokens=thought_budget, seed=seed,
        )
        first = self.llm.generate([rendered], sampling_params=thought_settings,
                                  use_tqdm=False)[0]
        thought = first.outputs[0]
        if "</think>" in thought.text:
            answer = thought.text.split("</think>", 1)[1].strip()
            try:
                complete = isinstance(json.loads(answer), dict)
            except (TypeError, ValueError):
                complete = False
            if complete:
                return Generation(answer, len(first.prompt_token_ids),
                                  len(thought.token_ids), monotonic() - started,
                                  raw_text=thought.text)
            prefix = rendered + thought.text.split("</think>", 1)[0] + "</think>\n"
        else:
            prefix = rendered + thought.text + "\n</think>\n"
        final_settings = SamplingParams(
            temperature=0.6, top_p=0.95, top_k=20,
            max_tokens=max_new_tokens - len(thought.token_ids), seed=seed + 1,
            guided_decoding=(GuidedDecodingParams(json=self.json_schema)
                             if self.json_schema is not None
                             else GuidedDecodingParams(json_object=True)),
        )
        second = self.llm.generate([prefix], sampling_params=final_settings,
                                   use_tqdm=False)[0]
        final = second.outputs[0]
        return Generation(final.text, len(first.prompt_token_ids) + len(second.prompt_token_ids),
                          len(thought.token_ids) + len(final.token_ids), monotonic() - started,
                          raw_text=thought.text + "\n</think>\n" + final.text)
