"""Optional offline MLX backend for a cached quantized model on Apple GPU."""

from __future__ import annotations

from pathlib import Path
from time import monotonic

from .runner import Generation


class LocalMLXBackend:
    def __init__(self, snapshot: Path) -> None:
        from mlx_lm import load

        if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
            raise ValueError("model snapshot must be an existing local directory")
        self.model, self.tokenizer = load(str(snapshot), trust_remote_code=False)

    def generate(self, prompt: str, max_new_tokens: int) -> Generation:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False,
        )
        started = monotonic()
        segments = []
        final = None
        for response in stream_generate(
            self.model, self.tokenizer, rendered,
            max_tokens=max_new_tokens, sampler=make_sampler(temp=0.0),
        ):
            segments.append(response.text)
            final = response
        return Generation("".join(segments),
                          final.prompt_tokens if final else len(self.tokenizer.encode(rendered)),
                          final.generation_tokens if final else 0,
                          monotonic() - started)
