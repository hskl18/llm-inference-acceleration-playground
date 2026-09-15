from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Protocol


TOKENIZERS_ENCODE_METHOD = "tokenizers.encode(add_special_tokens=false)"


class TokenCounter(Protocol):
    method: str

    def count(self, text: str) -> int: ...


class TokenizersCounter:
    method = TOKENIZERS_ENCODE_METHOD

    def __init__(self, tokenizer: object) -> None:
        self._tokenizer = tokenizer

    def count(self, text: str) -> int:
        encoding = self._tokenizer.encode(text, add_special_tokens=False)  # type: ignore[attr-defined]
        return len(encoding.ids)


def is_local_tokenizer_reference(tokenizer: str) -> bool:
    path = Path(tokenizer).expanduser()
    return path.is_absolute() or tokenizer.startswith((".", "~")) or path.exists()


@lru_cache(maxsize=8)
def load_token_counter(tokenizer: str, revision: str) -> TokenCounter:
    try:
        from tokenizers import Tokenizer  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "the tokenizers package is required for model-token benchmark metrics"
        ) from exc
    local_path = Path(tokenizer).expanduser()
    if local_path.exists():
        tokenizer_file = local_path / "tokenizer.json" if local_path.is_dir() else local_path
        loaded = Tokenizer.from_file(str(tokenizer_file))
    else:
        loaded = Tokenizer.from_pretrained(tokenizer, revision=revision)
    return TokenizersCounter(loaded)
