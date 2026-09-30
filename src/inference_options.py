"""Validated, opt-in llama.cpp tuning; defaults preserve the assistant."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class InferenceOptions:
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    flash_attn: str = "auto"
    spec_type: str = "none"
    parallel: int | None = None
    cache_reuse: int | None = None
    image_min_tokens: int | None = None
    image_max_tokens: int | None = None
    mtmd_batch_max_tokens: int | None = None
    slot_save_path: str | None = None

    def __post_init__(self):
        types = {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"}
        if self.cache_type_k not in types or self.cache_type_v not in types:
            raise ValueError("Unsupported KV cache type")
        if self.flash_attn not in {"auto", "on", "off"}:
            raise ValueError("flash_attn must be auto/on/off")
        if self.spec_type not in {"none", "ngram-simple", "ngram-cache"}:
            raise ValueError("This project supports none/ngram-simple/ngram-cache (no draft model)")
        for name in ("parallel", "image_min_tokens", "image_max_tokens", "mtmd_batch_max_tokens"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        if self.cache_reuse is not None and self.cache_reuse < 0:
            raise ValueError("cache_reuse cannot be negative")
        if self.image_min_tokens and self.image_max_tokens and self.image_min_tokens > self.image_max_tokens:
            raise ValueError("image_min_tokens exceeds image_max_tokens")

    @classmethod
    def from_environment(cls, prefix: str):
        values = {}
        names = {"CTK": "cache_type_k", "CTV": "cache_type_v", "FLASH_ATTN": "flash_attn",
                 "SPEC_TYPE": "spec_type", "PARALLEL": "parallel", "CACHE_REUSE": "cache_reuse",
                 "IMAGE_MIN_TOKENS": "image_min_tokens", "IMAGE_MAX_TOKENS": "image_max_tokens",
                 "MTMD_BATCH_MAX_TOKENS": "mtmd_batch_max_tokens", "SLOT_SAVE_PATH": "slot_save_path"}
        for suffix, field in names.items():
            value = os.getenv(f"{prefix}_{suffix}")
            if value:
                values[field] = value if field in {"cache_type_k", "cache_type_v", "flash_attn", "spec_type", "slot_save_path"} else int(value)
        return asdict(cls(**values))

    def arguments(self, *, vision: bool) -> list[str]:
        args = []
        for field, flag, default in (("cache_type_k", "--cache-type-k", "f16"),
                                     ("cache_type_v", "--cache-type-v", "f16"),
                                     ("flash_attn", "--flash-attn", "auto"),
                                     ("spec_type", "--spec-type", "none"),
                                     ("parallel", "--parallel", None),
                                     ("cache_reuse", "--cache-reuse", None),
                                     ("slot_save_path", "--slot-save-path", None)):
            value = getattr(self, field)
            if value != default:
                args += [flag, str(value)]
        image_fields = ("image_min_tokens", "image_max_tokens", "mtmd_batch_max_tokens")
        if not vision and any(getattr(self, field) is not None for field in image_fields):
            raise ValueError("Image token budgets require a vision model")
        for field in image_fields:
            value = getattr(self, field)
            if value is not None:
                args += ["--" + field.replace("_", "-"), str(value)]
        return args

    def prepare_paths(self):
        if self.slot_save_path:
            Path(self.slot_save_path).mkdir(parents=True, exist_ok=True)
