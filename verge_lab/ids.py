"""Deterministic identifiers used throughout Verge Lab artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Return a compact, reproducible JSON representation.

    Non-finite floats are rejected so an identifier can never describe data that
    strict JSON cannot represent.
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_id(prefix: str, *parts: Any, length: int = 12) -> str:
    """Create a readable content hash from JSON-serializable values."""

    if not prefix or not prefix.replace("-", "").replace("_", "").isalnum():
        raise ValueError("prefix must contain only letters, digits, '-' or '_'")
    if length < 8 or length > 64:
        raise ValueError("length must be between 8 and 64")
    digest = hashlib.sha256(canonical_json(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:length]}"
