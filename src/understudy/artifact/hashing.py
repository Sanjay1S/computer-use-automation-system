"""Canonical serialization and content hashing.

Approval binds to the hash, so the bytes that are hashed must not depend on key
order, float formatting or which optional fields happen to be set. The lifecycle
block is excluded: approving an artifact would otherwise change its hash and
invalidate the approval it just recorded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from understudy.artifact.schema import Capability


def canonical_json(capability: Capability) -> str:
    """The artifact as it is written to disk: stable key order, two-space indent."""
    return json.dumps(_dump(capability), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def content_hash(capability: Capability) -> str:
    """sha256 over everything but the lifecycle block."""
    payload = _dump(capability)
    payload.pop("lifecycle", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load(path: Path) -> Capability:
    return Capability.model_validate_json(path.read_text(encoding="utf-8"))


def save(capability: Capability, path: Path) -> str:
    """Write the artifact and return its content hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(capability), encoding="utf-8")
    return content_hash(capability)


def _dump(capability: Capability) -> dict[str, Any]:
    return capability.model_dump(mode="json", by_alias=True, exclude_none=True)
