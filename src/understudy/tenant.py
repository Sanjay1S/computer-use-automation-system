"""Tenant bindings: where one customer's instance of a vendor app lives, and how
that customer tightens the vendor policy. A capability never names a host; the
binding supplies it at run time."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from understudy.policy.models import PolicyOverlay


class TenantBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["understudy.tenant/v1"] = "understudy.tenant/v1"
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    vendor_app: str
    base_url: str
    """Scheme, host, optional port and the path the app is mounted at. No query."""
    sandbox: bool = False
    """Discovery sends page content to a model, so it only runs where this is true."""
    policy: PolicyOverlay = Field(default_factory=PolicyOverlay)

    @field_validator("base_url")
    @classmethod
    def _well_formed(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("base_url must be an absolute http(s) URL")
        if parts.query or parts.fragment or ";" in parts.path:
            raise ValueError("base_url takes no query, fragment or path parameters")
        return value.rstrip("/")

    @property
    def scheme(self) -> str:
        return urlsplit(self.base_url).scheme

    @property
    def host(self) -> str:
        return (urlsplit(self.base_url).hostname or "").lower()

    @property
    def port(self) -> int:
        parts = urlsplit(self.base_url)
        return parts.port or (443 if parts.scheme == "https" else 80)

    @property
    def base_path(self) -> str:
        return urlsplit(self.base_url).path.rstrip("/")


_ENV_REF = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def load_tenant(path: Path, env: Mapping[str, str] | None = None) -> TenantBinding:
    """String values may reference the environment as `${NAME}` or
    `${NAME:-default}`, so one setting (e.g. PARABANK_PORT) can drive both the
    container mapping and the tenant URL. Expanded after parsing, so a value can
    never change the document's structure."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TenantBinding.model_validate(_expand(raw, os.environ if env is None else env, path))


def _expand(value: Any, env: Mapping[str, str], path: Path) -> Any:
    if isinstance(value, str):

        def one(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in env and env[name] != "":
                return env[name]
            if default is None:
                raise ValueError(f"{path}: ${{{name}}} is not set and has no default")
            return default

        return _ENV_REF.sub(one, value)
    if isinstance(value, dict):
        return {k: _expand(v, env, path) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, env, path) for v in value]
    return value
