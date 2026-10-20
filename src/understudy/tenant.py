"""Tenant bindings: where one customer's instance of a vendor app lives, and how
that customer tightens the vendor policy. A capability never names a host; the
binding supplies it at run time."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
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


def load_tenant(path: Path) -> TenantBinding:
    return TenantBinding.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
