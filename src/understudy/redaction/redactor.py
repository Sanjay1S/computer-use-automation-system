"""Redaction, applied before anything is written.

Three layers, strongest first:
1. field-aware: a value's sensitivity comes from its input spec or the field it
   was typed into (label, input type, autocomplete);
2. value-aware: every non-public value the run has handled is masked wherever it
   reappears, e.g. inside a templated locator echoed in an error message;
3. regex detectors, for whatever the first two never saw.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from understudy.artifact.schema import Sensitivity
from understudy.redaction import detectors as det

SENSITIVE_QUERY_KEYS = frozenset(
    {"jsessionid", "sessionid", "session", "token", "access_token", "password", "passwd"}
    | {"secret", "code", "auth", "apikey", "api_key", "ssn"}
)

# `- textbox "Payee Name:" [ref=e7]: typed value`. The name is matched as a quoted
# string because legacy labels end in colons; [ \t] keeps it on one line.
_SNAPSHOT_VALUE = re.compile(
    r'^(?P<head>[ \t]*- (?:textbox|searchbox|spinbutton|combobox)(?: "(?:[^"\\]|\\.)*")?'
    r"(?: \[[^\]\n]*\])*):[ \t]+\S.*$",
    re.MULTILINE,
)
_DIGIT_RUN = re.compile(r"\d{4,}")

KEY_ENV = "UNDERSTUDY_REDACTION_KEY"


class Redactor:
    def __init__(self, key: bytes, known: Mapping[str, Sensitivity] | None = None) -> None:
        self._key = key
        # Longest first, so a value that contains another is masked whole.
        self._known = sorted(
            ((v, s) for v, s in (known or {}).items() if s != "public" and len(v) >= 3),
            key=lambda item: -len(item[0]),
        )

    @classmethod
    def from_env(cls) -> Redactor:
        """A stable key lets `[PII:…]` tags correlate across runs. Without one, a
        random key still hides values but tags only correlate within this process."""
        configured = os.environ.get(KEY_ENV)
        return cls(configured.encode("utf-8") if configured else secrets.token_bytes(32))

    def knowing(self, values: Mapping[str, Sensitivity]) -> Redactor:
        merged: dict[str, Sensitivity] = dict(self._known)
        merged.update(values)
        return Redactor(self._key, merged)

    # --- single values -----------------------------------------------------------

    def value(self, value: str, sensitivity: Sensitivity) -> str:
        if sensitivity == "public":
            return value
        if sensitivity == "secret":
            return "[SECRET]"
        if sensitivity == "pii":
            digest = hmac.new(self._key, value.encode("utf-8"), hashlib.sha256).hexdigest()
            # Letters only, so the digit-run detectors never re-mask a tag.
            return "[PII:" + "".join(chr(ord("a") + int(c, 16)) for c in digest[:8]) + "]"
        # At most a third of the value, and never more than four: a 5-digit ParaBank
        # account id keeps one digit, a 16-digit card number keeps the usual four.
        keep = min(4, len(value) // 3)
        return "••••" + (value[-keep:] if keep else "")

    @staticmethod
    def classify_field(
        *,
        input_type: str | None = None,
        label: str | None = None,
        name_attr: str | None = None,
        autocomplete: str | None = None,
    ) -> Sensitivity:
        if input_type == "password":
            return "secret"
        hint = (autocomplete or "").lower().split()
        if any(h in det.SECRET_AUTOCOMPLETE for h in hint):
            return "secret"
        if any(h in det.FINANCIAL_AUTOCOMPLETE for h in hint):
            return "financial_id"
        if any(h in det.PII_AUTOCOMPLETE for h in hint):
            return "pii"
        words = " ".join(filter(None, (label, name_attr))).lower().replace("_", " ")
        words = words.replace(".", " ").replace("-", " ")
        if det.SECRET_FIELD.search(words):
            return "secret"
        if det.FINANCIAL_FIELD.search(words):
            return "financial_id"
        if det.PII_FIELD.search(words):
            return "pii"
        if input_type == "email" or input_type == "tel":
            return "pii"
        return "public"

    # --- free text -----------------------------------------------------------------

    def text(self, text: str) -> str:
        out = det.JSESSIONID.sub("", text)
        for known, sensitivity in self._known:
            out = out.replace(known, self.value(known, sensitivity))
        for detector in det.DETECTORS:
            out = detector.pattern.sub(self._masker(detector.sensitivity), out)
        return out

    def _masker(self, sensitivity: Sensitivity) -> Callable[[re.Match[str]], str]:
        return lambda match: self.value(match.group(0), sensitivity)

    def url(self, url: str) -> str:
        parts = urlsplit(det.JSESSIONID.sub("", url))
        query = [
            (k, "[SECRET]" if k.lower() in SENSITIVE_QUERY_KEYS else self.text(v))
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
        ]
        # Built by hand rather than urlencode()d: this string is for reading in a log,
        # and percent-encoding the masks would only hide what was masked.
        rendered = "&".join(f"{k}={v}" for k, v in query)
        return urlunsplit((parts.scheme, parts.netloc, self.text(parts.path), rendered, ""))

    def snapshot(self, snapshot: str) -> str:
        """An accessibility snapshot for the evidence folder: typed values dropped,
        digit runs masked (option lists are full of account numbers)."""
        out = _SNAPSHOT_VALUE.sub(lambda m: m.group("head") + ": [value]", snapshot)
        return _DIGIT_RUN.sub(lambda m: "#" * len(m.group(0)), self.text(out))

    def html(self, html: str) -> str:
        """A DOM dump whose input values were already stripped in the page."""
        return _DIGIT_RUN.sub(lambda m: "#" * len(m.group(0)), self.text(html))

    def data(self, payload: Any) -> Any:
        """Every string in a JSON-shaped payload, redacted as free text."""
        if isinstance(payload, str):
            return self.text(payload)
        if isinstance(payload, Mapping):
            return {k: self.data(v) for k, v in payload.items()}
        if isinstance(payload, list | tuple):
            return [self.data(v) for v in payload]
        return payload
