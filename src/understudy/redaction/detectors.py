"""Regex detectors: the backstop behind field-aware redaction.

They catch what classification missed, in free text such as error messages, page
snapshots and DOM dumps. They are deliberately greedy: a masked digit run in a log
costs nothing, a leaked one cannot be taken back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from understudy.artifact.schema import Sensitivity

JSESSIONID = re.compile(r";jsessionid=[^/?#\s\"'<>]*", re.IGNORECASE)


@dataclass(frozen=True)
class Detector:
    name: str
    pattern: re.Pattern[str]
    sensitivity: Sensitivity


# Order matters: secrets before the digit-shaped detectors that would otherwise
# claim part of a token first.
DETECTORS: tuple[Detector, ...] = (
    Detector("bearer", re.compile(r"(?i)\bbearer\s+[\w.~+/-]+=*"), "secret"),
    Detector("jwt", re.compile(r"\beyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]{8,}"), "secret"),
    Detector("api_key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), "secret"),
    Detector("hex_token", re.compile(r"\b[0-9A-Fa-f]{32,}\b"), "secret"),
    Detector("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "pii"),
    Detector("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), "pii"),
    Detector("card", re.compile(r"\b\d(?:[ -]?\d){12,18}\b"), "financial_id"),
    Detector("long_digits", re.compile(r"\b\d{9,}\b"), "financial_id"),
    # Account ids are often short (ParaBank's have five digits). Years and small
    # counts survive; anything id-length does not.
    Detector("digit_run", re.compile(r"\b\d{5,}\b"), "financial_id"),
)

SECRET_FIELD = re.compile(r"pass(word|code|phrase)?|\bpin\b|secret|token|otp|cvv|cvc|csc")
PII_FIELD = re.compile(
    r"ssn|social security|tax ?id|e-?mail|phone|mobile|birth|\bdob\b|address|zip|postal"
    r"|first ?name|last ?name|full ?name|surname|user ?(name|id)|login ?id"
)
FINANCIAL_FIELD = re.compile(r"card ?(number|no)|\biban\b|routing|account ?(number|no|#)")

SECRET_AUTOCOMPLETE = frozenset({"current-password", "new-password", "one-time-code", "cc-csc"})
PII_AUTOCOMPLETE = frozenset(
    {"email", "tel", "name", "given-name", "family-name", "bday", "street-address"}
    | {"postal-code", "address-line1", "address-line2"}
)
FINANCIAL_AUTOCOMPLETE = frozenset({"cc-number", "cc-exp", "cc-name"})
