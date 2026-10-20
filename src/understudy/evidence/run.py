"""The run directory: one per discovery or replay run, written as the run goes.

Nothing reaches disk unredacted. `event()` takes raw values together with their
sensitivity and redacts them itself, so a caller cannot forget to; free-text
fields go through the redactor's text pass whatever their source.
"""

from __future__ import annotations

import json
import secrets
import string
import threading
from collections import deque
from collections.abc import Collection, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from understudy.artifact.schema import Sensitivity, Target
from understudy.redaction.redactor import Redactor

Actor = Literal["agent", "engine", "human", "system"]
EventKind = Literal[
    "observe",
    "decide",
    "act",
    "policy",
    "race",
    "recovery",
    "drift",
    "escalation",
    "control",
    "result",
]


class EventTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    fingerprint: dict[str, str] = Field(default_factory=dict)


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ts: datetime
    run_id: str
    seq: int
    actor: Actor
    kind: EventKind
    step_id: str | None = None
    member: str | None = None
    action: str | None = None
    target: EventTarget | None = None
    locator_used: str | None = None
    value: str | None = None
    url: str | None = None
    rationale: str | None = None
    duration_ms: int | None = None
    ok: bool | None = None
    error: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class RunContext:
    def __init__(
        self,
        root: Path,
        *,
        kind: str,
        redactor: Redactor,
        run_id: str | None = None,
        meta: Mapping[str, Any] | None = None,
        verbatim: Collection[str] = (),
        recent: int = 50,
    ) -> None:
        """`verbatim` names identifier keys in `meta` to keep unredacted (see write_json)."""
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        # Letters only: a hex suffix is all digits 6% of the time, and the redactor's
        # digit backstop then masks the run id as if it were an account number.
        suffix = "".join(secrets.choice(string.ascii_lowercase) for _ in range(6))
        self.run_id = run_id or f"{kind}-{stamp}-{suffix}"
        self.dir = root / self.run_id
        self.dir.mkdir(parents=True, exist_ok=False)
        self.redactor = redactor
        self._lock = threading.Lock()
        self._seq = 0
        self._shots = 0
        self._recent: deque[Event] = deque(maxlen=recent)
        self.write_json(
            "run.json",
            {**dict(meta or {}), "run_id": self.run_id, "kind": kind, "started_at": stamp},
            verbatim=("run_id", "kind", "started_at", *verbatim),
        )

    def know(self, values: Mapping[str, Sensitivity]) -> None:
        """Register the run's non-public values so they are masked wherever they
        reappear, including in strings this module did not build."""
        self.redactor = self.redactor.knowing(values)

    def event(
        self,
        kind: EventKind,
        *,
        actor: Actor,
        step_id: str | None = None,
        member: str | None = None,
        action: str | None = None,
        target: Target | None = None,
        locator_used: str | None = None,
        value: str | None = None,
        value_sensitivity: Sensitivity = "pii",
        url: str | None = None,
        rationale: str | None = None,
        duration_ms: int | None = None,
        ok: bool | None = None,
        error: str | None = None,
        **data: Any,
    ) -> Event:
        """`value_sensitivity` defaults to pii: a value nobody classified is hashed,
        not printed. Pass "public" explicitly for values that are."""
        r = self.redactor
        with self._lock:
            self._seq += 1
            event = Event(
                ts=datetime.now(UTC),
                run_id=self.run_id,
                seq=self._seq,
                actor=actor,
                kind=kind,
                step_id=step_id,
                member=member,
                action=action,
                target=_event_target(target, r),
                locator_used=r.text(locator_used) if locator_used else None,
                value=r.value(value, value_sensitivity) if value is not None else None,
                url=r.url(url) if url else None,
                rationale=r.text(rationale) if rationale else None,
                duration_ms=duration_ms,
                ok=ok,
                error=r.text(error) if error else None,
                data=r.data(data),
            )
            with (self.dir / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(event.model_dump_json(exclude_none=True) + "\n")
            self._recent.append(event)
        return event

    def recent(self, n: int = 10) -> list[Event]:
        with self._lock:
            return list(self._recent)[-n:]

    def screenshot_path(self, label: str) -> Path:
        with self._lock:
            self._shots += 1
            number = self._shots
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:40]
        path = self.dir / "screenshots" / f"{number:03d}_{safe}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def subdir(self, name: str) -> Path:
        path = self.dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(
        self,
        name: str,
        payload: BaseModel | Mapping[str, Any],
        *,
        verbatim: Collection[str] = (),
    ) -> Path:
        """Strings in the payload are redacted as free text. Raw outputs go back to
        the caller in memory; what is persisted is the redacted copy.

        `verbatim` names top-level keys that are identifiers, not data (run id,
        artifact hash, capability id): the detectors would otherwise mangle them, a
        sha256 into `[SECRET]` and a numeric id into a masked account number. Only
        system-generated values belong there; never inputs, outputs or page text."""
        dumped = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
        redacted = self.redactor.data({k: v for k, v in dumped.items() if k not in verbatim})
        raw = {**redacted, **{k: v for k, v in dumped.items() if k in verbatim}}
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(raw, indent=2, sort_keys=True, default=str, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path


def _event_target(target: Target | None, redactor: Redactor) -> EventTarget | None:
    if target is None:
        return None
    fingerprint = target.fingerprint.model_dump(exclude_none=True)
    return EventTarget(
        description=redactor.text(target.description),
        fingerprint={k: redactor.text(str(v)) for k, v in fingerprint.items()},
    )
