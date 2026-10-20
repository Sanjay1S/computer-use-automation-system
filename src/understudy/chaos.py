"""Fault injection for replay scenarios (`--inject`), on sandbox tenants only.

Network faults are request routes, installed before the surface so the policy guard
still judges every request first, and armed only once the capability's first step
starts: the login before it may hit the same routes (ParaBank's overview page loads
the accounts list too). Page faults run when their step starts. Every fault fires
once.

  latency:<route>:<ms>       hold the first matching request for <ms>, then send it
  http_503_once:<route>      answer the first matching request with a 503
  session_expired_at:<step>  clear cookies and reload, as when an expired session
                             is noticed on the next page load
  drop_commit_response       hold the commit request and never answer it: the page
                             never learns what happened
  app_error_at_commit        answer the commit request with a 500
  rename_target:<step>       relabel the step's target, as a redesign would
  unknown_modal_at:<step>    cover the page with a modal no profile knows

<route> is a glob over the path relative to the tenant's base URL. <step> is a step
id or its prefix before the first underscore (`s4` for `s4_submit`). A separator
of `=` is accepted after the kind (`session_expired_at=s3`).
"""

from __future__ import annotations

import contextlib
import fnmatch
import re
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, Request, Route

from understudy.artifact.schema import Capability, Step
from understudy.policy.gate import PolicyGate
from understudy.surface import locators

MODAL_ID = "understudy-chaos-modal"

_ARGS: dict[str, tuple[str, ...]] = {
    "latency": ("route", "ms"),
    "http_503_once": ("route",),
    "session_expired_at": ("step",),
    "drop_commit_response": (),
    "app_error_at_commit": (),
    "rename_target": ("step",),
    "unknown_modal_at": ("step",),
}

_SPEC = re.compile(r"^(?P<kind>[a-z0-9_]+)(?:[:=](?P<rest>.*))?$")

_MODAL_JS = """
(id) => {
  const d = document.createElement("div");
  d.id = id;
  d.setAttribute("role", "dialog");
  d.setAttribute("aria-modal", "true");
  d.setAttribute("aria-label", "Maintenance notice");
  d.style.cssText = "position:fixed;inset:0;z-index:2147483647;display:flex;"
    + "align-items:center;justify-content:center;background:rgba(0,0,0,.45)";
  d.innerHTML = '<div style="background:#fff;padding:24px;font:16px sans-serif">'
    + "<p>Scheduled maintenance tonight from 22:00.</p>"
    + '<button type="button">OK</button></div>';
  d.querySelector("button").addEventListener("click", () => d.remove());
  document.body.appendChild(d);
}
"""

_RENAME_JS = """
(e) => {
  const label = "Continue";
  if (e.tagName === "INPUT") { e.setAttribute("value", label); e.value = label; }
  else if (e.tagName !== "SELECT") { e.textContent = label; }
  if (e.id) { e.id = e.id + "-renamed"; }
  const name = e.getAttribute("name");
  if (name) { e.setAttribute("name", name + "-renamed"); }
}
"""


@dataclass(frozen=True)
class Fault:
    kind: str
    route: str | None = None
    ms: int | None = None
    step: str | None = None

    def __str__(self) -> str:
        args = [a for a in (self.route, self.ms, self.step) if a is not None]
        return ":".join([self.kind, *map(str, args)])


def parse_fault(spec: str, capability: Capability) -> Fault:
    match = _SPEC.match(spec.strip())
    if match is None or match.group("kind") not in _ARGS:
        raise ValueError(f"unknown fault {spec!r}; expected one of: {', '.join(_ARGS)}")
    kind, rest = match.group("kind"), match.group("rest")
    wanted = _ARGS[kind]
    if not wanted:
        if rest:
            raise ValueError(f"{kind} takes no arguments")
        return Fault(kind)
    if not rest:
        raise ValueError(f"{kind} needs: {':'.join(f'<{a}>' for a in wanted)}")
    if wanted == ("step",):
        return Fault(kind, step=_step_id(rest, capability))
    if wanted == ("route",):
        return Fault(kind, route=_route(rest))
    route, _, ms = rest.rpartition(":")
    if not route or not ms.isdigit() or not 0 < int(ms) <= 60_000:
        raise ValueError(f"{kind} needs <route>:<ms> with 1-60000 ms")
    return Fault(kind, route=_route(route), ms=int(ms))


class Chaos:
    def __init__(self, faults: Sequence[Fault], gate: PolicyGate) -> None:
        self.faults = list(faults)
        self.gate = gate
        self.fired: list[str] = []
        self._spent: set[int] = set()
        self._held: list[Route] = []
        self._armed = False

    def install(self, page: Page) -> None:
        """Before the WebSurface is built: Playwright runs the page routes registered
        last first, so the surface's policy guard then runs ahead of these."""
        if any(f.step is None for f in self.faults):
            page.route("**/*", self._route)

    def before_step(self, step: Step, page: Page, params: Mapping[str, str]) -> list[str]:
        self._armed = True
        fired: list[str] = []
        for index, fault in enumerate(self.faults):
            if fault.step != step.id or index in self._spent:
                continue
            self._spent.add(index)
            note = self._apply(fault, step, page, params)
            fired.append(f"{fault}{f' ({note})' if note else ''}")
        self.fired += fired
        return fired

    @contextlib.contextmanager
    def paused(self) -> Iterator[None]:
        """Network faults stand down while the session provider logs in again."""
        armed, self._armed = self._armed, False
        try:
            yield
        finally:
            self._armed = armed

    def release(self) -> None:
        """Abort held requests, so closing the page does not leave them dangling."""
        for route in self._held:
            with contextlib.suppress(PlaywrightError):
                route.abort("connectionaborted")
        self._held.clear()

    # --- internals -------------------------------------------------------------------

    def _apply(self, fault: Fault, step: Step, page: Page, params: Mapping[str, str]) -> str:
        if fault.kind == "session_expired_at":
            page.context.clear_cookies()
            page.reload(wait_until="load")
            return ""
        if fault.kind == "unknown_modal_at":
            page.evaluate(_MODAL_JS, MODAL_ID)
            return ""
        if step.target is None:
            return "the step has no target to rename"
        resolution = locators.resolve(page, step.target, params)
        if resolution.element is None:
            return "the target was not on the page yet; nothing renamed"
        resolution.element.evaluate(_RENAME_JS)
        return ""

    def _route(self, route: Route, request: Request) -> None:
        if not self._armed:
            route.fallback()
            return
        url = request.url
        for index, fault in enumerate(self.faults):
            if index in self._spent or not self._matches(fault, url):
                continue
            self._spent.add(index)
            self.fired.append(str(fault))
            if fault.kind == "latency":
                # Blocks this client while it waits; the browser keeps running.
                time.sleep((fault.ms or 0) / 1000)
                break
            if fault.kind == "http_503_once":
                route.fulfill(
                    status=503, content_type="text/html", body="<h1>Service Unavailable</h1>"
                )
                return
            if fault.kind == "app_error_at_commit":
                route.fulfill(status=500, body="")
                return
            self._held.append(route)  # drop_commit_response
            return
        route.fallback()

    def _matches(self, fault: Fault, url: str) -> bool:
        if fault.kind in ("drop_commit_response", "app_error_at_commit"):
            return self.gate.is_commit_request(url)
        if fault.route is None:
            return False
        route = self.gate.route_of(url)
        return route is not None and fnmatch.fnmatchcase(route.lower(), fault.route.lower())


def _step_id(ref: str, capability: Capability) -> str:
    matches = [s.id for s in capability.steps if ref in (s.id, s.id.split("_", 1)[0])]
    if len(matches) != 1:
        found = "no step" if not matches else f"steps {', '.join(matches)}"
        raise ValueError(f"step {ref!r} matches {found} in {capability.id}")
    return matches[0]


def _route(text: str) -> str:
    if not text.startswith("/"):
        raise ValueError(f"route {text!r} must start with / (relative to the base URL)")
    return text
