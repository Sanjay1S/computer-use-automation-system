"""Evidence capture: masked screenshots, sanitized DOM, failure bundles.

Every capture is redacted in the page or before it is written. Playwright traces
are not used here: they record raw input values and network bodies.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from playwright.sync_api import Frame, Page
from playwright.sync_api import Locator as PwLocator

from understudy.evidence.run import Event
from understudy.redaction.redactor import Redactor

_FIELDS_JS = """
() => Array.from(document.querySelectorAll("input, textarea, select")).map((el) => ({
  type: el.tagName === "INPUT" ? (el.getAttribute("type") || "text").toLowerCase() : null,
  name: el.getAttribute("name"),
  autocomplete: el.getAttribute("autocomplete"),
  label: (el.labels && el.labels.length
    ? Array.from(el.labels).map((l) => l.innerText).join(" ")
    : el.getAttribute("aria-label")) || null,
}))
"""

# Values typed into fields live in the `value` property, which cloning drops; the
# attribute is removed too because servers render prefilled values into it.
_SANITIZED_DOM_JS = """
() => {
  const clone = document.documentElement.cloneNode(true);
  clone.querySelectorAll("input").forEach((i) => i.removeAttribute("value"));
  clone.querySelectorAll("textarea").forEach((t) => { t.textContent = ""; });
  clone.querySelectorAll("script").forEach((s) => { s.textContent = ""; });
  return clone.outerHTML;
}
"""


def sensitive_fields(frame: Page | Frame) -> list[PwLocator]:
    fields = cast(list[dict[str, Any]], frame.evaluate(_FIELDS_JS))
    every = frame.locator("input, textarea, select")
    return [
        every.nth(i)
        for i, field in enumerate(fields)
        if Redactor.classify_field(
            input_type=field["type"],
            label=field["label"],
            name_attr=field["name"],
            autocomplete=field["autocomplete"],
        )
        != "public"
    ]


def screenshot(page: Page, path: Path, extra_masks: Sequence[str] = ()) -> Path:
    """Password and PII fields are painted over in every frame, plus any selectors
    the app profile names (e.g. a panel that prints account numbers)."""
    masks: list[PwLocator] = []
    for frame in page.frames:
        try:
            masks += sensitive_fields(frame)
        except Exception:  # a frame that navigated away mid-capture has nothing to mask
            continue
    masks += [page.locator(f"css={selector}") for selector in extra_masks]
    path.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(path), mask=masks, animations="disabled")
    return path


def sanitized_dom(page: Page, redactor: Redactor) -> str:
    parts = []
    for index, frame in enumerate(page.frames):
        try:
            html = cast(str, frame.evaluate(_SANITIZED_DOM_JS))
        except Exception:
            continue
        label = "main frame" if index == 0 else f"frame {index}: {redactor.url(frame.url)}"
        parts.append(f"<!-- {label} -->\n{redactor.html(html)}")
    return "\n".join(parts) + "\n"


def write_failure_bundle(
    page: Page,
    directory: Path,
    redactor: Redactor,
    *,
    reason: str,
    snapshot: str,
    console: Sequence[str],
    events: Sequence[Event],
    extra_masks: Sequence[str] = (),
) -> Path:
    """Everything needed to diagnose a failed step without re-running it."""
    directory.mkdir(parents=True, exist_ok=True)
    try:
        screenshot(page, directory / "screenshot.png", extra_masks)
    except Exception as error:  # the bundle is still worth writing without the picture
        (directory / "screenshot.error.txt").write_text(redactor.text(str(error)) + "\n")
    (directory / "dom.html").write_text(sanitized_dom(page, redactor), encoding="utf-8")
    (directory / "aria.txt").write_text(redactor.snapshot(snapshot) + "\n", encoding="utf-8")
    (directory / "console.txt").write_text(
        "".join(redactor.text(line) + "\n" for line in console), encoding="utf-8"
    )
    (directory / "events.jsonl").write_text(
        "".join(e.model_dump_json(exclude_none=True) + "\n" for e in events), encoding="utf-8"
    )
    (directory / "bundle.json").write_text(
        json.dumps(
            {
                "reason": redactor.text(reason),
                "url": redactor.url(page.url),
                "title": redactor.text(page.title()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return directory
