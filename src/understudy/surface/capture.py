"""Evidence capture: masked screenshots, sanitized DOM, failure bundles.

Every capture is redacted in the page or before it is written. Playwright traces
are not used here: they record raw input values and network bodies.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping, Sequence
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
# Masked panels (names, account lists) are emptied in the clone, never in the page:
# the live page may be the one an operator takes over.
_SANITIZED_DOM_JS = """
(masks) => {
  const clone = document.documentElement.cloneNode(true);
  clone.querySelectorAll("input").forEach((i) => i.removeAttribute("value"));
  clone.querySelectorAll("textarea").forEach((t) => { t.textContent = ""; });
  clone.querySelectorAll("script").forEach((s) => { s.textContent = ""; });
  masks.forEach((m) => {
    try { clone.querySelectorAll(m).forEach((e) => { e.textContent = "[masked]"; }); }
    catch (error) { /* a selector this frame cannot parse masks nothing here */ }
  });
  return clone.outerHTML;
}
"""

_MASKED_TEXT_JS = """
(masks) => masks.flatMap((m) => {
  try { return Array.from(document.querySelectorAll(m)); } catch (error) { return []; }
}).flatMap((e) => (e.innerText || e.textContent || "").split(/[\\n\\t]+/))
  .map((t) => t.replace(/\\s+/g, " ").trim()).filter((t) => t.length >= 3)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(path), mask=_masks(page, extra_masks), animations="disabled")
    return path


def screenshot_png(
    page: Page, *, clip: Mapping[str, float] | None = None, extra_masks: Sequence[str] = ()
) -> bytes:
    """A screenshot in memory, for showing to the model during discovery. Credential
    and PII fields are always masked. The profile's panel masks (`extra_masks`) are
    for filed evidence: the model reads those panels in the accessibility snapshot
    anyway, and needs them to act (a list of accounts to choose from)."""
    return page.screenshot(
        mask=_masks(page, extra_masks),
        animations="disabled",
        clip=cast(Any, dict(clip)) if clip else None,
    )


def _masks(page: Page, extra_masks: Sequence[str]) -> list[PwLocator]:
    masks: list[PwLocator] = []
    for frame in page.frames:
        try:
            masks += sensitive_fields(frame)
        except Exception:  # a frame that navigated away mid-capture has nothing to mask
            continue
    return masks + [page.locator(f"css={selector}") for selector in extra_masks]


def masked_texts(page: Page, masks: Sequence[str]) -> set[str]:
    """The visible text of the masked panels, line by line, for hiding it in captures
    that are not HTML (the accessibility snapshot, dialog text)."""
    texts: set[str] = set()
    for frame in page.frames:
        try:
            texts.update(cast(list[str], frame.evaluate(_MASKED_TEXT_JS, list(masks))))
        except Exception:
            continue
    return texts


def mask_snapshot(snapshot: str, texts: Collection[str]) -> str:
    """Hide the masked panels' text in an accessibility snapshot, where it appears as a
    line's whole text or a whole quoted name. Whole values only: a panel's "Account"
    header must not turn every "Account" elsewhere into a mask."""
    for text in sorted(texts, key=len, reverse=True):
        escaped = re.escape(text)
        snapshot = re.sub(rf"(: ){escaped}$", r"\1[masked]", snapshot, flags=re.MULTILINE)
        snapshot = snapshot.replace(f'"{text}"', '"[masked]"')
    return snapshot


def sanitized_dom(page: Page, redactor: Redactor, masks: Sequence[str] = ()) -> str:
    parts = []
    for index, frame in enumerate(page.frames):
        try:
            html = cast(str, frame.evaluate(_SANITIZED_DOM_JS, list(masks)))
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
    dialog: str | None = None,
) -> Path:
    """Everything needed to diagnose a failed step without re-running it. The masked
    panels are blanked in the DOM and their text is hidden everywhere else.
    `dialog` lists the native dialogs the page showed (already answered)."""
    directory.mkdir(parents=True, exist_ok=True)
    masked = masked_texts(page, extra_masks) if extra_masks else set()
    if dialog is not None:
        for text in sorted(masked, key=len, reverse=True):
            dialog = dialog.replace(text, "[masked]")
        (directory / "dialog.txt").write_text(redactor.text(dialog) + "\n", encoding="utf-8")
    try:
        screenshot(page, directory / "screenshot.png", extra_masks)
    except Exception as error:  # the bundle is still worth writing without the picture
        (directory / "screenshot.error.txt").write_text(redactor.text(str(error)) + "\n")
    (directory / "dom.html").write_text(
        sanitized_dom(page, redactor, extra_masks), encoding="utf-8"
    )
    (directory / "aria.txt").write_text(
        redactor.snapshot(mask_snapshot(snapshot, masked)) + "\n", encoding="utf-8"
    )
    _write_common(directory, redactor, page.url, page.title(), reason, console, events)
    return directory


def _write_common(
    directory: Path,
    redactor: Redactor,
    url: str,
    title: str,
    reason: str,
    console: Sequence[str],
    events: Sequence[Event],
) -> None:
    (directory / "console.txt").write_text(
        "".join(redactor.text(line) + "\n" for line in console), encoding="utf-8"
    )
    (directory / "events.jsonl").write_text(
        "".join(e.model_dump_json(exclude_none=True) + "\n" for e in events), encoding="utf-8"
    )
    summary = {
        "reason": redactor.text(reason),
        "url": redactor.url(url),
        "title": redactor.text(title),
    }
    (directory / "bundle.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
