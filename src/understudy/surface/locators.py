"""Locator synthesis (discovery, human recorder) and resolution (replay).

Synthesis collects every way the page offers to find an element, keeps only the
candidates that resolve to exactly that element, and writes down why each one is
as stable as it is. Resolution walks the chain in order and accepts the first
candidate that finds one visible element whose fingerprint agrees, so a look-alike
in the right place is refused rather than clicked.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import FrameLocator, Page
from playwright.sync_api import Locator as PwLocator

from understudy.artifact.schema import Fingerprint, Locator, Stability, Target, Verification
from understudy.surface.base import (
    Ancestor,
    AttemptOutcome,
    ElementFacts,
    LocatorAttempt,
    NearText,
    Resolution,
    TableContext,
)
from understudy.surface.snapshot import parse_line

Root = Page | FrameLocator | PwLocator

_PARAM = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
UNSAFE_IN_SELECTOR = re.compile(r"[\"'\\\[\]\n]")
_GENERATED_ID = re.compile(
    r"\d{3,}|[0-9a-f]{8,}|^(?:ember|ext-gen|j_id|jsf|react|ng-|mui-|radix-)|[:.]|__",
    re.IGNORECASE,
)
_SELECTOR_KEYS = {"css": None, "xpath": None, "table_cell": ("table", "inner")}
_ROWS = ":scope > thead > tr, :scope > tbody > tr, :scope > tfoot > tr, :scope > tr"
_TEXT_LIMIT = 60

_FACTS_JS = """
(el) => {
  const ROWS = "ROWS_SELECTOR";
  const norm = (s) => (s || "").replace(/\\s+/g, " ").trim();
  const attr = (n) => { const v = el.getAttribute(n); return v === null || v === "" ? null : v; };
  const tag = el.tagName.toLowerCase();
  let inputType = null;
  if (tag === "input") inputType = (el.getAttribute("type") || "text").toLowerCase();
  if (tag === "button") inputType = (el.getAttribute("type") || "submit").toLowerCase();

  let label = null;
  if (el.labels && el.labels.length) {
    label = norm(Array.from(el.labels).map((l) => l.innerText).join(" "));
  }
  const labelledBy = el.getAttribute("aria-labelledby");
  if (!label && labelledBy) {
    label = norm(labelledBy.split(/\\s+/).map((id) => {
      const n = document.getElementById(id); return n ? n.innerText : "";
    }).join(" "));
  }
  if (!label && attr("aria-label")) label = norm(attr("aria-label"));

  const form = el.form || el.closest("form");
  const formButtons = form ? Array.from(form.querySelectorAll(
    "button, input[type=submit], input[type=button], input[type=image]"
  )).map((b) => norm(
    b.tagName === "INPUT" ? b.value : (b.getAttribute("aria-label") || b.innerText)
  ))
    .filter(Boolean) : [];

  const isControl = (n) => ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(n.tagName)
    || !!n.querySelector("input, select, textarea, button");
  const anchorFor = (node) => {
    for (let s = node.previousElementSibling; s; s = s.previousElementSibling) {
      if (isControl(s)) return null;
      if (["BR", "HR", "SCRIPT", "STYLE"].includes(s.tagName) || !norm(s.innerText)) continue;
      return s;
    }
    return null;
  };
  let near = null;
  if (["input", "select", "textarea"].includes(tag)) {
    let anchor = anchorFor(el), hops = 0, parent = null;
    if (!anchor && el.parentElement && el.parentElement !== document.body) {
      parent = el.parentElement; anchor = anchorFor(parent); hops = 1;
    }
    if (anchor && norm(anchor.innerText).length <= 200) {
      near = { anchor_tag: anchor.tagName.toLowerCase(), text: norm(anchor.innerText), hops,
               parent_tag: parent ? parent.tagName.toLowerCase() : null };
    }
  }

  let table = null;
  const cell = el.closest("td, th");
  const tbl = cell && cell.closest("table");
  if (cell && tbl) {
    const rows = Array.from(tbl.querySelectorAll(ROWS));
    const header = rows.find((r) => r.querySelector(":scope > th")) || null;
    const row = cell.parentElement;
    if (header && row !== header) {
      const heads = Array.from(header.cells).map((c) => norm(c.innerText));
      const data = rows.filter((r) => r !== header);
      let keyCol = -1;
      for (let i = 0; i < heads.length && keyCol < 0; i++) {
        const t = row.cells[i] ? norm(row.cells[i].innerText) : "";
        if (!t || !heads[i]) continue;
        const same = data.filter((r) => r.cells[i] && norm(r.cells[i].innerText) === t);
        if (same.length === 1) keyCol = i;
      }
      if (keyCol >= 0) {
        table = { id: tbl.id || null, classes: Array.from(tbl.classList),
                  column: heads[cell.cellIndex] || cell.cellIndex,
                  key_column: heads[keyCol], row_key: norm(row.cells[keyCol].innerText) };
      }
    }
  }

  const nthOf = (n) => Array.from(n.parentElement ? n.parentElement.children : [n])
    .filter((c) => c.tagName === n.tagName).indexOf(n) + 1;
  const ancestry = [];
  let n = el.parentElement;
  for (; n && n !== document.documentElement && ancestry.length < 8; n = n.parentElement) {
    ancestry.push({ tag: n.tagName.toLowerCase(), nth: nthOf(n), id: n.id || null });
  }

  const isField = ["input", "select", "textarea"].includes(tag);
  const text = isField ? null : norm(el.innerText).slice(0, 200);
  return {
    tag, input_type: inputType, label: label || null,
    test_id: attr("data-testid"), id_attr: attr("id"), name_attr: attr("name"),
    placeholder: attr("placeholder"), value_attr: tag === "input" ? attr("value") : null,
    text: text || null, href: tag === "a" && el.getAttribute("href") !== null ? el.href : null,
    autocomplete: attr("autocomplete"),
    form_action: form && form.getAttribute("action") !== null ? form.action : null,
    form_buttons: formButtons, near, table, nth_of_type: nthOf(el), ancestry,
  };
}
"""

_TABLE_JS = """
(tbl, [column, keyColumn, rowKey]) => {
  const ROWS = "ROWS_SELECTOR";
  const norm = (s) => (s || "").replace(/\\s+/g, " ").trim();
  const rows = Array.from(tbl.querySelectorAll(ROWS));
  const header = rows.find((r) => r.querySelector(":scope > th"));
  if (!header) return { error: "the table has no header row" };
  const heads = Array.from(header.cells).map((c) => norm(c.innerText));
  const col = typeof column === "number" ? column : heads.indexOf(column);
  const key = heads.indexOf(keyColumn);
  if (col < 0 || col >= heads.length) return { error: `no column ${JSON.stringify(column)}` };
  if (key < 0) return { error: `no column headed ${JSON.stringify(keyColumn)}` };
  const matches = [];
  rows.forEach((r, i) => {
    if (r !== header && r.cells[key] && norm(r.cells[key].innerText) === rowKey) matches.push(i);
  });
  return { rows: matches, col };
}
"""

_FACTS_JS = _FACTS_JS.replace("ROWS_SELECTOR", _ROWS)
_TABLE_JS = _TABLE_JS.replace("ROWS_SELECTOR", _ROWS)

_FRAME_SELECTOR_JS = """
(f) => {
  const tag = f.tagName.toLowerCase();
  const name = f.getAttribute("name"), id = f.id;
  if (name) return `${tag}[name=${JSON.stringify(name)}]`;
  if (id) return `${tag}#${CSS.escape(id)}`;
  const same = Array.from(f.parentElement.children).filter((c) => c.tagName === f.tagName);
  return `${tag}:nth-of-type(${same.indexOf(f) + 1})`;
}
"""


# --- facts -----------------------------------------------------------------------


def read_facts(element: PwLocator, *, timeout_ms: int = 2_000) -> ElementFacts:
    """Everything the surface can establish about one element.

    Role and name come from an element-level accessibility snapshot, which is the
    computation `get_by_role` matches against. Taking one replaces Playwright's
    page-level snapshot, after which refs (including a ref locator passed in here)
    stop resolving; so it is taken last, and `WebSurface` retakes the page-level
    snapshot straight after.
    """
    frames = frame_path(element)
    raw = cast(dict[str, Any], element.evaluate(_FACTS_JS, timeout=timeout_ms))
    first_line = element.aria_snapshot(timeout=timeout_ms).split("\n", 1)[0]
    role, name = parse_line(first_line) if first_line else (None, None)
    table = raw.get("table")
    near = raw.get("near")
    return ElementFacts(
        tag=raw["tag"],
        role=role,
        name=name,
        input_type=raw.get("input_type"),
        label=raw.get("label"),
        test_id=raw.get("test_id"),
        id_attr=raw.get("id_attr"),
        name_attr=raw.get("name_attr"),
        placeholder=raw.get("placeholder"),
        value_attr=raw.get("value_attr"),
        text=raw.get("text"),
        href=raw.get("href"),
        autocomplete=raw.get("autocomplete"),
        form_action=raw.get("form_action"),
        form_buttons=tuple(raw.get("form_buttons") or ()),
        near=NearText(**near) if near else None,
        table=_table_context(table) if table else None,
        nth_of_type=int(raw.get("nth_of_type") or 1),
        ancestry=tuple(Ancestor(a["tag"], a["nth"], a["id"]) for a in raw.get("ancestry") or ()),
        frame_path=frames,
    )


def frame_path(element: PwLocator) -> tuple[str, ...]:
    handle = element.element_handle(timeout=2_000)
    frame = handle.owner_frame()
    path: list[str] = []
    while frame is not None and frame.parent_frame is not None:
        path.append(cast(str, frame.frame_element().evaluate(_FRAME_SELECTOR_JS)))
        frame = frame.parent_frame
    return tuple(reversed(path))


def _table_context(raw: Mapping[str, Any]) -> TableContext:
    if raw.get("id") and _is_ident(raw["id"]):
        selector = f"table#{raw['id']}"
    elif raw.get("classes"):
        selector = "table" + "".join(f".{c}" for c in raw["classes"] if _is_ident(c))
    else:
        selector = "table"
    return TableContext(selector, raw["column"], raw["key_column"], raw["row_key"])


# --- building Playwright locators ------------------------------------------------


def root_for(page: Page, frames: Sequence[str]) -> Page | FrameLocator:
    root: Page | FrameLocator = page
    for selector in frames:
        root = root.frame_locator(selector)
    return root


def build(root: Root, locator: Locator, params: Mapping[str, str]) -> tuple[PwLocator | None, str]:
    """A Playwright locator for one artifact locator, or None with the reason when
    the locator cannot match anything (only table_cell can tell that in advance)."""
    value = substitute(locator.value, params, locator.by) if locator.templated else locator.value
    if locator.by == "role":
        spec = _as_dict(value)
        return (
            root.get_by_role(
                cast(Any, spec["role"]), name=spec.get("name"), exact=spec.get("exact", True)
            ),
            "",
        )
    if locator.by == "table_cell":
        return _table_cell(root, _as_dict(value))
    text = _as_str(value)
    if locator.by == "test_id":
        return root.get_by_test_id(text), ""
    if locator.by == "label":
        return root.get_by_label(text, exact=True), ""
    if locator.by == "placeholder":
        return root.get_by_placeholder(text, exact=True), ""
    if locator.by == "text":
        return root.get_by_text(text, exact=True), ""
    if locator.by == "css":
        return root.locator(f"css={text}"), ""
    return root.locator(f"xpath={text}"), ""


def substitute(value: str | dict[str, Any], params: Mapping[str, str], by: str) -> Any:
    """Fill `{param}` placeholders. A value that would break out of a CSS or XPath
    string is refused rather than escaped: the selector is not ours to rewrite."""
    selector_keys = _SELECTOR_KEYS.get(by, ())

    def fill(text: str, is_selector: bool) -> str:
        def one(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in params:
                raise ValueError(f"param {name!r} is not bound")
            if is_selector and UNSAFE_IN_SELECTOR.search(params[name]):
                raise ValueError(f"param {name!r} has characters that cannot go in a {by} selector")
            return params[name]

        return _PARAM.sub(one, text)

    if isinstance(value, str):
        return fill(value, selector_keys is None)
    return {
        k: fill(v, selector_keys is not None and k in selector_keys) if isinstance(v, str) else v
        for k, v in value.items()
    }


def _table_cell(root: Root, spec: Mapping[str, Any]) -> tuple[PwLocator | None, str]:
    table = root.locator(f"css={spec['table']}")
    count = table.count()
    if count != 1:
        return None, f"table {spec['table']!r} matched {count} elements"
    found = cast(
        dict[str, Any],
        table.evaluate(_TABLE_JS, [spec["column"], spec["key_column"], spec["row_key"]]),
    )
    if "error" in found:
        return None, str(found["error"])
    rows = cast(list[int], found["rows"])
    if not rows:
        return None, f"no row has {spec['key_column']!r} = the row key"
    cells = [
        table.locator(_ROWS).nth(i).locator(":scope > td, :scope > th").nth(found["col"])
        for i in rows
    ]
    cell = cells[0]
    for other in cells[1:]:
        cell = cell.or_(other)
    if spec.get("inner"):
        cell = cell.locator(f"css={spec['inner']}")
    return cell, ""


# --- fingerprints ----------------------------------------------------------------


def fingerprint_mismatches(
    fingerprint: Fingerprint, facts: ElementFacts, params: Mapping[str, str]
) -> list[str]:
    problems = []
    for field in ("role", "name", "tag", "input_type", "label"):
        expected = getattr(fingerprint, field)
        if expected is None:
            continue
        expected = _PARAM.sub(lambda m: params.get(m.group(1), m.group(0)), expected)
        actual = getattr(facts, field)
        if _norm(expected) != _norm(actual or ""):
            problems.append(f"{field}: expected {expected!r}, found {actual!r}")
    return problems


# --- resolution ------------------------------------------------------------------


def resolve(
    page: Page,
    target: Target,
    params: Mapping[str, str],
    *,
    require_visible: bool = True,
) -> Resolution[PwLocator]:
    """One non-waiting pass over the fallback chain. Waiting is the caller's job."""
    attempts: list[LocatorAttempt] = []
    root: Root = root_for(page, target.frame_path)
    if target.scope is not None:
        scope, detail = _try_build(root, target.scope, params)
        count = scope.count() if scope is not None else 0
        if scope is None or count != 1:
            attempts.append(
                LocatorAttempt(
                    -1, f"scope:{target.scope.by}", _miss(detail, count), count, detail or "scope"
                )
            )
            return Resolution(attempts)
        root = scope

    for index, locator in enumerate(target.locators):
        built, detail = _try_build(root, locator, params)
        if built is None:
            attempts.append(LocatorAttempt(index, locator.by, _miss(detail, 0), 0, detail))
            continue
        try:
            count = built.count()
            if count != 1:
                attempts.append(LocatorAttempt(index, locator.by, _miss("", count), count))
                continue
            if require_visible and not built.is_visible():
                attempts.append(LocatorAttempt(index, locator.by, "hidden", 1))
                continue
            facts = read_facts(built)
        except PlaywrightError as error:
            attempts.append(
                LocatorAttempt(index, locator.by, "error", 0, f"error: {_first_line(error)}")
            )
            continue
        problems = fingerprint_mismatches(target.fingerprint, facts, params)
        if problems:
            attempts.append(
                LocatorAttempt(index, locator.by, "fingerprint_mismatch", 1, "; ".join(problems))
            )
            continue
        attempts.append(LocatorAttempt(index, locator.by, "matched", 1))
        return Resolution(attempts, built, facts, index)
    return Resolution(attempts)


def _miss(detail: str, count: int) -> AttemptOutcome:
    if detail.startswith("error"):
        return "error"
    return "ambiguous" if count > 1 else "no_match"


def _try_build(
    root: Root, locator: Locator, params: Mapping[str, str]
) -> tuple[PwLocator | None, str]:
    try:
        return build(root, locator, params)
    except (ValueError, KeyError, TypeError, PlaywrightError) as error:
        return None, f"error: {_first_line(error)}"


# --- synthesis -------------------------------------------------------------------


@dataclass(frozen=True)
class Synthesis:
    facts: ElementFacts
    target: Target | None
    """None when no candidate was verified: the step cannot be replayed as recorded."""
    rejected: list[tuple[str, str]]
    """(locator kind, why it was dropped), for the review summary."""


def synthesize(
    page: Page,
    element: PwLocator,
    *,
    run_id: str,
    params: Mapping[str, str] | None = None,
    description: str | None = None,
    facts: ElementFacts | None = None,
) -> Synthesis:
    params = params or {}
    # The handle first: reading facts can leave a ref locator unresolvable.
    handle = element.element_handle(timeout=2_000)
    facts = facts or read_facts(element)
    frame_root = root_for(page, facts.frame_path)
    candidates = candidates_for(facts)

    verified, rejected = _verify(frame_root, candidates, handle, run_id)
    scope: Locator | None = None
    if not any(c.stability != "low" for c in verified):
        scoped = _scoped_attempt(frame_root, facts, candidates, handle, run_id)
        if scoped is not None:
            scope, verified, rejected = scoped

    if not verified:
        return Synthesis(facts, None, rejected)
    templated, notes = _template_all(verified, params)
    fingerprint = _template_fingerprint(facts.fingerprint(), params)
    target = Target(
        description=description or _describe(facts),
        fingerprint=fingerprint,
        locators=templated,
        frame_path=list(facts.frame_path),
        scope=scope,
    )
    return Synthesis(facts, target, rejected + notes)


def candidates_for(facts: ElementFacts) -> list[Locator]:
    """Every locator the element offers, in the order replay should try them."""
    out: list[Locator] = []
    tag = facts.tag

    if facts.test_id:
        out.append(
            _loc(
                "test_id",
                facts.test_id,
                "high",
                "A test id exists for automation; it survives restyling and copy changes.",
            )
        )
    if facts.role and facts.name:
        data_like = bool(re.search(r"\d{3,}", facts.name))
        out.append(
            _loc(
                "role",
                {"role": facts.role, "name": facts.name, "exact": True},
                "medium" if data_like else "high",
                "Role and accessible name, as a user perceives the element. It survives markup "
                "changes and moves with the label text."
                + (
                    " The name looks like data, so it is only as stable as that value."
                    if data_like
                    else ""
                ),
            )
        )
    if facts.label:
        out.append(
            _loc(
                "label",
                facts.label,
                "high",
                "The field's associated label; survives layout changes.",
            )
        )
    if facts.placeholder:
        out.append(
            _loc(
                "placeholder",
                facts.placeholder,
                "medium",
                "Placeholder text, which tends to be reworded.",
            )
        )
    if facts.id_attr and not _GENERATED_ID.search(facts.id_attr) and _is_ident(facts.id_attr):
        out.append(
            _loc(
                "css",
                f"{tag}#{facts.id_attr}",
                "medium",
                "A hand-written id (it does not look generated). It survives restyling, "
                "not a rewrite of the page.",
            )
        )
    if facts.name_attr and not UNSAFE_IN_SELECTOR.search(facts.name_attr):
        out.append(
            _loc(
                "css",
                f"{tag}[name='{facts.name_attr}']",
                "high",
                "The form field's name is what the server reads, so it changes only "
                "with the backend.",
            )
        )
    if (
        tag == "input"
        and facts.input_type in ("button", "submit")
        and facts.value_attr
        and not UNSAFE_IN_SELECTOR.search(facts.value_attr)
    ):
        out.append(
            _loc(
                "css",
                f"input[type='{facts.input_type}'][value='{facts.value_attr}']",
                "medium",
                "The value attribute is the button's label; it moves with it.",
            )
        )
    if facts.near and facts.is_form_control:
        out.append(_near_locator(facts, facts.near))
    if facts.text and not facts.is_form_control and len(facts.text) <= _TEXT_LIMIT:
        data_like = bool(re.search(r"\d{3,}", facts.text))
        out.append(
            _loc(
                "text",
                facts.text,
                "low" if data_like else "medium",
                "Exact visible text; moves with copy changes.",
            )
        )
    if facts.table:
        t = facts.table
        by_index = isinstance(t.column, int)
        out.append(
            _loc(
                "table_cell",
                {
                    "table": t.table,
                    "column": t.column,
                    "key_column": t.key_column,
                    "row_key": t.row_key,
                    **({"inner": tag} if tag not in ("td", "th") else {}),
                },
                "low" if by_index else "medium",
                "Found by the row's key cell, so it follows the row when rows are added or "
                "reordered; the column is "
                + ("its position, because its header is blank." if by_index else "its header."),
            )
        )
    structural = _structural_css(facts)
    if structural:
        out.append(
            _loc(
                "css",
                structural,
                "low",
                "Position in the DOM below the nearest stable id. Any layout change "
                "breaks it; last resort.",
            )
        )
    return out


def _verify(
    root: Root, candidates: list[Locator], handle: Any, run_id: str
) -> tuple[list[Locator], list[tuple[str, str]]]:
    verified: list[Locator] = []
    rejected: list[tuple[str, str]] = []
    for candidate in candidates:
        built, detail = _try_build(root, candidate, {})
        if built is None:
            rejected.append((candidate.by, detail or "matched nothing"))
            continue
        try:
            count = built.count()
            same = count == 1 and bool(built.evaluate("(e, t) => e === t", handle))
        except PlaywrightError as error:
            rejected.append((candidate.by, f"error: {_first_line(error)}"))
            continue
        if count != 1:
            rejected.append((candidate.by, f"matched {count} elements, not 1"))
        elif not same:
            rejected.append((candidate.by, "matched a different element"))
        else:
            verified.append(
                candidate.model_copy(
                    update={"verified": Verification(unique=True, same_element=True, run_id=run_id)}
                )
            )
    return verified, rejected


def _scoped_attempt(
    root: Root, facts: ElementFacts, candidates: list[Locator], handle: Any, run_id: str
) -> tuple[Locator, list[Locator], list[tuple[str, str]]] | None:
    """When nothing better than a structural path is unique on the page, try again
    inside the nearest ancestor with a stable id ("role+name, scoped")."""
    anchor = next(
        (a for a in facts.ancestry if a.id and not _GENERATED_ID.search(a.id) and _is_ident(a.id)),
        None,
    )
    if anchor is None:
        return None
    scope = _loc(
        "css",
        f"{anchor.tag}#{anchor.id}",
        "medium",
        "The nearest ancestor with a hand-written id; the locators below are "
        "unique only inside it.",
    )
    built, _ = _try_build(root, scope, {})
    if built is None or built.count() != 1:
        return None
    verified, rejected = _verify(
        built, [c for c in candidates if c.stability != "low"], handle, run_id
    )
    if not verified:
        return None
    scope = scope.model_copy(
        update={"verified": Verification(unique=True, same_element=True, run_id=run_id)}
    )
    return scope, verified, rejected


def _template_all(
    locators: list[Locator], params: Mapping[str, str]
) -> tuple[list[Locator], list[tuple[str, str]]]:
    notes: list[tuple[str, str]] = []
    values: dict[str, list[str]] = {}
    for name, value in params.items():
        if len(value) >= 3:
            values.setdefault(value, []).append(name)
    ambiguous = {v for v, names in values.items() if len(names) > 1}
    for value in ambiguous:
        notes.append(
            (
                "template",
                f"params {', '.join(values[value])} share a value; "
                "locators containing it were left literal",
            )
        )
    usable = {v: names[0] for v, names in values.items() if v not in ambiguous}

    out = []
    for locator in locators:
        new_value, changed = _template_value(locator.value, usable)
        out.append(
            locator.model_copy(update={"value": new_value, "templated": changed})
            if changed
            else locator
        )
    return out, notes


def _template_value(
    value: str | dict[str, Any], usable: Mapping[str, str]
) -> tuple[str | dict[str, Any], bool]:
    def one(text: str) -> tuple[str, bool]:
        changed = False
        for literal, name in usable.items():
            pattern = re.compile(rf"(?<![\w]){re.escape(literal)}(?![\w])")
            if pattern.search(text):
                text = pattern.sub("{" + name + "}", text)
                changed = True
        return text, changed

    if isinstance(value, str):
        return one(value)
    any_changed = False
    result: dict[str, Any] = {}
    for key, part in value.items():
        if isinstance(part, str):
            part, changed = one(part)
            any_changed = any_changed or changed
        result[key] = part
    return result, any_changed


def _template_fingerprint(fingerprint: Fingerprint, params: Mapping[str, str]) -> Fingerprint:
    usable = {v: k for k, v in params.items() if len(v) >= 3}
    if len(usable) != len([v for v in params.values() if len(v) >= 3]):
        return fingerprint
    updates = {}
    for field in ("name", "label"):
        current = getattr(fingerprint, field)
        if current is not None:
            new, changed = _template_value(current, usable)
            if changed:
                updates[field] = new
    return fingerprint.model_copy(update=updates) if updates else fingerprint


def _near_locator(facts: ElementFacts, near: NearText) -> Locator:
    if len(near.text) <= _TEXT_LIMIT:
        test, stability = f"normalize-space(.)={_xpath_literal(near.text)}", "medium"
    else:
        prefix = near.text[:_TEXT_LIMIT].rsplit(" ", 1)[0]
        test, stability = f"starts-with(normalize-space(.), {_xpath_literal(prefix)})", "low"
    anchor = f"//{near.anchor_tag}[{test}]"
    if near.hops == 0:
        path = f"{anchor}/following-sibling::{facts.tag}[1]"
    else:
        path = f"{anchor}/following-sibling::{near.parent_tag}[1]//{facts.tag}"
    return _loc(
        "xpath",
        path,
        cast(Stability, stability),
        "Anchored on the text placed beside the control, because the page gives it "
        "no <label> or accessible name. It moves if that text is reworded.",
    )


def _structural_css(facts: ElementFacts) -> str | None:
    parts = [f"{facts.tag}:nth-of-type({facts.nth_of_type})"]
    for ancestor in facts.ancestry:
        if ancestor.id and not _GENERATED_ID.search(ancestor.id) and _is_ident(ancestor.id):
            return " > ".join([f"#{ancestor.id}", *reversed(parts)])
        if ancestor.tag == "body":
            return " > ".join(["body", *reversed(parts)])
        parts.append(f"{ancestor.tag}:nth-of-type({ancestor.nth_of_type})")
    return None


def _describe(facts: ElementFacts) -> str:
    what = facts.role or facts.tag
    named = facts.name or facts.label or (facts.near.text if facts.near else None)
    return f"{what} {named!r}" if named else what


def _loc(by: str, value: str | dict[str, Any], stability: Stability, note: str) -> Locator:
    return Locator(by=cast(Any, by), value=value, stability=stability, note=note)


def _as_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("this locator kind takes an object value")
    return value


def _as_str(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("this locator kind takes a string value")
    return value


def _is_ident(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][\w-]*", value))


def _xpath_literal(text: str) -> str:
    if "'" not in text:
        return f"'{text}'"
    if '"' not in text:
        return f'"{text}"'
    return "concat(" + ', "\'", '.join(f"'{part}'" for part in text.split("'")) + ")"


def _norm(text: str) -> str:
    return " ".join(text.split())


def _first_line(error: BaseException) -> str:
    return str(error).strip().split("\n", 1)[0][:200]
