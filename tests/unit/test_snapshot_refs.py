"""Model-facing refs: issued once, never reused, stale the moment they leave the snapshot."""

from __future__ import annotations

import pytest

from understudy.surface.base import StaleRef
from understudy.surface.snapshot import RefTable, parse_line, scrub_for_model

FIRST = """- generic [active] [ref=e1]:
  - heading "Open New Account" [level=1] [ref=e2]
  - combobox [ref=e3]:
    - option "CHECKING" [selected]
  - button "Open New Account" [ref=e4] [cursor=pointer]"""

# The same document after the panel switched: e2 and e3 are gone, e5 is new.
SECOND = """- generic [active] [ref=e1]:
  - button "Open New Account" [ref=e4] [cursor=pointer]
  - heading "Account Opened!" [level=1] [ref=e5]"""

# A navigation: Playwright gives the new document a frame prefix.
AFTER_NAVIGATION = """- generic [active] [ref=f2e1]:
  - heading "Open New Account" [level=1] [ref=f2e2]"""


def test_refs_are_rewritten_in_order() -> None:
    table = RefTable()
    out = table.rewrite(FIRST)
    assert "[ref=e" not in out
    assert '- heading "Open New Account" [level=1] [ref=ref_2]' in out
    assert '- button "Open New Account" [ref=ref_4] [cursor=pointer]' in out
    node = table.resolve("ref_4")
    assert (node.pw_ref, node.role, node.name) == ("e4", "button", "Open New Account")


def test_an_element_keeps_its_ref_across_snapshots() -> None:
    table = RefTable()
    table.rewrite(FIRST)
    out = table.rewrite(SECOND)
    assert "[ref=ref_4]" in out
    assert "[ref=ref_5]" in out
    assert table.resolve("ref_4").pw_ref == "e4"


def test_a_ref_that_left_the_snapshot_is_stale_not_reassigned() -> None:
    table = RefTable()
    table.rewrite(FIRST)
    table.rewrite(SECOND)
    with pytest.raises(StaleRef, match="ref_2 is not in the latest snapshot"):
        table.resolve("ref_2")


def test_a_new_document_never_reuses_old_model_refs() -> None:
    table = RefTable()
    table.rewrite(FIRST)
    out = table.rewrite(AFTER_NAVIGATION)
    assert "[ref=ref_5]" in out and "[ref=ref_6]" in out
    with pytest.raises(StaleRef):
        table.resolve("ref_1")


def test_an_unknown_ref_says_so() -> None:
    table = RefTable()
    table.rewrite(FIRST)
    with pytest.raises(StaleRef, match="not a ref this run issued"):
        table.resolve("ref_99")


def test_generation_counts_snapshots() -> None:
    table = RefTable()
    table.rewrite(FIRST)
    table.rewrite(SECOND)
    assert table.generation == 2


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ('  - button "Open New Account" [ref=e4] [cursor=pointer]', ("button", "Open New Account")),
        ('- link "Say \\"hi\\"" [ref=e9]', ("link", 'Say "hi"')),
        ("- combobox [ref=e3]:", ("combobox", None)),
        ("- generic [active] [ref=e1]:", (None, None)),
        ("- paragraph [ref=e6]: Welcome John Smith", ("paragraph", None)),
        ('- heading "Error!" [level=1]', ("heading", "Error!")),
        ("  - /url: admin.htm", (None, None)),
    ],
)
def test_parse_line(line: str, expected: tuple[str | None, str | None]) -> None:
    assert parse_line(line) == expected


def test_the_model_snapshot_hides_passwords_by_value_and_drops_session_ids() -> None:
    snapshot = "\n".join(
        [
            "- textbox [ref=e3]: john",
            "- textbox [ref=e4]: hunter2",
            '- link "Register" [ref=e9]:',
            "  - /url: register.htm;jsessionid=C66C519E79FF448D",
        ]
    )
    out = scrub_for_model(snapshot, {"hunter2", ""})
    assert "- textbox [ref=e3]: john" in out
    assert "- textbox [ref=e4]: [SECRET]" in out
    assert "  - /url: register.htm" in out and "jsessionid" not in out
