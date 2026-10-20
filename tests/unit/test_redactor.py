"""Redaction: value classes, field classification, free text, URLs, snapshots."""

from __future__ import annotations

import pytest

from understudy.redaction.redactor import KEY_ENV, Redactor

KEY = b"test-key"


@pytest.fixture
def r() -> Redactor:
    return Redactor(KEY)


def test_value_classes(r: Redactor) -> None:
    assert r.value("hunter2", "secret") == "[SECRET]"
    assert r.value("13344", "financial_id") == "••••4"  # a third, at most
    assert r.value("4111111111111111", "financial_id") == "••••1111"
    assert r.value("123", "financial_id") == "••••3"
    assert r.value("12", "financial_id") == "••••"
    assert r.value("SAVINGS", "public") == "SAVINGS"
    tag = r.value("jane@example.com", "pii")
    assert tag.startswith("[PII:") and len(tag) == len("[PII:12345678]")
    assert "jane" not in tag


def test_pii_tags_correlate_under_one_key_and_differ_across_keys(r: Redactor) -> None:
    assert r.value("Jane Doe", "pii") == Redactor(KEY).value("Jane Doe", "pii")
    assert r.value("Jane Doe", "pii") != Redactor(b"other").value("Jane Doe", "pii")
    assert r.value("Jane Doe", "pii") != r.value("John Doe", "pii")


def test_from_env_uses_a_stable_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, "configured")
    assert Redactor.from_env().value("x1y2", "pii") == Redactor.from_env().value("x1y2", "pii")
    monkeypatch.delenv(KEY_ENV)
    assert Redactor.from_env().value("x1y2", "pii") != Redactor.from_env().value("x1y2", "pii")


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"input_type": "password"}, "secret"),
        ({"autocomplete": "one-time-code"}, "secret"),
        ({"label": "PIN"}, "secret"),
        ({"autocomplete": "cc-number"}, "financial_id"),
        ({"label": "Account #:"}, "financial_id"),
        ({"name_attr": "card_number"}, "financial_id"),
        ({"label": "Social Security Number"}, "pii"),
        ({"name_attr": "customer.ssn"}, "pii"),
        ({"label": "Username"}, "pii"),
        ({"input_type": "email"}, "pii"),
        ({"name_attr": "payee.phoneNumber"}, "pii"),
        ({"label": "Amount"}, "public"),
        ({"name_attr": "fromAccountId"}, "public"),
    ],
)
def test_field_classification(kwargs: dict[str, str], expected: str) -> None:
    assert Redactor.classify_field(**kwargs) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "secret",
    [
        "123-45-6789",
        "jane.doe+bank@example.co.uk",
        "Bearer abc.def-ghi",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.c2lnbmF0dXJlLXZhbHVl",
        "sk-ant-api03-abcdefghijklmnopqrstuvwxyz",
        "C66C519E79FF448DF8A46FD3B361B6C5",
        "4111 1111 1111 1111",
        "123456789012",
    ],
)
def test_regex_backstop_catches_unclassified_values(r: Redactor, secret: str) -> None:
    out = r.text(f"error near {secret} here")
    assert secret not in out
    assert out.startswith("error near ") and out.endswith(" here")


def test_known_values_are_masked_wherever_they_reappear(r: Redactor) -> None:
    known = r.knowing({"13344": "financial_id", "Jane Q": "pii", "SAVINGS": "public"})
    out = known.text("locator text='13344' and name 'Jane Q' chose SAVINGS")
    assert "13344" not in out and "3344" not in out and "••••4" in out
    assert "Jane Q" not in out
    assert "SAVINGS" in out


def test_text_drops_session_ids(r: Redactor) -> None:
    assert r.text("see lookup.htm;jsessionid=C66C519E for help") == "see lookup.htm for help"


def test_url_strips_session_ids_and_sensitive_query_values(r: Redactor) -> None:
    out = r.url(
        "http://localhost:8089/parabank/login.htm;jsessionid=C66C519E79FF448D"
        "?token=abc&id=123-45-6789&type=SAVINGS#access_token=zzz"
    )
    assert out == (
        "http://localhost:8089/parabank/login.htm?token=[SECRET]&id=[PII:"
        + out.split("[PII:")[1].split("]")[0]
        + "]&type=SAVINGS"
    )
    assert "jsessionid" not in out.lower() and "zzz" not in out


def test_url_masks_known_values_in_paths(r: Redactor) -> None:
    known = r.knowing({"13344": "financial_id"})
    assert known.url("http://h/parabank/accounts/13344") == "http://h/parabank/accounts/••••4"


def test_unregistered_account_ids_are_masked_in_urls_and_text(r: Redactor) -> None:
    assert r.url("http://h/parabank/activity.htm?id=12345") == (
        "http://h/parabank/activity.htm?id=••••5"
    )
    assert r.text("Could not find account #54321 in 2026") == (
        "Could not find account #••••1 in 2026"
    )


def test_pii_tags_survive_the_digit_backstop(r: Redactor) -> None:
    tag = r.value("Jane Doe", "pii")
    assert tag[5:-1].isalpha()
    assert r.text(f"user {tag}") == f"user {tag}"


def test_snapshot_drops_typed_values_and_masks_digit_runs(r: Redactor) -> None:
    snapshot = "\n".join(
        [
            '- textbox "Payee Name:" [ref=e7]: Jane Q Public',
            "- textbox [ref=e8]: hunter2",
            "- combobox [ref=e5]:",
            '  - option "13344" [selected]',
            '- link "Open New Account" [ref=e9]:',
            "  - /url: openaccount.htm;jsessionid=C66C519E",
        ]
    )
    out = r.snapshot(snapshot)
    assert '- textbox "Payee Name:" [ref=e7]: [value]' in out
    assert "- textbox [ref=e8]: [value]" in out
    assert "hunter2" not in out and "Jane Q" not in out
    assert '- option "••••4" [selected]' in out
    assert "- combobox [ref=e5]:" in out
    assert "jsessionid" not in out


def test_data_redacts_nested_strings(r: Redactor) -> None:
    out = r.data({"a": ["123-45-6789", 4], "b": {"c": "ok"}, "n": None})
    assert out["a"][0] != "123-45-6789" and out["a"][1] == 4
    assert out["b"] == {"c": "ok"} and out["n"] is None
