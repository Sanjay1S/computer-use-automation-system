"""The operator console over real HTTP on 127.0.0.1, against a control plane with no
browser behind it: what it serves, what it refuses, and that a decision made through
it lands on the plane."""

from __future__ import annotations

import http.client
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from understudy.control.console import Console
from understudy.control.interventions import Intervention, Subject
from understudy.control.plane import ControlPlane

RUN = "replay-test-run"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


@pytest.fixture
def plane() -> ControlPlane:
    return ControlPlane()


@pytest.fixture
def console(plane: ControlPlane, tmp_path: Path) -> Iterator[Console]:
    with Console(plane, tmp_path, port=0) as running:
        yield running


def opened(
    plane: ControlPlane, tmp_path: Path, screenshot: str | None = "shot.png"
) -> Intervention:
    (tmp_path / RUN).mkdir(exist_ok=True)
    (tmp_path / RUN / "shot.png").write_bytes(PNG)
    (tmp_path / "outside.png").write_bytes(PNG)
    lease = plane.start(RUN, automation="engine")
    return plane.open(
        lease,
        kind="takeover",
        subject=Subject(kind="capability", id="parabank.open_account"),
        reason="unknown_blocking_state",
        detail="a modal covers the target",
        screenshot=screenshot,
        resume_points=["s3_funding_account"],
    )


def request(
    console: Console,
    method: str,
    path: str,
    body: Any = None,
    *,
    host: str | None = None,
    content_type: str = "application/json",
    token: str | None = None,
) -> tuple[int, Any, dict[str, str]]:
    connection = http.client.HTTPConnection("127.0.0.1", console.port, timeout=5)
    headers = {"Host": host or f"127.0.0.1:{console.port}"}
    data = None
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        headers["Content-Type"] = content_type
    connection.request(method, f"/{token or console.token}{path}", body=data, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    found = dict(response.getheaders())
    connection.close()
    if found.get("Content-Type", "").startswith("application/json"):
        return response.status, json.loads(raw), found
    return response.status, raw, found


def test_the_page_and_the_state_are_served_under_the_token(
    console: Console, plane: ControlPlane, tmp_path: Path
) -> None:
    intervention = opened(plane, tmp_path)

    status, page, headers = request(console, "GET", "/")
    assert status == 200 and b"Operator console" in page
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert headers["Cache-Control"] == "no-store"

    status, state, _ = request(console, "GET", "/api/state")
    assert status == 200
    assert state["run_id"] == RUN and state["state"] == "AWAITING_HUMAN"
    assert [i["id"] for i in state["interventions"]] == [intervention.id]
    assert console.url == f"http://127.0.0.1:{console.port}/{console.token}/"


def test_a_wrong_token_or_host_gets_nothing(
    console: Console, plane: ControlPlane, tmp_path: Path
) -> None:
    opened(plane, tmp_path)

    assert request(console, "GET", "/api/state", token="not-the-token")[0] == 404
    # DNS rebinding: a page on another name that resolves here sends its own Host.
    assert request(console, "GET", "/api/state", host="evil.example:80")[0] == 404
    assert request(console, "GET", "/api/state", host=f"localhost:{console.port}")[0] == 200


def test_the_bare_token_path_redirects_to_its_slash(console: Console) -> None:
    status, _, headers = request(console, "GET", "")
    assert status == 301
    assert headers["Location"] == f"/{console.token}/"


def test_a_claim_and_a_decision_land_on_the_plane(
    console: Console, plane: ControlPlane, tmp_path: Path
) -> None:
    intervention = opened(plane, tmp_path)
    base = f"/api/interventions/{intervention.id}"

    status, claimed, _ = request(console, "POST", f"{base}/claim", {"operator": "dana"})
    assert status == 200 and claimed["status"] == "claimed"
    assert plane.state == "HUMAN" and plane.lease.operator == "dana"

    status, refused, _ = request(console, "POST", f"{base}/resume", {"operator": "lee"})
    assert status == 409 and "dana holds this session" in refused["error"]

    status, decided, _ = request(
        console, "POST", f"{base}/resume", {"operator": "dana", "note": "closed it"}
    )
    assert status == 200
    assert decided["decision"]["action"] == "resume"
    assert decided["decision"]["note"] == "closed it"


@pytest.mark.parametrize(
    ("body", "content_type", "expected"),
    [
        ({"operator": "dana"}, "text/plain", 415),
        (b"{not json", "application/json", 400),
        ([1, 2], "application/json", 400),
        ({"operator": 7}, "application/json", 400),
        ({"note": "no name"}, "application/json", 400),
        ({"operator": "dana", "pad": "x" * 20_000}, "application/json", 400),
    ],
)
def test_malformed_decisions_are_refused(
    console: Console,
    plane: ControlPlane,
    tmp_path: Path,
    body: Any,
    content_type: str,
    expected: int,
) -> None:
    intervention = opened(plane, tmp_path)

    status, _, _ = request(
        console,
        "POST",
        f"/api/interventions/{intervention.id}/claim",
        body,
        content_type=content_type,
    )
    assert status == expected
    assert plane.state == "AWAITING_HUMAN"


def test_unknown_actions_and_interventions_are_404(
    console: Console, plane: ControlPlane, tmp_path: Path
) -> None:
    intervention = opened(plane, tmp_path)

    assert request(console, "POST", f"/api/interventions/{intervention.id}/explode", {})[0] == 404
    assert (
        request(console, "POST", "/api/interventions/iv-nosuch/claim", {"operator": "d"})[0] == 404
    )
    assert request(console, "GET", "/api/nothing")[0] == 404


def test_the_screenshot_is_served_from_inside_the_run_folder_only(
    console: Console, plane: ControlPlane, tmp_path: Path
) -> None:
    intervention = opened(plane, tmp_path)
    status, data, headers = request(
        console, "GET", f"/api/interventions/{intervention.id}/screenshot"
    )
    assert status == 200 and data == PNG and headers["Content-Type"] == "image/png"


def test_a_screenshot_path_outside_the_run_folder_is_never_served(
    console: Console, plane: ControlPlane, tmp_path: Path
) -> None:
    intervention = opened(plane, tmp_path, screenshot="../outside.png")

    status, _, _ = request(console, "GET", f"/api/interventions/{intervention.id}/screenshot")
    assert status == 404
