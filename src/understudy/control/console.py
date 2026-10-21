"""The operator console: a minimal local web page for interventions.

Stdlib HTTP on a background thread, bound to 127.0.0.1, with a random token in
every path. It reads the control plane and records claims and decisions on it. It
never touches the browser: the operator acts in the live window itself, and
automation acts on their decision.

    GET  /<token>/                                       the page
    GET  /<token>/api/state                              state, lease, interventions
    GET  /<token>/api/interventions/<id>/screenshot      masked; ?live=1 for the latest
    POST /<token>/api/interventions/<id>/<action>        {"operator", "note", "resume_from"}

Stated limits: localhost and a URL token, not SSO or roles; the operator name is
whatever they type.
"""

from __future__ import annotations

import json
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

from understudy.control.interventions import OperatorAction
from understudy.control.plane import ControlPlane, Refused

PAGE = (Path(__file__).parent / "console.html").read_text(encoding="utf-8")
MAX_BODY = 16_384
ACTIONS = frozenset({"claim", "approve", "deny", "resume", "complete", "abort"})
DEFAULT_PORT = 8765


class Console:
    def __init__(
        self,
        plane: ControlPlane,
        runs_dir: Path,
        *,
        port: int = DEFAULT_PORT,
        token: str | None = None,
    ) -> None:
        self.plane = plane
        self.runs_dir = runs_dir
        self.token = token or secrets.token_urlsafe(18)
        self._server = ThreadingHTTPServer(("127.0.0.1", port), self._handler())
        self._server.daemon_threads = True
        self.port = int(self._server.server_address[1])
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/{self.token}/"

    def start(self) -> str:
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="understudy-console", daemon=True
        )
        self._thread.start()
        return self.url

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> Console:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # --- requests ------------------------------------------------------------------

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        console = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "understudy-console"

            def do_GET(self) -> None:
                parts = console._parts(self)
                if parts is None:
                    return
                route, query = parts
                if route == []:
                    return console._send(self, HTTPStatus.OK, PAGE, "text/html; charset=utf-8")
                if route == ["api", "state"]:
                    return console._json(self, HTTPStatus.OK, console.plane.snapshot())
                if (
                    len(route) == 4
                    and route[:2] == ["api", "interventions"]
                    and (route[3] == "screenshot")
                ):
                    return console._screenshot(self, route[2], live="live" in query)
                console._error(self, HTTPStatus.NOT_FOUND, "not found")

            def do_POST(self) -> None:
                parts = console._parts(self)
                if parts is None:
                    return
                route, _ = parts
                if len(route) != 4 or route[:2] != ["api", "interventions"]:
                    return console._error(self, HTTPStatus.NOT_FOUND, "not found")
                console._decide(self, route[2], route[3])

            def log_message(self, format: str, *args: object) -> None:
                return  # request lines carry the token

        return Handler

    def _parts(self, request: BaseHTTPRequestHandler) -> tuple[list[str], dict[str, Any]] | None:
        """The route below the token, or None after answering. A wrong token is a 404,
        as is a Host header naming anything but this console (DNS rebinding)."""
        host = request.headers.get("Host", "")
        if host not in (f"127.0.0.1:{self.port}", f"localhost:{self.port}"):
            self._error(request, HTTPStatus.NOT_FOUND, "not found")
            return None
        split = urlsplit(request.path)
        segments = split.path.split("/")
        if len(segments) < 2 or not secrets.compare_digest(segments[1], self.token):
            self._error(request, HTTPStatus.NOT_FOUND, "not found")
            return None
        if len(segments) == 2:  # /<token> without its slash: relative links need it
            request.send_response(HTTPStatus.MOVED_PERMANENTLY)
            request.send_header("Location", f"/{self.token}/")
            request.end_headers()
            return None
        return [s for s in segments[2:] if s], parse_qs(split.query)

    def _decide(self, request: BaseHTTPRequestHandler, intervention_id: str, action: str) -> None:
        if action not in ACTIONS:
            return self._error(request, HTTPStatus.NOT_FOUND, f"unknown action {action!r}")
        if request.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            return self._error(request, HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "send JSON")
        try:
            length = int(request.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if not 0 <= length <= MAX_BODY:
            return self._error(request, HTTPStatus.BAD_REQUEST, "a JSON body of at most 16 KB")
        try:
            body = json.loads(request.rfile.read(length) or b"{}")
        except ValueError:
            return self._error(request, HTTPStatus.BAD_REQUEST, "the body is not JSON")
        if not isinstance(body, dict):
            return self._error(request, HTTPStatus.BAD_REQUEST, "the body must be an object")
        fields = {k: body.get(k) for k in ("operator", "note", "resume_from")}
        if any(v is not None and not isinstance(v, str) for v in fields.values()):
            return self._error(request, HTTPStatus.BAD_REQUEST, "fields must be strings")
        try:
            updated = self.plane.act(
                intervention_id,
                cast(OperatorAction, action),
                operator=fields["operator"] or "",
                note=fields["note"],
                resume_from=fields["resume_from"] or None,
            )
        except KeyError:
            return self._error(request, HTTPStatus.NOT_FOUND, "no such intervention")
        except Refused as refusal:
            return self._error(request, HTTPStatus.CONFLICT, str(refusal))
        except ValueError as error:
            return self._error(request, HTTPStatus.BAD_REQUEST, str(error))
        self._json(request, HTTPStatus.OK, updated.model_dump(mode="json"))

    def _screenshot(
        self, request: BaseHTTPRequestHandler, intervention_id: str, live: bool
    ) -> None:
        try:
            intervention = self.plane.get(intervention_id)
        except KeyError:
            return self._error(request, HTTPStatus.NOT_FOUND, "no such intervention")
        relative = (live and intervention.live_screenshot) or intervention.screenshot
        run_id = self.plane.run_id
        if not relative or run_id is None:
            return self._error(request, HTTPStatus.NOT_FOUND, "no screenshot")
        run_dir = (self.runs_dir / run_id).resolve()
        path = (run_dir / relative).resolve()
        if not path.is_relative_to(run_dir) or not path.is_file():
            return self._error(request, HTTPStatus.NOT_FOUND, "no screenshot")
        try:
            data = path.read_bytes()
        except OSError:  # being rewritten by the live refresh
            return self._error(request, HTTPStatus.SERVICE_UNAVAILABLE, "try again")
        self._send(request, HTTPStatus.OK, data, "image/png")

    # --- responses -------------------------------------------------------------------

    def _json(self, request: BaseHTTPRequestHandler, status: HTTPStatus, payload: Any) -> None:
        self._send(request, status, json.dumps(payload, default=str), "application/json")

    def _error(self, request: BaseHTTPRequestHandler, status: HTTPStatus, message: str) -> None:
        self._json(request, status, {"error": message})

    @staticmethod
    def _send(
        request: BaseHTTPRequestHandler, status: HTTPStatus, body: str | bytes, content_type: str
    ) -> None:
        data = body.encode("utf-8") if isinstance(body, str) else body
        request.send_response(status)
        request.send_header("Content-Type", content_type)
        request.send_header("Content-Length", str(len(data)))
        request.send_header("Cache-Control", "no-store")
        request.send_header("X-Content-Type-Options", "nosniff")
        request.send_header("Referrer-Policy", "no-referrer")
        request.send_header(
            "Content-Security-Policy",
            "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; "
            "script-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'",
        )
        request.end_headers()
        request.wfile.write(data)
