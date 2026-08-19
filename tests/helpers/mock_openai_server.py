"""Minimal OpenAI-compatible mock server for subprocess-based E2E tests.

Why this exists
---------------
Phase 3 crash-recovery (and any future challenge-runner that exercises the
real agent loop) needs to spawn `hermes chat -q "..."` in a subprocess and
kill it mid-turn. That requires an LLM endpoint the agent can actually talk
to — but we don't want to burn real API credits or depend on network in CI.

This server speaks just enough of the OpenAI Chat Completions API
(non-streaming) to drive one tool-call iteration of the agent. It is
deliberately tiny: ~150 lines, stdlib only (``http.server``), no external
deps. It returns a canned tool_call on the first request and a text
finish on the second, so the agent will run exactly one tool round-trip
before completing — long enough to be SIGTERM'd mid-loop.

Usage
-----
    from tests.helpers.mock_openai_server import MockOpenAIServer

    with MockOpenAIServer(tool_name="read_file") as srv:
        # srv.url  -> http://127.0.0.1:<port>/v1
        # srv.requests  -> list of received payloads (for assertions)
        subprocess.run(["hermes", "chat", "-q", "...",
                        "--base-url", srv.url, ...])

The server picks a free port on its own (no fixture coordination needed)
and serves on a background daemon thread, so it stops automatically when
the process exits (or via the context-manager ``__exit__``).

Limitations (intentional)
-------------------------
- Non-streaming only. The agent must run with ``display.streaming: false``
  (or ``_disable_streaming`` set). Streaming SSE would multiply complexity
  by 3x for no extra coverage here.
- One canned tool_call, then one text response. Not a general-purpose mock.
- No auth validation. The agent sends ``Authorization: Bearer test``; we
  ignore it.
- Ignores /models, /embeddings, etc. — only /chat/completions is handled.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, List, Optional


class _MockState:
    """Per-server mutable state shared with the handler."""

    def __init__(self, tool_name: str, tool_args: dict,
                 final_text: str, call_count_limit: int):
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.final_text = final_text
        self.call_count_limit = call_count_limit
        self.call_count = 0
        self.requests: List[dict] = []


def _make_handler(state: _MockState):
    class _Handler(BaseHTTPRequestHandler):
        # Silence default logging — keeps test output clean.
        def log_message(self, fmt, *args):  # noqa: D401
            pass

        def _send_json(self, code: int, body: dict):
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            # Capture the request body for later assertions.
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw or b"{}")
            except Exception:
                payload = {"_raw": raw.decode("utf-8", "replace")}
            state.requests.append(payload)

            # Only the chat-completions endpoint matters.
            if not self.path.endswith("/chat/completions"):
                self._send_json(404, {"error": {"message": "not found"}})
                return

            state.call_count += 1
            # After call_count_limit tool-call responses, switch to a plain
            # text finish so the agent loop terminates normally (if we let
            # it run that far — the test usually SIGTERMs earlier).
            if state.call_count <= state.call_count_limit:
                self._send_json(200, _tool_call_response(state))
            else:
                self._send_json(200, _text_response(state))

        def do_GET(self):
            # /models is sometimes probed by clients; answer minimally.
            if self.path.endswith("/models"):
                self._send_json(200, {"object": "list", "data": [
                    {"id": "mock-model", "object": "model"}
                ]})
            else:
                self._send_json(404, {"error": {"message": "not found"}})

    return _Handler


def _tool_call_response(state: _MockState) -> dict:
    """A chat.completions response with one tool_call — drives one tool round."""
    return {
        "id": f"chatcmpl-mock-{state.call_count}",
        "object": "chat.completion",
        "created": 0,
        "model": "mock-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call_{state.call_count}",
                    "type": "function",
                    "function": {
                        "name": state.tool_name,
                        "arguments": json.dumps(state.tool_args),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _text_response(state: _MockState) -> dict:
    """A plain text completion — ends the agent loop."""
    return {
        "id": f"chatcmpl-mock-{state.call_count}",
        "object": "chat.completion",
        "created": 0,
        "model": "mock-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": state.final_text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
    }


class MockOpenAIServer:
    """Context-managed mock OpenAI-compatible server.

    Example::

        with MockOpenAIServer(tool_name="terminal",
                              tool_args={"command": "echo hi"}) as srv:
            run_agent_against(srv.url)
    """

    def __init__(
        self,
        tool_name: str = "read_file",
        tool_args: Optional[dict] = None,
        final_text: str = "Done.",
        call_count_limit: int = 1,
    ):
        self._state = _MockState(
            tool_name=tool_name,
            tool_args=tool_args or {"path": "/tmp/nonexistent-mock.txt"},
            final_text=final_text,
            call_count_limit=call_count_limit,
        )
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    @property
    def requests(self) -> List[dict]:
        return self._state.requests

    @property
    def call_count(self) -> int:
        return self._state.call_count

    def __enter__(self) -> "MockOpenAIServer":
        # Pick a free port (port=0 lets the OS assign one).
        self._server = ThreadingHTTPServer(
            ("127.0.0.1", 0), _make_handler(self._state)
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(timeout=2)
        return False
