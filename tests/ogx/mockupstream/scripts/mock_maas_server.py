#!/usr/bin/env python3
"""Mock MaaS / genai-proxy upstream for the remote::passthrough provider.

Stands in for the Gen AI BFF's genai-proxy endpoint (the upstream the
``genai-bff-proxy`` ``remote::passthrough`` inference provider forwards to).
Python stdlib only, so it runs on a plain UBI python image.

Endpoints:

- ``POST /v1/chat/completions`` — OpenAI-compatible chat completion.
  When the request carries ``tools`` and no prior tool result, a fixed
  ``get_weather`` tool call is returned (drives OGX's server-side tool loop);
  once a tool result is present in the messages, a final answer echoing the
  tool output is returned. Supports both plain JSON and SSE streaming
  (``stream: true``), which is what OGX's built-in responses provider uses.
- ``GET /v1/models`` — minimal model listing.
- ``GET /healthz`` — readiness probe target.
- ``GET /v1/debug/last-headers`` — the last recorded request (path, headers,
  body). Headers are normalized to lowercase (HTTP headers are
  case-insensitive).
- ``GET /v1/debug/requests`` — the last ``MAX_RECORDED_REQUESTS`` requests.
- any other ``POST`` — recorded and answered with 200 (lets the mock act as a
  passive receiver, e.g. for OGX's OTLP metrics export).
"""

import json
import re
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 9999
MAX_RECORDED_REQUESTS = 100
RECORDED_REQUESTS: deque[dict] = []


def _record(path: str, headers, body: dict | None) -> None:
    RECORDED_REQUESTS.append({
        "path": path,
        "headers": {k.lower(): v for k, v in headers.items()},
        "body": body,
    })
    while len(RECORDED_REQUESTS) > MAX_RECORDED_REQUESTS:
        RECORDED_REQUESTS.popleft()


class MockMaaSHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, chunks: list[dict]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for chunk in chunks:
            data = (json.dumps(chunk) + "\n\n").encode()
            payload = b"data: " + data
            self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")

    # --- response builders -------------------------------------------------

    def _base_chunk(self, model: str) -> dict:
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "created": 1770000000,
            "model": model,
            "object": "chat.completion.chunk",
        }

    def _tool_call_chunks(self, model: str, city: str) -> list[dict]:
        return [
            {
                **self._base_chunk(model),
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}],
            },
            {
                **self._base_chunk(model),
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_weather_1",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": json.dumps({"city": city})},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {**self._base_chunk(model), "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ]

    def _final_chunks(self, model: str, text: str) -> list[dict]:
        return [
            {
                **self._base_chunk(model),
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
            },
            {**self._base_chunk(model), "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]

    def _tool_call_response(self, model: str, city: str) -> dict:
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": 1770000000,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_weather_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": json.dumps({"city": city})},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }

    def _final_response(self, model: str, text: str) -> dict:
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": 1770000000,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }

    # --- HTTP methods --------------------------------------------------------

    def do_GET(self) -> None:
        if self.path == "/v1/debug/last-headers":
            self._send_json(RECORDED_REQUESTS[-1] if RECORDED_REQUESTS else {})
        elif self.path == "/v1/debug/requests":
            self._send_json({"data": list(RECORDED_REQUESTS)})
        elif self.path == "/v1/models":
            self._send_json({
                "object": "list",
                "data": [{"id": "test-maas-model", "object": "model", "owned_by": "mock-maas"}],
            })
        elif self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else {}
        # OTLP exports and other binary payloads (protobuf) are recorded as-is.
        except ValueError:
            body = {"_raw_length": len(raw)}

        _record(path=self.path, headers=self.headers, body=body)

        if self.path != "/v1/chat/completions":
            # Passive receiver (e.g. OTLP export): accept and record anything.
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        messages = body.get("messages", [])
        model = body.get("model", "test-maas-model")
        stream = bool(body.get("stream"))
        has_tool_result = any(message.get("role") == "tool" for message in messages)
        tools = body.get("tools") or []

        if tools and not has_tool_result:
            user_text = next(
                (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
                "weather in Boston",
            )
            city_match = re.search(r"(?:in|for)\s+([A-Z][a-zA-Z]+)", user_text or "")
            city = city_match.group(1) if city_match else "Boston"
            if stream:
                self._send_sse(self._tool_call_chunks(model, city))
            else:
                self._send_json(self._tool_call_response(model, city))
        elif has_tool_result:
            tool_content = next((m.get("content") for m in messages if m.get("role") == "tool"), "no tool output")
            text = f"final-answer: {tool_content}"
            if stream:
                self._send_sse(self._final_chunks(model, text))
            else:
                self._send_json(self._final_response(model, text))
        else:
            # Plain request (no tools): echo the last user message.
            user_text = next(
                (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
                "hello",
            )
            text = f"mock-maas echo: {user_text}"
            if stream:
                self._send_sse(self._final_chunks(model, text))
            else:
                self._send_json(self._final_response(model, text))

    def log_message(self, fmt: str, *args) -> None:
        print(f"mock-maas: {self.path}", flush=True)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), MockMaaSHandler)
    print(f"Mock MaaS server running on http://0.0.0.0:{PORT}/v1", flush=True)
    server.serve_forever()
