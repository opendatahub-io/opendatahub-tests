#!/usr/bin/env python3
"""Minimal Model Context Protocol server (streamable-http transport).

Python stdlib only, so it runs on a plain UBI python image. Implements the
JSON-RPC methods OGX's ``remote::model-context-protocol`` tool runtime uses:
``initialize``, ``notifications/initialized``, ``tools/list``, ``tools/call``
and ``ping``. ``GET`` returns 405, which the streamable-http transport spec
permits for servers without server-initiated streams.

Exposes a single deterministic tool::

    get_weather(city: str) -> "Weather in <city>: sunny, 21C"
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 9000
# The streamable-http endpoint this server is deployed under (must match
# MOCK_MCP_ENDPOINT_PATH in tests/ogx/mockupstream/constants.py).
ENDPOINT_PATH = "/mcp"
TOOL_NAME = "get_weather"
# Reject bodies larger than this before reading them, so a client cannot
# exhaust the pod's memory (128 MiB limit) with a single large POST.
MAX_REQUEST_BODY_BYTES = 8 * 1024 * 1024


class MockMCPHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # Socket timeout: a client that opens a request but never finishes sending
    # the body must not hold the handler thread forever.
    timeout = 60

    def _json_response(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        # No server-initiated streams; 405 is spec-permitted.
        self.send_response(405)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:
        # Only the configured endpoint speaks MCP; anything else is a client
        # bug (e.g. a dropped path in the connector URL) and must fail the
        # tests rather than silently succeed.
        if self.path != ENDPOINT_PATH:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = -1
        if length < 0 or length > MAX_REQUEST_BODY_BYTES:
            self.send_response(413)
            self.send_header("Content-Length", "0")
            self.close_connection = True
            self.end_headers()
            return
        message = json.loads(self.rfile.read(length)) if length else {}
        method = message.get("method")
        message_id = message.get("id")

        if method == "initialize":
            self._json_response({
                "jsonrpc": "2.0",
                "id": message_id,
                "result": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "mock-mcp", "version": "1.0.0"},
                },
            })
        elif method == "notifications/initialized":
            # A notification has no response body.
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif method == "tools/list":
            self._json_response({
                "jsonrpc": "2.0",
                "id": message_id,
                "result": {
                    "tools": [
                        {
                            "name": TOOL_NAME,
                            "description": "Get current weather for a city",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        }
                    ]
                },
            })
        elif method == "tools/call":
            params = message.get("params", {})
            name = params.get("name")
            if name == TOOL_NAME:
                # Validate the required argument and echo it in the result, so
                # the tests can detect OGX sending wrong tool arguments.
                city = (params.get("arguments") or {}).get("city")
                if not isinstance(city, str) or not city:
                    self._json_response({
                        "jsonrpc": "2.0",
                        "id": message_id,
                        "error": {"code": -32602, "message": "Missing required argument: city"},
                    })
                    return
                self._json_response({
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Weather in {city}: sunny, 21C"}],
                        "isError": False,
                    },
                })
            else:
                self._json_response({
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "error": {"code": -32602, "message": f"Unknown tool: {name}"},
                })
        elif method == "ping":
            self._json_response({"jsonrpc": "2.0", "id": message_id, "result": {}})
        else:
            self._json_response({
                "jsonrpc": "2.0",
                "id": message_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            })

    def log_message(self, fmt: str, *args) -> None:
        # Log the method only — never the raw path or query parameters, which
        # could carry request-derived values.
        print(f"mock-mcp: {self.command} request received", flush=True)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), MockMCPHandler)
    print(f"Mock MCP server running on http://0.0.0.0:{PORT}", flush=True)
    server.serve_forever()
