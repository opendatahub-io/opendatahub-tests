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
TOOL_NAME = "get_weather"
TOOL_RESULT_TEXT = "Weather in Boston: sunny, 21C"


class MockMCPHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

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
        length = int(self.headers.get("Content-Length", 0))
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
                self._json_response({
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": {
                        "content": [{"type": "text", "text": TOOL_RESULT_TEXT}],
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
        print(f"mock-mcp: {self.path}", flush=True)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), MockMCPHandler)
    print(f"Mock MCP server running on http://0.0.0.0:{PORT}", flush=True)
    server.serve_forever()
