"""Mock upstream servers deployed in-cluster by the OGX feature test suites.

This package is not a test suite. It provides:

- ``constants``: names, ports and images of the mock deployments
- ``scripts``: the standalone (stdlib-only) mock server implementations
- ``deployments``: context managers that deploy the mocks into a test namespace
- ``ogx_config``: helpers for OGXServer specs deployed via ``spec.overrideConfig``

The mocks stand in for external dependencies so the suites run hermetically on
any cluster, without credentials or egress:

- **mock MaaS** — OpenAI-compatible chat-completions upstream for the
  ``remote::passthrough`` inference provider. Records every received request
  (path, headers, body) and exposes the recordings via debug endpoints, which
  is what lets tests assert on the headers OGX forwarded (token passthrough,
  OTLP export).
- **mock MCP** — Model Context Protocol server (streamable-http transport)
  exposing a single ``get_weather`` tool, used to verify OGX's MCP tool
  runtime end to end.
"""
