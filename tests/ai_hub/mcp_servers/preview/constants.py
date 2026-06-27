from typing import Any

import yaml

MCP_SERVERS_INLINE_DATA: str = """\
mcp_servers:
  - name: kubernetes-mcp
    description: "Kubernetes cluster management"
    provider: "CNCF"
    version: "1.0.0"
  - name: kubernetes-admin
    description: "Kubernetes admin operations"
    provider: "CNCF"
    version: "0.9.0"
  - name: kubernetes-experimental
    description: "Experimental Kubernetes features"
    provider: "CNCF"
    version: "0.1.0"
  - name: prometheus-mcp
    description: "Prometheus metrics query"
    provider: "CNCF"
    version: "2.1.0"
  - name: grafana-dashboard
    description: "Grafana dashboard management"
    provider: "Grafana Labs"
    version: "1.5.0"
"""

MCP_SERVERS_LIST: list[dict[str, Any]] = yaml.safe_load(MCP_SERVERS_INLINE_DATA).get("mcp_servers", [])
MCP_SERVER_NAMES: set[str] = {server["name"] for server in MCP_SERVERS_LIST}
TOTAL_SERVERS: int = len(MCP_SERVERS_LIST)
