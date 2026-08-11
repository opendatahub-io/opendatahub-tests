class RhoaiMcpImages:
    RHOAI_MCP: str = (
        "quay.io/opendatahub/odh-rhoai-mcp@sha256:845f5bc5516c5681ecb886db6f154ee3c2eec995f40cada75f0c8a24a6b1c858"
    )
    # rhoai-mcp is not managed by an Operator; use the floating tag to test the latest build from the MCP catalog.
    RHOAI_MCP_ODH_STABLE: str = "quay.io/opendatahub/odh-rhoai-mcp:odh-stable"  # noqa: IMG002
