class RhoaiMcpImages:
    RHOAI_MCP_RHOAI_DIGEST: str = (
        "registry.redhat.io/rhoai/odh-rhoai-mcp-rhel9@sha256:985b3251644445cd5375d7deb2ae5d7853529b199ac10c1af9cb1d445ef539e3"
    )
    # rhoai-mcp is not managed by an Operator; use the floating tag to test the latest build from the MCP catalog.
    RHOAI_MCP_ODH_STABLE: str = "quay.io/opendatahub/odh-rhoai-mcp:odh-stable"  # noqa: IMG002
    RHOAI_MCP_RHOAI_VERSION: str = "registry.redhat.io/rhoai/odh-rhoai-mcp-rhel9:rhoai-3.6-ea.1"  # noqa: IMG002
