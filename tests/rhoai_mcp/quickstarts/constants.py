RHOAI_MCP_QUICKSTART_NAME: str = "peoplemesh"
RHOAI_MCP_UNKNOWN_QUICKSTART_NAME: str = "no-such-quickstart"
RHOAI_MCP_QUICKSTART_REQUIRED_ACTIONS: tuple[str, ...] = ("CHECK_PRE_REQS", "INSTALL")
RHOAI_MCP_QUICKSTART_OPEN_TOOLS: tuple[str, ...] = (
    "list_quickstarts",
    "get_quickstart_manifest",
)
RHOAI_MCP_QUICKSTART_GOVERNED_TOOLS: tuple[str, ...] = (
    "run_quickstart_action",
    "get_quickstart_action_status",
    "get_quickstart_action_logs",
)
RHOAI_MCP_QUICKSTART_JOB_NAMESPACE: str = "openshift-quickstarts"
RHOAI_MCP_QUICKSTART_CHECK_PRE_REQS_ACTION: str = "CHECK_PRE_REQS"
RHOAI_MCP_QUICKSTART_LOGS_UNAVAILABLE: str = "<logs unavailable>"
# Each finished installer Job phase and the result status the installer reports with it.
RHOAI_MCP_QUICKSTART_PRE_REQS_OUTCOMES: dict[str, str] = {
    "Complete": "success",
    "Failed": "prerequisites_failed",
}
