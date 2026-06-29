"""Utilities for Elyra extension testing.

This module contains reusable functions for interacting with Elyra
JupyterLab extensions and runtime configurations in workbench pods.
"""

import json
import re
from typing import Any

import structlog
from ocp_resources.pod import ExecOnPodError, Pod

from utilities.constants import Timeout
from utilities.general import collect_pod_information

LOGGER = structlog.get_logger(name=__name__)

# Path to user-created Elyra runtime configurations inside workbench container
ELYRA_RUNTIMES_DIR = "/opt/app-root/src/.local/share/jupyter/metadata/runtimes"


def parse_elyra_extensions(labextension_output: str) -> dict[str, dict[str, Any]]:
    """Parse jupyter labextension list output to extract Elyra-related extensions.

    Matches any extension with "elyra" in the name (case-insensitive).

    Args:
        labextension_output: Raw output from `jupyter labextension list` command

    Returns:
        Dict mapping extension names to metadata (version, enabled, status)

    Example:
        Input: "odh-elyra v1.0.0 enabled OK"
        Output: {"odh-elyra": {"version": "1.0.0", "enabled": True, "status": "OK"}}
    """
    elyra_extensions = {}

    for line in labextension_output.split("\n"):
        line = line.strip()

        # Skip empty lines and lines without "elyra" (case-insensitive)
        if not line or "elyra" not in line.lower():
            continue

        # Match extension line format: name v1.2.3 enabled/disabled OK/other-status
        # Extension names can include: @, /, ., -, and alphanumerics
        match = re.match(r"^([\w@/.-]+)\s+v([\d.]+)\s+(enabled|disabled)\s+(\S+)", line)
        if match:
            name, version, enabled_str, status = match.groups()
            elyra_extensions[name] = {
                "version": version,
                "enabled": enabled_str == "enabled",
                "status": status,
            }

    return elyra_extensions


def list_runtime_configs(pod: Pod, container: str) -> list[str]:
    """List Elyra runtime configuration files in the workbench pod.

    Args:
        pod: Workbench pod instance
        container: Name of the notebook container

    Returns:
        List of runtime config filenames (e.g., ["odh_dsp.json", "custom-runtime.json"])

    Raises:
        AssertionError: If command execution fails or directory doesn't exist
    """
    try:
        output = pod.execute(
            container=container,
            command=["sh", "-c", f"ls {ELYRA_RUNTIMES_DIR}/*.json 2>/dev/null || true"],
            timeout=Timeout.TIMEOUT_1MIN,
        )
    except ExecOnPodError as e:
        collect_pod_information(pod)
        raise AssertionError(
            f"Failed to list runtime configs in '{ELYRA_RUNTIMES_DIR}' on pod '{pod.name}': {e}"
        ) from e

    if not output or not output.strip():
        return []

    filenames = []
    for line in output.strip().split("\n"):
        line = line.strip()
        if line and line.endswith(".json"):
            filenames.append(line.split("/")[-1])

    return filenames


def read_runtime_config(pod: Pod, container: str, filename: str) -> dict[str, Any]:
    """Read and parse an Elyra runtime configuration file.

    Args:
        pod: Workbench pod instance
        container: Name of the notebook container
        filename: Name of the runtime config file (e.g., "odh_dsp.json")

    Returns:
        Parsed JSON content as dictionary

    Raises:
        AssertionError: If file read fails or JSON is invalid
    """
    file_path = f"{ELYRA_RUNTIMES_DIR}/{filename}"

    try:
        output = pod.execute(
            container=container,
            command=["cat", file_path],
            timeout=Timeout.TIMEOUT_1MIN,
        )
    except ExecOnPodError as e:
        collect_pod_information(pod)
        raise AssertionError(f"Failed to read runtime config '{file_path}' on pod '{pod.name}': {e}") from e

    try:
        return json.loads(output)
    except json.JSONDecodeError as e:
        raise AssertionError(
            f"Runtime config '{filename}' contains invalid JSON on pod '{pod.name}'. "
            f"Error: {e}. File size: {len(output)} bytes"
        ) from e


def compare_runtime_config_semantics(baseline: dict[str, Any], current: dict[str, Any], filename: str) -> list[str]:
    """Compare runtime configurations semantically, focusing on critical fields.

    Args:
        baseline: Runtime config captured before upgrade
        current: Runtime config after upgrade
        filename: Config filename for error messages

    Returns:
        List of difference descriptions (empty if configs match semantically)

    Critical fields compared:
        - display_name
        - schema_name
        - metadata.runtime_type
        - metadata.api_endpoint

    Fields ignored:
        - Timestamps, ordering, extra fields
    """
    differences = []

    critical_fields = ["display_name", "schema_name"]
    for field in critical_fields:
        baseline_value = baseline.get(field)
        current_value = current.get(field)

        if baseline_value != current_value:
            differences.append(f"Field '{field}' changed: '{baseline_value}' -> '{current_value}'")

    baseline_metadata = baseline.get("metadata", {})
    current_metadata = current.get("metadata", {})

    metadata_critical_fields = ["runtime_type", "api_endpoint"]
    for field in metadata_critical_fields:
        baseline_value = baseline_metadata.get(field)
        current_value = current_metadata.get(field)

        if baseline_value != current_value:
            differences.append(f"metadata.{field} changed: '{baseline_value}' -> '{current_value}'")

    return differences
