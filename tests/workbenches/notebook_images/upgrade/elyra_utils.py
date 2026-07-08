"""Utilities for Elyra extension testing.

This module contains reusable functions for interacting with Elyra
JupyterLab extensions and runtime configurations in workbench pods.
"""

import json
import re
from typing import TYPE_CHECKING, Any

import structlog
from ocp_resources.notebook import Notebook
from ocp_resources.pod import ExecOnPodError, Pod

from utilities.general import collect_pod_information

if TYPE_CHECKING:
    from tests.workbenches.notebook_images.utils import WorkbenchImageBaseline, WorkbenchImageSpec

LOGGER = structlog.get_logger(name=__name__)

# Path to Elyra runtime configurations inside workbench container
ELYRA_RUNTIMES_DIR = "/opt/app-root/src/.local/share/jupyter/metadata/runtimes"


def parse_elyra_extensions(labextension_output: str) -> dict[str, dict[str, Any]]:
    """Parse jupyter labextension list output to extract Elyra-related extensions.

    Matches any extension with "elyra" in the name (case-insensitive).
    Extension names can include: @, /, ., -, and alphanumerics

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

        if not line or "elyra" not in line.lower():
            continue

        # Match extension line format: name v1.2.3 enabled/disabled OK/other-status
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
            timeout=60,
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
            timeout=60,
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

    Fields compared:
        - display_name: User-visible name in Elyra UI
        - schema_name: Runtime type identifier
        - metadata.runtime_type: Backend type (e.g., 'KUBEFLOW_PIPELINES')
        - metadata.api_endpoint: Pipeline API URL
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


def verify_pre_upgrade_elyra_installed(
    pod: Pod,
    notebook: Notebook,
    spec: WorkbenchImageSpec,
) -> None:
    """Verify Elyra is installed and healthy.

    This function asserts that Elyra extensions are present and healthy.
    If Elyra is missing, the test fails.

    Use this function only in test files for images that must have Elyra
    (e.g., JupyterLab datascience images). Do not use for minimal images.
    """
    try:
        output = pod.execute(
            container=notebook.name,
            command=["jupyter", "labextension", "list"],
            timeout=60,
        )
    except ExecOnPodError as e:
        LOGGER.error(f"Failed to execute 'jupyter labextension list' on pod '{pod.name}': {e}")
        raise AssertionError(
            f"Failed to execute 'jupyter labextension list' command on pod '{pod.name}'. "
            f"Cannot verify Elyra installation. Error: {e}"
        ) from e

    elyra_extensions = parse_elyra_extensions(labextension_output=output)

    assert elyra_extensions, (
        f"No Elyra extensions found in {spec.ide} workbench image. "
        f"Elyra is required for this image type but is not installed. "
        f"Expected to find Elyra extensions via 'jupyter labextension list'."
    )

    # Verify all extensions are healthy (enabled + OK status)
    unhealthy = []
    for name, metadata in elyra_extensions.items():
        if not (metadata["enabled"] and metadata["status"] == "OK"):
            unhealthy.append(name)

    assert not unhealthy, (
        f"Found {len(unhealthy)} unhealthy Elyra extension(s): {', '.join(unhealthy)}. "
        f"All Elyra extensions must be enabled with OK status."
    )


def verify_post_upgrade_elyra_extensions_preserved(
    pod: Pod,
    notebook: Notebook,
    baseline: WorkbenchImageBaseline,
) -> None:
    """Verify Elyra extensions survived upgrade unchanged.

    If baseline.elyra_extensions is None, the test fails.
    """
    assert baseline.elyra_extensions is not None, (
        "No Elyra extensions in baseline. This test requires Elyra to have been present pre-upgrade."
    )

    try:
        output = pod.execute(
            container=notebook.name,
            command=["jupyter", "labextension", "list"],
            timeout=60,
        )
    except ExecOnPodError as e:
        collect_pod_information(pod)
        raise AssertionError(
            f"Failed to execute 'jupyter labextension list' on pod '{pod.name}' after upgrade. "
            f"Cannot verify Elyra extensions preservation. Error: {e}"
        ) from e

    current_extensions = parse_elyra_extensions(labextension_output=output)

    # Check for removed extensions
    missing = set(baseline.elyra_extensions.keys()) - set(current_extensions.keys())
    assert not missing, (
        f"Elyra extensions removed during upgrade: {', '.join(sorted(missing))}. "
        f"Pre: {sorted(baseline.elyra_extensions.keys())}, "
        f"Post: {sorted(current_extensions.keys())}"
    )

    # Check for status degradation
    status_changes = []
    for name, baseline_meta in baseline.elyra_extensions.items():
        current_meta = current_extensions[name]
        if baseline_meta["enabled"] != current_meta["enabled"] or baseline_meta["status"] != current_meta["status"]:
            status_changes.append(
                f"{name}: enabled {baseline_meta['enabled']}→{current_meta['enabled']}, "
                f"status {baseline_meta['status']}→{current_meta['status']}"
            )

    assert not status_changes, "Elyra extensions changed status during upgrade:\n  " + "\n  ".join(status_changes)


def verify_post_upgrade_elyra_runtime_configs_preserved(
    pod: Pod,
    notebook: Notebook,
    baseline: WorkbenchImageBaseline,
) -> None:
    """Verify Elyra runtime configs survived upgrade unchanged.

    If baseline.elyra_extensions is None, the test fails.
    Runtime configs may be empty (Elyra installed but no pipelines configured).
    """
    assert baseline.elyra_extensions is not None, (
        "No Elyra extensions in baseline. This test requires Elyra to have been present pre-upgrade."
    )

    # Allow empty configs (Elyra installed but no pipelines configured)
    if not baseline.runtime_configs:
        current_files = list_runtime_configs(pod=pod, container=notebook.name)
        if current_files:
            LOGGER.info(f"{len(current_files)} runtime config(s) added during upgrade (allowed)")
        return

    current_files = list_runtime_configs(pod=pod, container=notebook.name)
    current_filenames = set(current_files)
    baseline_filenames = set(baseline.runtime_configs.keys())

    # Check for deleted configs
    missing_files = baseline_filenames - current_filenames
    assert not missing_files, f"Runtime config files deleted during upgrade: {', '.join(sorted(missing_files))}"

    # Log additions (allowed during migration)
    added_files = current_filenames - baseline_filenames
    if added_files:
        LOGGER.info(f"Runtime config files added during upgrade (allowed): {', '.join(sorted(added_files))}")

    # Compare each baseline config semantically
    differences = []
    for filename, baseline_config in baseline.runtime_configs.items():
        current_config = read_runtime_config(pod=pod, container=notebook.name, filename=filename)
        diff = compare_runtime_config_semantics(
            baseline=baseline_config,
            current=current_config,
            filename=filename,
        )
        if diff:
            differences.extend(diff)

    assert not differences, "Runtime configs modified during upgrade:\n  " + "\n  ".join(differences)
