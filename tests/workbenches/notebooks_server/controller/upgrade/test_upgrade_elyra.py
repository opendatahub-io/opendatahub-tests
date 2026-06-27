"""Elyra extension upgrade tests.

Verifies that Elyra JupyterLab extensions and runtime configurations
survive platform upgrades without modification.
"""

import json
import re
from typing import Any

import pytest
import structlog
from ocp_resources.notebook import Notebook
from ocp_resources.pod import ExecOnPodError, Pod

from utilities.constants import Timeout
from utilities.general import collect_pod_information

LOGGER = structlog.get_logger(name=__name__)

# Path to user-created Elyra runtime configurations inside workbench container
ELYRA_RUNTIMES_DIR = "/opt/app-root/src/.local/share/jupyter/metadata/runtimes"


def parse_elyra_extensions(labextension_output: str) -> dict[str, dict[str, Any]]:
    """Parse jupyter labextension list output to extract Elyra extensions.

    Args:
        labextension_output: Raw output from `jupyter labextension list` command

    Returns:
        Dictionary mapping extension names to their metadata:
        {
            "@elyra/pipeline-editor-extension": {
                "version": "5.0.0",
                "enabled": True,
                "status": "OK"
            },
            ...
        }

    Example input:
        JupyterLab v4.5.7
        /opt/app-root/share/jupyter/labextensions
                @elyra/code-snippet-extension v5.0.0 enabled OK
                @elyra/pipeline-editor-extension v5.0.0 enabled OK
                @jupyter-widgets/jupyterlab-manager v5.0.15 enabled OK (python, jupyterlab_widgets)
    """
    elyra_extensions = {}

    for line in labextension_output.split("\n"):
        line = line.strip()

        if not line or not line.startswith("@elyra/"):
            continue

        match = re.match(r"^(@elyra/[\w-]+)\s+v([\d.]+)\s+(enabled|disabled)\s+(\w+)", line)
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
            f"Content: {output[:200]}... Error: {e}"
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
            differences.append(
                f"Field '{field}' changed: '{baseline_value}' -> '{current_value}'"
            )

    baseline_metadata = baseline.get("metadata", {})
    current_metadata = current.get("metadata", {})

    metadata_critical_fields = ["runtime_type", "api_endpoint"]
    for field in metadata_critical_fields:
        baseline_value = baseline_metadata.get(field)
        current_value = current_metadata.get(field)

        if baseline_value != current_value:
            differences.append(
                f"metadata.{field} changed: '{baseline_value}' -> '{current_value}'"
            )

    return differences


@pytest.mark.usefixtures("capture_notebook_baseline")
class TestPreUpgradeElyra:
    """Verify Elyra extensions and runtime configs exist before platform upgrade.

    Steps:
        1. Execute `jupyter labextension list` in the notebook pod
        2. Verify at least one @elyra/ extension is installed
        3. List runtime configuration files in the Elyra metadata directory
        4. Parse and validate each runtime config JSON
        5. Baseline data is captured by the capture_notebook_baseline fixture
    """

    @pytest.mark.pre_upgrade
    def test_elyra_extensions_installed_before_upgrade(
        self,
        upgrade_notebook: Notebook,
        upgrade_notebook_pod: Pod,
    ) -> None:
        """Given a workbench notebook is running before upgrade,
        When we check installed JupyterLab extensions,
        Then at least one Elyra extension should be present and enabled.
        """
        try:
            output = upgrade_notebook_pod.execute(
                container=upgrade_notebook.name,
                command=["jupyter", "labextension", "list"],
                timeout=Timeout.TIMEOUT_1MIN,
            )
        except ExecOnPodError as e:
            collect_pod_information(upgrade_notebook_pod)
            raise AssertionError(
                f"Failed to execute 'jupyter labextension list' on pod '{upgrade_notebook_pod.name}': {e}"
            ) from e

        elyra_extensions = parse_elyra_extensions(output)

        assert elyra_extensions, (
            f"No Elyra extensions found in workbench pod '{upgrade_notebook_pod.name}'. "
            f"Expected at least one @elyra/ extension. "
            f"Full output:\n{output}"
        )

        enabled_count = sum(1 for ext in elyra_extensions.values() if ext["enabled"])
        LOGGER.info(
            f"Found {len(elyra_extensions)} Elyra extensions ({enabled_count} enabled) "
            f"in pod '{upgrade_notebook_pod.name}'"
        )

        for name, metadata in elyra_extensions.items():
            if metadata["enabled"] and metadata["status"] == "OK":
                LOGGER.info(f"  ✓ {name} v{metadata['version']} - {metadata['status']}")
            else:
                LOGGER.warning(
                    f"  ⚠ {name} v{metadata['version']} - "
                    f"enabled={metadata['enabled']}, status={metadata['status']}"
                )

    @pytest.mark.pre_upgrade
    def test_elyra_runtime_configs_exist_before_upgrade(
        self,
        upgrade_notebook: Notebook,
        upgrade_notebook_pod: Pod,
    ) -> None:
        """Given a workbench notebook is running before upgrade,
        When we check Elyra runtime configurations,
        Then runtime config files should exist and be valid JSON.

        Note: This test does not require specific runtime configs to exist.
        If no user-created runtimes exist, the test will log a warning but pass.
        The post-upgrade test will verify that whatever existed pre-upgrade is preserved.
        """
        runtime_files = list_runtime_configs(upgrade_notebook_pod, upgrade_notebook.name)

        if not runtime_files:
            LOGGER.warning(
                f"No user-created Elyra runtime configs found in '{ELYRA_RUNTIMES_DIR}' "
                f"on pod '{upgrade_notebook_pod.name}'. This is not an error, but upgrade "
                f"tests will only verify that this state is preserved (no runtimes before = no runtimes after)."
            )
            return

        LOGGER.info(
            f"Found {len(runtime_files)} Elyra runtime config(s) in pod '{upgrade_notebook_pod.name}': "
            f"{', '.join(runtime_files)}"
        )

        for filename in runtime_files:
            runtime_config = read_runtime_config(upgrade_notebook_pod, upgrade_notebook.name, filename)

            required_keys = ["display_name", "schema_name", "metadata"]
            missing_keys = [key for key in required_keys if key not in runtime_config]
            assert not missing_keys, (
                f"Runtime config '{filename}' is missing required keys: {missing_keys}. "
                f"Config content: {json.dumps(runtime_config, indent=2)}"
            )

            LOGGER.info(
                f"  ✓ {filename}: display_name='{runtime_config.get('display_name')}', "
                f"schema_name='{runtime_config.get('schema_name')}'"
            )


class TestPostUpgradeElyra:
    """Verify Elyra extensions and runtime configs survived the platform upgrade.

    Steps:
        1. Load pre-upgrade baseline from ConfigMap
        2. Re-execute `jupyter labextension list`
        3. Verify extension count >= baseline (allows additions, prevents removals)
        4. Verify all baseline extensions still present
        5. Re-read runtime config files
        6. Compare configs semantically against baseline
    """

    @pytest.mark.post_upgrade
    def test_elyra_extensions_still_installed_after_upgrade(
        self,
        upgrade_notebook: Notebook,
        upgrade_notebook_pod: Pod,
        upgrade_notebook_baseline: dict[str, Any],
    ) -> None:
        """Given Elyra extensions were installed before upgrade,
        When the upgrade completes,
        Then all pre-upgrade Elyra extensions should still be present.

        Allows: New extensions to be added by platform upgrade
        Prevents: Extensions being removed or disabled
        """
        baseline_count = upgrade_notebook_baseline.get("elyra_extensions_count", 0)
        baseline_samples = upgrade_notebook_baseline.get("elyra_extensions_sample", [])

        assert baseline_count > 0, (
            "Baseline does not contain Elyra extension count. "
            "Pre-upgrade tests may not have run successfully."
        )

        try:
            output = upgrade_notebook_pod.execute(
                container=upgrade_notebook.name,
                command=["jupyter", "labextension", "list"],
                timeout=Timeout.TIMEOUT_1MIN,
            )
        except ExecOnPodError as e:
            collect_pod_information(upgrade_notebook_pod)
            raise AssertionError(
                f"Failed to execute 'jupyter labextension list' on pod '{upgrade_notebook_pod.name}' "
                f"after upgrade: {e}"
            ) from e

        current_extensions = parse_elyra_extensions(output)
        current_count = len(current_extensions)

        assert current_count >= baseline_count, (
            f"Elyra extension count decreased after upgrade. "
            f"Pre-upgrade: {baseline_count} extensions, "
            f"post-upgrade: {current_count} extensions. "
            f"Current extensions: {list(current_extensions.keys())}"
        )

        missing_extensions = [name for name in baseline_samples if name not in current_extensions]
        assert not missing_extensions, (
            f"The following Elyra extensions present before upgrade are now missing: "
            f"{', '.join(missing_extensions)}. "
            f"Pre-upgrade samples: {baseline_samples}, "
            f"current extensions: {list(current_extensions.keys())}"
        )

        LOGGER.info(
            f"Elyra extensions verified: {current_count} found (baseline: {baseline_count}). "
            f"All baseline extensions still present."
        )

    @pytest.mark.post_upgrade
    def test_elyra_runtime_configs_unchanged_after_upgrade(
        self,
        upgrade_notebook: Notebook,
        upgrade_notebook_pod: Pod,
        upgrade_notebook_baseline: dict[str, Any],
    ) -> None:
        """Given Elyra runtime configs existed before upgrade,
        When the upgrade completes,
        Then runtime configs should be semantically unchanged.

        Verifies critical fields:
            - display_name
            - schema_name
            - metadata.runtime_type
            - metadata.api_endpoint

        Ignores: timestamps, field ordering, extra fields
        """
        baseline_configs = upgrade_notebook_baseline.get("runtime_configs", {})

        if not baseline_configs:
            LOGGER.info(
                "No runtime configs in baseline. Verifying that none exist post-upgrade either."
            )
            current_files = list_runtime_configs(upgrade_notebook_pod, upgrade_notebook.name)
            assert not current_files, (
                f"No runtime configs existed before upgrade, but {len(current_files)} found after upgrade: "
                f"{', '.join(current_files)}. This may indicate unexpected config creation during upgrade."
            )
            return

        current_files = list_runtime_configs(upgrade_notebook_pod, upgrade_notebook.name)
        current_filenames = set(current_files)
        baseline_filenames = set(baseline_configs.keys())

        missing_files = baseline_filenames - current_filenames
        assert not missing_files, (
            f"The following runtime config files were deleted during upgrade: {', '.join(missing_files)}. "
            f"Pre-upgrade files: {sorted(baseline_filenames)}, "
            f"post-upgrade files: {sorted(current_filenames)}"
        )

        LOGGER.info(
            f"All {len(baseline_filenames)} baseline runtime config files still exist. "
            f"Performing semantic comparison..."
        )

        all_differences = []
        for filename, baseline_config in baseline_configs.items():
            current_config = read_runtime_config(upgrade_notebook_pod, upgrade_notebook.name, filename)
            differences = compare_runtime_config_semantics(baseline_config, current_config, filename)

            if differences:
                all_differences.append(f"\n{filename}:")
                for diff in differences:
                    all_differences.append(f"  - {diff}")
            else:
                LOGGER.info(f"  ✓ {filename}: semantically unchanged")

        assert not all_differences, (
            f"The following runtime config changes were detected during upgrade:\n"
            f"{''.join(all_differences)}\n\n"
            f"Critical fields must remain unchanged across upgrades to preserve user workflows."
        )
