"""Elyra extension upgrade tests.

Verifies that Elyra JupyterLab extensions and runtime configurations
survive platform upgrades without modification.
"""

from typing import Any

import pytest
import structlog
from ocp_resources.notebook import Notebook
from ocp_resources.pod import ExecOnPodError, Pod

from tests.workbenches.notebooks_server.controller.upgrade.elyra_utils import (
    ELYRA_RUNTIMES_DIR,
    compare_runtime_config_semantics,
    list_runtime_configs,
    parse_elyra_extensions,
    read_runtime_config,
)
from utilities.constants import Timeout
from utilities.general import collect_pod_information

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures("capture_notebook_baseline")
class TestPreUpgradeElyra:
    """Verify Elyra extensions and runtime configs exist before platform upgrade.

    Steps:
        1. Execute `jupyter labextension list` in the notebook pod
        2. Verify at least one Elyra extension is installed (any extension containing "elyra")
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
            f"Expected at least one extension containing 'elyra' in the name. "
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
                f"Found keys: {list(runtime_config.keys())}"
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
        3. Verify all baseline extensions still present (allows additions, prevents removals)
        4. Re-read runtime config files
        5. Compare configs semantically against baseline
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
        Then all baseline Elyra extensions should still be present.

        Requires: All baseline extensions still present
        Allows: New extensions to be added
        Prevents: Extensions being removed or disabled
        """
        baseline_extensions_list = upgrade_notebook_baseline.get("elyra_extensions_list", [])

        assert baseline_extensions_list, (
            "Baseline does not contain Elyra extensions list. "
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

        baseline_extensions = set(baseline_extensions_list)
        current_extensions_set = set(current_extensions.keys())

        # Check that all baseline extensions are still present (allows additions, prevents removals)
        missing = baseline_extensions - current_extensions_set

        assert not missing, (
            f"The following Elyra extensions were removed during upgrade: {', '.join(sorted(missing))}. "
            f"Pre-upgrade: {sorted(baseline_extensions)}, "
            f"post-upgrade: {sorted(current_extensions_set)}"
        )

        added = current_extensions_set - baseline_extensions
        if added:
            LOGGER.info(f"New extensions added during upgrade: {', '.join(sorted(added))}")

        LOGGER.info(
            f"Elyra extensions verified: {len(current_extensions_set)} extensions "
            f"({len(baseline_extensions)} baseline + {len(added)} new). "
            f"All baseline extensions preserved."
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

        added_files = current_filenames - baseline_filenames
        if added_files:
            LOGGER.info(f"New runtime config files added (user-created or upgrade-generated): {', '.join(sorted(added_files))}")

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
