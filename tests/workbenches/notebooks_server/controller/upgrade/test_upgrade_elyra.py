"""Elyra extension upgrade tests.

Verifies that Elyra JupyterLab extensions and runtime configurations
survive platform upgrades without modification.

Notes on Elyra runtime configurations:
- Elyra creates a runtime config directory at /opt/app-root/src/.local/share/jupyter/metadata/runtimes/
- Runtime configs may or may not exist depending on user setup
- Pre-upgrade tests WARN if no runtime configs exist
  - Elyra will not function correctly, but this is technically allowable as a migration state
- Post-upgrade tests verify existing configs are preserved:
  - Existing configs must NOT be deleted or modified
  - New configs MAY be added (e.g., auto-created odh_dsp.json)
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
    """Verify Elyra extensions and runtime configs before upgrade.

    Allows workbenches without Elyra, so if Elyra is not installed, tests are skipped.
    """

    @pytest.mark.pre_upgrade
    def test_elyra_extensions_installed_before_upgrade(
        self,
        upgrade_notebook: Notebook,
        upgrade_notebook_pod: Pod,
    ) -> None:
        """Verify Elyra extensions are installed and healthy before upgrade."""
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

        elyra_extensions = parse_elyra_extensions(labextension_output=output)

        if not elyra_extensions:
            pytest.skip("No Elyra extensions found - Elyra is not installed in this workbench")

        # Verify all Elyra extensions are healthy (enabled + OK status)
        unhealthy = []
        for name, metadata in elyra_extensions.items():
            if metadata["enabled"] and metadata["status"] == "OK":
                LOGGER.info(f"  ✓ {name} v{metadata['version']} - enabled, {metadata['status']}")
            else:
                LOGGER.error(
                    f"  ✗ {name} v{metadata['version']} - enabled={metadata['enabled']}, status={metadata['status']}"
                )
                unhealthy.append(name)

        assert not unhealthy, (
            f"Found {len(unhealthy)} unhealthy Elyra extension(s): {', '.join(unhealthy)}. "
            f"All Elyra extensions must be enabled with OK status before running upgrade tests. "
            f"Fix the broken extensions before proceeding."
        )

    @pytest.mark.pre_upgrade
    def test_elyra_runtime_configs_exist_before_upgrade(
        self,
        upgrade_notebook: Notebook,
        upgrade_notebook_pod: Pod,
    ) -> None:
        """Validate Elyra runtime configs before upgrade.

        Warns if no configs exist (Elyra will not function correctly, but is a technically allowable state).
        Validates JSON structure of any configs found.
        """
        runtime_files = list_runtime_configs(pod=upgrade_notebook_pod, container=upgrade_notebook.name)

        if not runtime_files:
            LOGGER.warning(
                f"No Elyra runtime configs found in '{ELYRA_RUNTIMES_DIR}'. "
                f"Elyra will not function correctly without runtime configurations."
            )
            return

        LOGGER.info(f"Found {len(runtime_files)} Elyra runtime config(s): {', '.join(runtime_files)}")

        for filename in runtime_files:
            runtime_config = read_runtime_config(
                pod=upgrade_notebook_pod, container=upgrade_notebook.name, filename=filename
            )

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
    """Verify Elyra extensions and runtime configs preserved after upgrade.

    Allows workbenches without Elyra, so if Elyra was not installed pre-upgrade, tests are skipped.
    """

    @pytest.mark.post_upgrade
    def test_elyra_extensions_still_installed_after_upgrade(
        self,
        upgrade_notebook: Notebook,
        upgrade_notebook_pod: Pod,
        upgrade_notebook_baseline: dict[str, Any],
    ) -> None:
        """Verify baseline Elyra extensions preserved after upgrade.

        Allows additions, prevents removals or status degradation.
        """
        baseline_extensions = upgrade_notebook_baseline.get("elyra_extensions")

        # Skip if Elyra wasn't installed pre-upgrade
        if baseline_extensions is None:
            pytest.skip("No Elyra extensions in baseline - Elyra was not installed pre-upgrade")

        try:
            output = upgrade_notebook_pod.execute(
                container=upgrade_notebook.name,
                command=["jupyter", "labextension", "list"],
                timeout=Timeout.TIMEOUT_1MIN,
            )
        except ExecOnPodError as e:
            collect_pod_information(upgrade_notebook_pod)
            raise AssertionError(
                f"Failed to execute 'jupyter labextension list' on pod '{upgrade_notebook_pod.name}' after upgrade: {e}"
            ) from e

        current_extensions = parse_elyra_extensions(labextension_output=output)

        # Check for removed extensions
        missing = set(baseline_extensions.keys()) - set(current_extensions.keys())
        assert not missing, (
            f"The following Elyra extensions were removed during upgrade: {', '.join(sorted(missing))}. "
            f"Pre-upgrade: {sorted(baseline_extensions.keys())}, "
            f"post-upgrade: {sorted(current_extensions.keys())}"
        )

        # Check for status changes (enabled/status degradation)
        status_changes = []
        for name, baseline_meta in baseline_extensions.items():
            current_meta = current_extensions[name]

            # Compare enabled and status fields
            if baseline_meta["enabled"] != current_meta["enabled"] or baseline_meta["status"] != current_meta["status"]:
                status_changes.append(
                    f"{name}: enabled {baseline_meta['enabled']}→{current_meta['enabled']}, "
                    f"status {baseline_meta['status']}→{current_meta['status']}"
                )

        assert not status_changes, "The following Elyra extensions changed status during upgrade:\n  " + "\n  ".join(
            status_changes
        )

        # Log new extensions (informational)
        added = set(current_extensions.keys()) - set(baseline_extensions.keys())
        if added:
            LOGGER.info(f"New Elyra extensions added during upgrade: {', '.join(sorted(added))}")

        LOGGER.info(
            f"Elyra extensions verified: {len(baseline_extensions)} baseline extensions preserved "
            f"({len(added)} new extensions added)"
        )

    @pytest.mark.post_upgrade
    def test_elyra_runtime_configs_unchanged_after_upgrade(
        self,
        upgrade_notebook: Notebook,
        upgrade_notebook_pod: Pod,
        upgrade_notebook_baseline: dict[str, Any],
    ) -> None:
        """Verify baseline runtime configs preserved after upgrade.

        Allows additions, prevents deletions or modifications.
        Compares: display_name, schema_name, metadata.runtime_type, metadata.api_endpoint
        """
        baseline_extensions = upgrade_notebook_baseline.get("elyra_extensions")
        baseline_configs = upgrade_notebook_baseline.get("runtime_configs")

        # Skip if Elyra wasn't installed pre-upgrade
        if baseline_extensions is None:
            pytest.skip("No Elyra extensions in baseline - Elyra was not installed pre-upgrade")

        # If baseline is {} (empty dict), it means Elyra was installed but had no runtime configs
        if not baseline_configs:
            current_files = list_runtime_configs(pod=upgrade_notebook_pod, container=upgrade_notebook.name)
            if current_files:
                LOGGER.info(f"{len(current_files)} runtime config(s) added during upgrade: {', '.join(current_files)}")
            else:
                LOGGER.info("No runtime configs before or after upgrade.")
            return

        current_files = list_runtime_configs(pod=upgrade_notebook_pod, container=upgrade_notebook.name)
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
            LOGGER.info(f"New runtime config(s) added: {', '.join(sorted(added_files))}")

        LOGGER.info(f"Verifying {len(baseline_filenames)} baseline config(s) semantically unchanged...")

        all_differences = []
        for filename, baseline_config in baseline_configs.items():
            current_config = read_runtime_config(
                pod=upgrade_notebook_pod, container=upgrade_notebook.name, filename=filename
            )
            differences = compare_runtime_config_semantics(
                baseline=baseline_config, current=current_config, filename=filename
            )

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
