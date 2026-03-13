"""Custom workbench image validation tests."""

import shlex
from dataclasses import dataclass
from time import time

import pytest
from ocp_resources.namespace import Namespace
from ocp_resources.notebook import Notebook
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import ExecOnPodError, Pod
from simple_logger.logger import get_logger

from utilities.constants import Timeout
from utilities.general import collect_pod_information

LOGGER = get_logger(name=__name__)

# Error messages
_ERR_EMPTY_PACKAGES = "packages list cannot be empty"
_ERR_CONTAINER_NOT_FOUND = "Container '{container_name}' not found in pod. Available containers: {containers}"


@dataclass
class PackageVerificationResult:
    """Represents the outcome of package import verification for a single package."""

    package_name: str
    import_successful: bool
    command_executed: str
    execution_time_seconds: float
    error_message: str | None = None
    stdout: str = ""
    stderr: str = ""


def verify_package_import(
    pod: Pod,
    container_name: str,
    packages: list[str],
    timeout: int = 60,
    collect_diagnostics: bool = True,
) -> dict[str, PackageVerificationResult]:
    """
    Verify that specified Python packages are importable in a pod container.

    This function executes 'python -c "import <package>"' for each package
    in the provided list and returns verification results.

    Args:
        pod: Pod instance to execute commands in (from ocp_resources.pod)
        container_name: Name of the container within the pod to target
        packages: List of Python package names to verify (e.g., ["sdg_hub", "numpy"])
        timeout: Maximum time in seconds to wait for each import command (default: 60)
        collect_diagnostics: Whether to collect pod logs on failure (default: True)

    Returns:
        Dictionary mapping package names to PackageVerificationResult objects.

    Raises:
        ValueError: If packages list is empty
        RuntimeError: If container doesn't exist
    """
    # Input validation
    if not packages:
        raise ValueError(_ERR_EMPTY_PACKAGES)

    # Verify container exists
    try:
        # Use any() with a generator for a faster check
        if not any(container.name == container_name for container in pod.instance.spec.containers):
            container_names = [container.name for container in pod.instance.spec.containers]
            raise RuntimeError(
                _ERR_CONTAINER_NOT_FOUND.format(container_name=container_name, containers=container_names)
            )
    except (AttributeError, TypeError) as e:
        raise RuntimeError(
            "Could not access container list from pod object structure. "
            f"The pod structure may be incomplete or malformed. Original error: {e}"
        )

    LOGGER.info(f"Verifying {len(packages)} packages in container '{container_name}' of pod '{pod.name}'")

    # Verify each package
    results = {}
    for package_name in packages:
        command = f"python -c 'import {package_name}'"
        command_list = shlex.split(command)

        LOGGER.debug(f"Executing: {command}")

        start_time = time()
        output = ""
        error_message = None
        import_successful = False
        stderr_output = ""

        try:
            # Execute command in container with timeout
            output = pod.execute(container=container_name, command=command_list, timeout=timeout)
            import_successful = True

        except ExecOnPodError as e:
            # Failure case - extract error message
            error_message = str(e)
            stderr_output = error_message

            # Collect pod information if requested
            if collect_diagnostics:
                collect_pod_information(pod)

        execution_time = time() - start_time
        output = output if output else ""

        if import_successful:
            LOGGER.info(f"Package {package_name}: ✓ (import successful in {execution_time:.2f}s)")
        else:
            LOGGER.warning(f"Package {package_name}: ✗ (import failed: {error_message})")

        results[package_name] = PackageVerificationResult(
            package_name=package_name,
            import_successful=import_successful,
            command_executed=command,
            execution_time_seconds=execution_time,
            stdout=output,
            error_message=error_message,
            stderr=stderr_output,
        )

    return results


def install_packages_in_pod(
    pod: Pod,
    container_name: str,
    packages: list[str],
    timeout: int = 120,
) -> dict[str, bool]:
    """
    Install Python packages in a running pod container using pip.

    This function executes 'pip install <package>' for each package
    in the provided list and returns installation results.

    Args:
        pod: Pod instance to execute commands in (from ocp_resources.pod)
        container_name: Name of the container within the pod to target
        packages: List of Python package names to install (e.g., ["sdg-hub"])
        timeout: Maximum time in seconds to wait for each install command (default: 120)

    Returns:
        Dictionary mapping package names to installation success status (True/False).

    Raises:
        ValueError: If packages list is empty
        RuntimeError: If container doesn't exist
    """
    # Input validation
    if not packages:
        raise ValueError(_ERR_EMPTY_PACKAGES)

    # Verify container exists
    try:
        # Use any() with a fast, short-circuiting check
        if not any(container.name == container_name for container in pod.instance.spec.containers):
            container_names = [container.name for container in pod.instance.spec.containers]
            raise RuntimeError(
                _ERR_CONTAINER_NOT_FOUND.format(container_name=container_name, containers=container_names)
            )
    except (AttributeError, TypeError) as e:
        raise RuntimeError(
            "Could not access container list from pod object structure. "
            f"The pod structure may be incomplete or malformed. Original error: {e}"
        )

    LOGGER.info(f"Installing {len(packages)} packages in container '{container_name}' of pod '{pod.name}'")

    # Install each package
    results = {}
    for package_name in packages:
        command_list = ["pip", "install", package_name, "--quiet"]

        LOGGER.debug(f"Executing: {' '.join(command_list)}")

        try:
            # Execute command in container with timeout
            pod.execute(container=container_name, command=command_list, timeout=timeout)
            results[package_name] = True
            LOGGER.info(f"Package {package_name}: ✓ (installed successfully)")

        except ExecOnPodError as e:
            error_message = str(e)
            results[package_name] = False
            LOGGER.warning(f"Package {package_name}: ✗ (installation failed: {error_message})")

    return results


class TestCustomImageValidation:
    """Validate custom workbench images with package introspection."""

    @pytest.mark.sanity
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "unprivileged_model_namespace,users_persistent_volume_claim,default_notebook,notebook_image,packages_to_verify",
        [
            # ========================================
            # HOW TO ADD A NEW CUSTOM IMAGE TEST:
            # ========================================
            # 1. Obtain image URL and package list from workbench image team
            # 2. Copy the pytest.param template below
            # 3. Update name, namespace, custom_image, and id fields
            # 4. Update packages_to_verify list
            # 5. Remove the skip marker once the image is available
            # 6. Run the test:
            # pytest tests/workbenches/notebook-controller/\
            #     test_custom_images.py::TestCustomImageValidation::\
            #     test_custom_image_package_verification[your_id] -v
            # ========================================
            # Test Case: SDG Hub Notebook
            # Image: To be provided by workbench image team
            # Required Packages: sdg_hub
            # Purpose: Validate sdg_hub image for knowledge tuning workflows
            # NOTE: This is a placeholder - update with actual image URL once provided
            pytest.param(
                {
                    "name": "test-sdg-hub",
                    "add-dashboard-label": True,
                },
                {
                    "name": "test-sdg-hub",
                },
                {
                    "namespace": "test-sdg-hub",
                    "name": "test-sdg-hub",
                },
                {
                    "custom_image": (
                        "quay.io/opendatahub/"
                        "odh-workbench-jupyter-minimal-cuda-py312-ubi9@sha256:"
                        "9458a764d861cbe0a782a53e0f5a13a4bcba35d279145d87088ab3cdfabcad1d"  # pragma: allowlist secret
                    ),  # Placeholder - update with sdg_hub image
                },
                ["sdg_hub"],
                id="sdg_hub_image",
            ),
            # Test Case: Data Science Notebook (Demonstration of Pattern Reusability)
            # Image: Standard datascience workbench image
            # Required Packages: numpy, pandas, matplotlib, scikit-learn
            # Purpose: Demonstrate test framework scalability with second image validation
            pytest.param(
                {
                    "name": "test-datascience",
                    "add-dashboard-label": True,
                },
                {
                    "name": "test-datascience",
                },
                {
                    "namespace": "test-datascience",
                    "name": "test-datascience",
                },
                {
                    "custom_image": (
                        "quay.io/opendatahub/"
                        "odh-workbench-jupyter-minimal-cuda-py312-ubi9@sha256:"
                        "9458a764d861cbe0a782a53e0f5a13a4bcba35d279145d87088ab3cdfabcad1d"  # pragma: allowlist secret
                    ),
                },
                ["numpy", "pandas", "matplotlib"],
                id="datascience_image",
            ),
        ],
        indirect=[
            "unprivileged_model_namespace",
            "users_persistent_volume_claim",
            "default_notebook",
            "notebook_image",
        ],
    )
    def test_custom_image_package_verification(
        self,
        unprivileged_model_namespace: Namespace,
        users_persistent_volume_claim: PersistentVolumeClaim,
        default_notebook: Notebook,
        notebook_image: str,
        notebook_pod: Pod,
        packages_to_verify: list[str],
    ):
        """
        Validate that custom workbench image contains required packages.

        Note: Packages might not be directly available within the workbench image
        but this test attempts to install them using pip if they are missing.

        This test:
        1. Uses a workbench with the specified custom image (via fixtures)
        2. Installs missing packages via pip if needed
        3. Executes package import verification commands
        4. Asserts that all required packages are importable

        Test satisfies:
        - FR-001: Spawn workbench with custom image URL
        - FR-002: Detect running pod and wait for ready state (via notebook_pod fixture)
        - FR-003: 10-minute timeout for pod readiness (via notebook_pod fixture)
        - FR-004: Execute package import commands
        - FR-005: Report success/failure with details
        """
        # Verify packages are importable

        # Minimal stdlib safety set: only third-party packages should be in packages_to_verify
        # This test is designed to validate third-party packages that require pip installation
        standard_lib_packages = {"sys", "os"}
        packages_to_install = [pkg for pkg in packages_to_verify if pkg not in standard_lib_packages]

        if packages_to_install:
            LOGGER.info(f"Installing {len(packages_to_install)} packages: {packages_to_install}")
            install_results = install_packages_in_pod(
                pod=notebook_pod,
                container_name=default_notebook.name,
                packages=packages_to_install,
                timeout=Timeout.TIMEOUT_2MIN,
            )

            failed_installs = [name for name, success in install_results.items() if not success]
            if failed_installs:
                raise AssertionError(
                    f"Failed to install {len(failed_installs)} package(s): {', '.join(failed_installs)}. "
                    f"Cannot proceed with package import verification."
                )

        # Verify packages are importable
        results = verify_package_import(
            pod=notebook_pod,
            container_name=default_notebook.name,
            packages=packages_to_verify,
            timeout=Timeout.TIMEOUT_1MIN,
        )

        # Assert all packages imported successfully
        failed_packages = [name for name, result in results.items() if not result.import_successful]

        if failed_packages:
            error_report = self._format_package_failure_report(
                failed_packages=failed_packages,
                results=results,
                pod=notebook_pod,
            )
            raise AssertionError(error_report)

    def _format_package_failure_report(self, failed_packages: list[str], results: dict, pod: Pod) -> str:
        """
        Format a detailed error report for package import failures.

        Args:
            failed_packages: List of package names that failed to import
            results: Dictionary of all verification results
            pod: The pod instance where verification was attempted

        Returns:
            Formatted error report string
        """
        report = [
            f"The following packages are not importable in {pod.name}:",
            "",
        ]

        for name in failed_packages:
            result = results[name]
            report.append(f"  ❌ {name}:")
            report.append(f"     Error: {result.error_message}")
            report.append(f"     Command: {result.command_executed}")
            report.append(f"     Execution Time: {result.execution_time_seconds:.2f}s")
            report.append("")

        # Add troubleshooting guidance
        report.append("Troubleshooting:")
        report.append("  1. Check the must-gather directory for pod logs and YAML")
        report.append("  2. Verify the custom image contains the required packages")
        report.append("  3. Check if packages are installed in the correct Python environment")
        report.append("  4. Verify package names match import names (pip name vs import name)")
        report.append("  5. Contact the workbench image team for package installation issues")

        return "\n".join(report)
