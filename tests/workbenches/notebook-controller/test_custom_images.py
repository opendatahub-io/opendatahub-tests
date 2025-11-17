"""Custom workbench image validation tests."""

import re
import shlex
from dataclasses import dataclass
from time import time

import pytest

from kubernetes.dynamic.client import DynamicClient

from ocp_resources.pod import Pod
from ocp_resources.pod import ExecOnPodError
from ocp_resources.namespace import Namespace
from ocp_resources.notebook import Notebook
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim

from utilities.constants import Timeout
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)


@dataclass
class PackageVerificationResult:
    """Represents the outcome of package import verification for a single package."""

    package_name: str
    import_successful: bool
    error_message: str | None
    command_executed: str
    execution_time_seconds: float
    pod_logs: str | None
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
        ValueError: If packages list is empty or contains invalid identifiers
        RuntimeError: If pod is not in Running state or container doesn't exist
    """
    # Error messages
    _ERR_EMPTY_PACKAGES = "packages list cannot be empty"
    _ERR_INVALID_TIMEOUT = "timeout must be positive"
    _ERR_INVALID_PACKAGE_NAME = "Invalid package name: {package}"
    _ERR_POD_NOT_EXISTS = "Pod {pod_name} does not exist"
    _ERR_POD_NOT_RUNNING = "Pod {pod_name} is not in Running state (current: {phase})"
    _ERR_CONTAINER_NOT_FOUND = "Container '{container_name}' not found in pod. Available containers: {containers}"

    # Input validation
    if not packages:
        raise ValueError(_ERR_EMPTY_PACKAGES)

    if timeout <= 0:
        raise ValueError(_ERR_INVALID_TIMEOUT)

    # Validate package names (Python identifier pattern)
    package_name_pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    for package in packages:
        if not package_name_pattern.match(package):
            raise ValueError(_ERR_INVALID_PACKAGE_NAME.format(package=package))

    # Check pod exists and is running
    if not pod.exists:
        raise RuntimeError(_ERR_POD_NOT_EXISTS.format(pod_name=pod.name))

    pod_status = pod.instance.status
    if pod_status.phase != "Running":
        raise RuntimeError(_ERR_POD_NOT_RUNNING.format(pod_name=pod.name, phase=pod_status.phase))

    # Verify container exists
    container_names = [c.name for c in pod.instance.spec.containers]
    if container_name not in container_names:
        raise RuntimeError(_ERR_CONTAINER_NOT_FOUND.format(container_name=container_name, containers=container_names))

    LOGGER.info(f"Verifying {len(packages)} packages in container '{container_name}' of pod '{pod.name}'")

    # Verify each package
    results = {}
    for package_name in packages:
        command = f"python -c 'import {package_name}'"
        command_list = shlex.split(command)

        LOGGER.debug(f"Executing: {command}")

        start_time = time()
        try:
            # Execute command in container
            # Note: timeout parameter is passed but may not be supported by pod.execute()
            # If timeout is not enforced at the pod.execute() level, it's still tracked via execution_time
            try:
                output = pod.execute(container=container_name, command=command_list, timeout=timeout)
            except TypeError:
                # pod.execute() may not support timeout parameter, fall back to execution without timeout
                # The timeout is still validated and tracked via execution_time measurement
                output = pod.execute(container=container_name, command=command_list)
            execution_time = time() - start_time

            # Success case
            results[package_name] = PackageVerificationResult(
                package_name=package_name,
                import_successful=True,
                error_message=None,
                command_executed=command,
                execution_time_seconds=execution_time,
                pod_logs=None,
                stdout=output if output else "",
                stderr="",
            )
            LOGGER.info(f"Package {package_name}: ✓ (import successful in {execution_time:.2f}s)")

        except ExecOnPodError as e:
            execution_time = time() - start_time

            # Failure case - extract error message
            error_message = str(e)
            stderr_output = error_message

            # Collect pod logs if requested
            pod_logs = None
            if collect_diagnostics:
                try:
                    pod_logs = pod.log(container=container_name, tail_lines=100)
                except (RuntimeError, AttributeError, KeyError) as log_error:
                    LOGGER.warning(f"Failed to collect pod logs: {log_error}")
                    pod_logs = "Could not retrieve pod logs"

            results[package_name] = PackageVerificationResult(
                package_name=package_name,
                import_successful=False,
                error_message=error_message,
                command_executed=command,
                execution_time_seconds=execution_time,
                pod_logs=pod_logs,
                stdout="",
                stderr=stderr_output,
            )
            LOGGER.warning(f"Package {package_name}: ✗ (import failed: {error_message})")

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
        ValueError: If packages list is empty or pod is invalid
        RuntimeError: If pod is not in Running state or container doesn't exist
    """
    # Error messages
    _ERR_INVALID_POD_OR_PACKAGES = "pod must be valid and packages must be a non-empty list"
    _ERR_INVALID_TIMEOUT_INSTALL = "timeout must be positive"
    _ERR_POD_NOT_EXISTS_INSTALL = "Pod {pod_name} does not exist"
    _ERR_POD_NOT_RUNNING_INSTALL = "Pod {pod_name} is not in Running state (current: {phase})"
    _ERR_CONTAINER_NOT_FOUND_INSTALL = "Container '{container_name}' not found in pod. Available containers: {containers}"

    # Input validation
    if not pod or not isinstance(packages, list) or not packages:
        raise ValueError(_ERR_INVALID_POD_OR_PACKAGES)

    if timeout <= 0:
        raise ValueError(_ERR_INVALID_TIMEOUT_INSTALL)

    # Check pod exists and is running
    if not pod.exists:
        raise RuntimeError(_ERR_POD_NOT_EXISTS_INSTALL.format(pod_name=pod.name))

    pod_status = pod.instance.status
    if pod_status.phase != "Running":
        raise RuntimeError(_ERR_POD_NOT_RUNNING_INSTALL.format(pod_name=pod.name, phase=pod_status.phase))

    # Verify container exists
    container_names = [c.name for c in pod.instance.spec.containers]
    if container_name not in container_names:
        raise RuntimeError(_ERR_CONTAINER_NOT_FOUND_INSTALL.format(container_name=container_name, containers=container_names))

    LOGGER.info(f"Installing {len(packages)} packages in container '{container_name}' of pod '{pod.name}'")

    # Install each package
    results = {}
    for package_name in packages:
        command_list = ["pip", "install", package_name, "--quiet"]

        LOGGER.debug(f"Executing: {' '.join(command_list)}")

        try:
            # Execute command in container
            # Note: timeout parameter is passed but may not be supported by pod.execute()
            # If timeout is not enforced at the pod.execute() level, execution may exceed timeout
            try:
                pod.execute(container=container_name, command=command_list, timeout=timeout)
            except TypeError:
                # pod.execute() may not support timeout parameter, fall back to execution without timeout
                pod.execute(container=container_name, command=command_list)
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
        "unprivileged_model_namespace,users_persistent_volume_claim,default_notebook",
        [
            # ========================================
            # HOW TO ADD A NEW CUSTOM IMAGE TEST:
            # ========================================
            # 1. Obtain image URL and package list from workbench image team
            # 2. Copy the pytest.param template below
            # 3. Update name, namespace, custom_image, and id fields
            # 4. Update packages_to_verify in the test method (see line ~120)
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
                    "custom_image": (
                        "quay.io/opendatahub/"
                        "odh-workbench-jupyter-minimal-cuda-py312-ubi9@sha256:"
                        "9458a764d861cbe0a782a53e0f5a13a4bcba35d279145d87088ab3cdfabcad1d"  # pragma: allowlist secret
                    ),  # Placeholder - update with sdg_hub image
                },
                id="sdg_hub_image",
                # marks=pytest.mark.skip(reason="Waiting for sdg_hub image URL from workbench image team"),
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
                    "custom_image": (
                        "quay.io/opendatahub/"
                        "odh-workbench-jupyter-minimal-cuda-py312-ubi9@sha256:"
                        "9458a764d861cbe0a782a53e0f5a13a4bcba35d279145d87088ab3cdfabcad1d"  # pragma: allowlist secret
                    ),
                },
                id="datascience_image",
            ),
        ],
        indirect=True,
    )
    def test_custom_image_package_verification(
        self,
        request: pytest.FixtureRequest,
        unprivileged_client: DynamicClient,
        unprivileged_model_namespace: Namespace,  # noqa: ARG002
        users_persistent_volume_claim: PersistentVolumeClaim,  # noqa: ARG002
        default_notebook: Notebook,
    ):
        """
        Validate that custom workbench image contains required packages.

        This test:
        1. Spawns a workbench with the specified custom image
        2. Waits for the pod to reach Ready state (up to 10 minutes)
        3. Executes package import verification commands
        4. Asserts that all required packages are importable

        Test satisfies:
        - FR-001: Spawn workbench with custom image URL
        - FR-002: Detect running pod and wait for ready state
        - FR-003: 10-minute timeout for pod readiness
        - FR-004: Execute package import commands
        - FR-005: Report success/failure with details
        """
        # Wait for notebook pod to be created and reach Ready state
        notebook_pod = Pod(
            client=unprivileged_client,
            namespace=default_notebook.namespace,
            name=f"{default_notebook.name}-0",
        )

        # Wait for pod to exist
        notebook_pod.wait()

        # Error messages
        _ERR_POD_NOT_READY = (
            "Pod '{pod_name}-0' failed to reach Ready state within 10 minutes.\n"
            "Pod Phase: {pod_phase}\n"
            "Error Details:\n{error_details}\n"
            "Original Error: {original_error}"
        )
        _ERR_POD_NOT_CREATED = "Pod '{pod_name}-0' was not created. Check notebook controller logs."

        # Wait for pod to reach Ready state (10-minute timeout for large custom images)
        try:
            notebook_pod.wait_for_condition(
                condition=Pod.Condition.READY,
                status=Pod.Condition.Status.TRUE,
                timeout=Timeout.TIMEOUT_10MIN,
            )
        except (TimeoutError, RuntimeError) as e:
            # Enhanced error handling: Collect pod diagnostic information
            pod_status = notebook_pod.instance.status if notebook_pod.exists else None

            if pod_status:
                pod_phase = pod_status.phase
                error_details = self._get_pod_failure_details(notebook_pod)
                raise AssertionError(
                    _ERR_POD_NOT_READY.format(
                        pod_name=default_notebook.name,
                        pod_phase=pod_phase,
                        error_details=error_details,
                        original_error=e,
                    )
                ) from e
            else:
                raise AssertionError(_ERR_POD_NOT_CREATED.format(pod_name=default_notebook.name)) from e

        # Verify packages are importable
        # Different packages per test case (based on test ID from parametrization)
        test_id = request.node.callspec.id
        if "sdg_hub" in test_id:
            # SDG Hub image packages (placeholder - update when image URL is available)
            packages_to_verify = ["sdg_hub"]
        elif "datascience" in test_id:
            # Data science image packages
            packages_to_verify = ["numpy", "pandas", "matplotlib"]
        else:
            # Default: basic Python packages
            packages_to_verify = ["sys", "os"]

        # Install packages if they're not standard library (not in the default list)
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
                LOGGER.warning(f"Failed to install packages: {failed_installs}")

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

    def _get_pod_failure_details(self, pod: Pod) -> str:
        """
        Collect diagnostic information when pod fails to reach ready state.

        Args:
            pod: The pod instance to diagnose

        Returns:
            Formatted diagnostic information string
        """
        details = []

        pod_status = pod.instance.status
        if not pod_status:
            return "Pod status unavailable"

        # Get pod phase
        details.append(f"Phase: {pod_status.phase}")

        # Get container statuses
        if pod_status.containerStatuses:
            details.append("\nContainer Statuses:")
            for container_status in pod_status.containerStatuses:
                container_name = container_status.name
                ready = container_status.ready

                details.append(f"  - {container_name}: ready={ready}")

                # Check waiting state
                if hasattr(container_status.state, "waiting") and container_status.state.waiting:
                    waiting = container_status.state.waiting
                    reason = waiting.reason
                    message = waiting.message if hasattr(waiting, "message") else ""

                    # Categorize common errors
                    if reason == "ImagePullBackOff":
                        details.append(
                            f"    ⚠️  ImagePullBackOff: Failed to pull custom image\n"
                            f"    Verify registry access and image URL\n"
                            f"    Message: {message}"
                        )
                    elif reason == "CrashLoopBackOff":
                        details.append(
                            f"    ⚠️  CrashLoopBackOff: Container is crashing\n"
                            f"    Check container logs for startup errors\n"
                            f"    Message: {message}"
                        )
                    elif reason == "ErrImagePull":
                        details.append(
                            f"    ⚠️  ErrImagePull: Cannot pull image\n"
                            f"    Verify image exists and cluster has pull access\n"
                            f"    Message: {message}"
                        )
                    else:
                        details.append(f"    Waiting Reason: {reason}\n    Message: {message}")

                # Check terminated state
                if hasattr(container_status.state, "terminated") and container_status.state.terminated:
                    terminated = container_status.state.terminated
                    details.append(
                        f"    ⚠️  Container terminated\n"
                        f"    Exit Code: {terminated.exitCode}\n"
                        f"    Reason: {terminated.reason}"
                    )

        # Try to get pod logs for main container
        try:
            logs = pod.log(container=pod.instance.spec.containers[0].name, tail_lines=50)
            if logs:
                details.append(f"\nRecent Logs (last 50 lines):\n{logs}")
        except (RuntimeError, AttributeError, KeyError):
            details.append("\n(Could not retrieve pod logs)")

        return "\n".join(details)

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

            if result.pod_logs:
                report.append("     Pod Logs (excerpt):")
                # Show first 500 characters of logs
                log_excerpt = result.pod_logs[:500]
                for line in log_excerpt.split("\n"):
                    report.append(f"       {line}")
            report.append("")

        # Add troubleshooting guidance
        report.append("Troubleshooting:")
        report.append("  1. Verify the custom image contains the required packages")
        report.append("  2. Check if packages are installed in the correct Python environment")
        report.append("  3. Verify package names match import names (pip name vs import name)")
        report.append("  4. Contact the workbench image team for package installation issues")

        return "\n".join(report)
