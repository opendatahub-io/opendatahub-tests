"""
Pytest fixtures for MLServer model car (OCI ImageVolume) tests.

MLServer modelcar uses Kubernetes ImageVolume feature which requires:
- OCP 4.20+ (enables ImageVolume by default)
- OCP 4.19 has K8s 1.32 with ImageVolume as alpha (feature-gated, not enabled by default)

These tests are automatically skipped on OCP < 4.20.
"""

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from semver import Version

LOGGER = structlog.get_logger(name=__name__)

# MLServer modelcar requires OCP 4.20+ for ImageVolume support
# OCP 4.19 ended Full Support phase (final maintenance support ends 2026-12-17)
MLSERVER_MODELCAR_MIN_OCP_VERSION = Version.parse("4.20.0")


@pytest.fixture(scope="session", autouse=True)
def skip_mlserver_modelcar_if_unsupported_ocp_version(
    admin_client: DynamicClient,
    openshift_version: Version,
) -> None:
    """Skip MLServer modelcar tests if OCP version < 4.20 (ImageVolume not supported)."""
    if openshift_version < MLSERVER_MODELCAR_MIN_OCP_VERSION:
        message = (
            f"Skipping MLServer modelcar tests: ImageVolume feature requires OCP 4.20+. "
            f"Current cluster: OCP {openshift_version}, minimum required: {MLSERVER_MODELCAR_MIN_OCP_VERSION}. "
            f"OCP 4.19 has ImageVolume as alpha (feature-gated), not enabled by default."
        )
        LOGGER.info(message)
        pytest.skip(reason=message)
