from typing import Any

import pytest
import structlog
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from semver import Version
from syrupy.extensions.json import JSONSnapshotExtension
from utilities.version import get_rhoai_version_from_env, should_skip_for_marker

from utilities.infra import get_product_version

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture
def response_snapshot(snapshot: Any) -> Any:
    return snapshot.use_extension(extension_class=JSONSnapshotExtension)


@pytest.fixture(scope="session")
def rhoai_version(admin_client: DynamicClient) -> Version | None:
    """Detect RHOAI product version from env var or cluster CSV."""
    env_version = get_rhoai_version_from_env()
    if env_version:
        LOGGER.info(f"RHOAI version from environment: {env_version}")
        return env_version

    try:
        cluster_version = get_product_version(admin_client=admin_client)
        LOGGER.info(f"RHOAI version from cluster CSV: {cluster_version}")
        return cluster_version
    except Exception:
        LOGGER.warning("Could not detect RHOAI version — version-based gating disabled")
        return None


@pytest.fixture(autouse=True)
def _skip_by_rhoai_version(request: FixtureRequest, rhoai_version: Version | None) -> None:
    """Autouse fixture that skips tests based on deprecated/min_version markers."""
    if rhoai_version is None:
        return

    for marker_name in ("deprecated", "min_version"):
        marker = request.node.get_closest_marker(marker_name)
        if marker:
            skip, reason = should_skip_for_marker(marker=marker, rhoai_version=rhoai_version, marker_name=marker_name)
            if skip:
                pytest.skip(reason)
