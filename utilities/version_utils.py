import pytest
from typing import Any
from simple_logger.logger import get_logger


def compare_versions(version1: str, version2: str) -> int:
    """
    Compare two version strings.
    Returns: -1 if version1 < version2, 0 if equal, 1 if version1 > version2
    """
    from packaging import version

    try:
        return (version.parse(version1) > version.parse(version2)) - (version.parse(version1) < version.parse(version2))
    except Exception as e:
        LOGGER.error(f"[VERSION COMPARE] Error comparing versions: {e}")
        return 0


def skip_if_version_less_than(min_version: str) -> tuple[Any, ...]:
    """
    Skip tests if version is less than specified minimum.

    Works for both UPSTREAM and DOWNSTREAM distributions:
    - DOWNSTREAM: Uses RHODS version checking
    - UPSTREAM: Uses environment variable control

    Usage:
        pytestmark = [
            pytest.mark.serverless,
            *skip_if_version_less_than("2.23.0"),
        ]
    Args:
        min_version: Minimum required version (e.g., "2.23.0")

    Returns:
        Tuple of pytest marks for downstream OR skipif mark for upstream
    """
    try:
        from pytest_testconfig import config as py_config

        distribution = py_config.get("distribution", "")

        if distribution == "downstream":
            # Use existing RHODS version checking fixtures
            return (
                pytest.mark.parametrize("skip_if_downstream_version_less_than", [min_version], indirect=True),
                pytest.mark.usefixtures("skip_if_downstream_version_less_than"),
            )
        else:
            # For non-downstream distributions, return empty tuple
            return ()
    except Exception as e:
        print(f"⚠️  Version check setup failed: {e}, proceeding with test")
        return ()


def skip_if_version_greater_than(max_version: str) -> tuple[Any, ...]:
    """
    Skip tests if version is greater than specified maximum.

    Usage:
        pytestmark = [
            pytest.mark.serverless,
            *skip_if_version_greater_than("2.25.0"),
        ]

    Args:
        max_version: Maximum allowed version (e.g., "2.25.0")

    Returns:
        Tuple of pytest marks for downstream
    """
    try:
        from pytest_testconfig import config as py_config

        distribution = py_config.get("distribution", "")

        if distribution == "downstream":
            # Use existing RHODS version checking fixtures
            return (
                pytest.mark.parametrize("skip_if_downstream_version_greater_than", [max_version], indirect=True),
                pytest.mark.usefixtures("skip_if_downstream_version_greater_than"),
            )
        else:
            # For non-downstream distributions, return empty tuple
            return ()
    except Exception as e:
        print(f"⚠️  Version check setup failed: {e}, proceeding with test")
        return ()
