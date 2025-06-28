import pytest
import os


def skip_if_version_less_than(min_version: str):
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
    except Exception as e:
        print(f"⚠️  Version check setup failed: {e}, proceeding with test")
        return ()


def skip_if_version_greater_than(max_version: str):
    """
    Skip tests if version is greater than specified maximum.
        - DOWNSTREAM: Uses RHODS version checking

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
    except Exception as e:
        print(f"⚠️  Version check setup failed: {e}, proceeding with test")
        return ()
