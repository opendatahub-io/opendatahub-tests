"""
Helper utilities for Model Registry factory fixtures.

This module contains helper functions that can be imported by test files
without violating the NIC001 linting rule (importing from conftest.py).
"""

import pytest
from typing import Dict, Any, List, Tuple, Optional
from utilities.constants import DscComponents


def create_dsc_component_patch(
    namespace: str,
    management_state: str = DscComponents.ManagementState.MANAGED,
) -> Dict[str, Any]:
    """Helper function to create DSC component patch for Model Registry."""
    return {
        "component_patch": {
            DscComponents.MODELREGISTRY: {
                "managementState": management_state,
                "registriesNamespace": namespace,
            },
        }
    }


def create_test_parametrization(
    namespace: str,
    additional_params: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[Any]]:
    """
    Helper function to create parametrization for tests using Model Registry factory fixtures.

    Args:
        namespace: The namespace to use for Model Registry resources
        additional_params: Additional parameters to include in the parametrization

    Returns:
        Tuple of (param_names, param_values) for pytest.mark.parametrize
    """
    param_names = "updated_dsc_component_state_scope_class"
    param_values = [pytest.param(create_dsc_component_patch(namespace=namespace), id=f"mr-{namespace}")]

    if additional_params:
        # Add additional parameters to the parametrization
        additional_param_names = list(additional_params.keys())
        additional_param_values = list(additional_params.values())

        param_names += ", " + ", ".join(additional_param_names)
        param_values = [
            pytest.param(
                create_dsc_component_patch(namespace=namespace), *additional_param_values, id=f"mr-{namespace}"
            )  # noqa: FCN001
        ]

    return param_names, param_values


class ModelRegistryTestHelper:
    """Helper class for common Model Registry test patterns."""

    @staticmethod
    def get_standard_dsc_parametrization(namespace: str) -> Tuple[str, List[Any]]:
        """Get standard DSC parametrization for Model Registry tests."""
        return create_test_parametrization(namespace=namespace)  # noqa: FCN001

    @staticmethod
    def get_oauth_vs_istio_parametrization(namespace: str) -> Tuple[str, List[Any]]:
        """Get parametrization for testing both OAuth and Istio configurations."""
        return (
            "updated_dsc_component_state_scope_class, is_model_registry_oauth",
            [
                pytest.param(
                    create_dsc_component_patch(namespace=namespace), {"use_oauth_proxy": True}, id="oauth-proxy"
                ),  # noqa: FCN001
                pytest.param(create_dsc_component_patch(namespace=namespace), {"use_oauth_proxy": False}, id="istio"),  # noqa
            ],
        )

    @staticmethod
    def create_multi_namespace_parametrization(namespaces: List[str]) -> Tuple[str, List[Any]]:
        """Create parametrization for testing multiple namespaces."""
        return (
            "updated_dsc_component_state_scope_class",
            [
                pytest.param(create_dsc_component_patch(namespace=namespace), id=f"namespace-{namespace}")  # noqa
                for namespace in namespaces
            ],
        )
