"""
Helper utilities for Model Registry factory fixtures.

This module contains helper functions and classes that can be imported by test files
without violating the NIC001 linting rule (importing from conftest.py).
"""

import pytest
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from simple_logger.logger import get_logger

from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.deployment import Deployment
from ocp_resources.model_registry_modelregistry_opendatahub_io import ModelRegistry
from tests.model_registry.constants import MR_DB_IMAGE_DIGEST, MODEL_REGISTRY_DB_SECRET_STR_DATA
from utilities.constants import DscComponents

LOGGER = get_logger(name=__name__)


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
    def get_multi_namespace_parametrization(namespaces: List[str]) -> Tuple[str, List[Any]]:
        """Get parametrization for testing across multiple namespaces."""
        return (
            "updated_dsc_component_state_scope_class",
            [
                pytest.param(create_dsc_component_patch(namespace=namespace), id=f"namespace-{namespace}")  # noqa
                for namespace in namespaces
            ],
        )


# =============================================================================
# FACTORY CLASSES
# =============================================================================


@dataclass
class ModelRegistryDBConfig:
    """Configuration for Model Registry Database resources."""

    name_prefix: str = "mr-db"
    namespace: str = "default"
    teardown: bool = True
    mysql_image: str = MR_DB_IMAGE_DIGEST
    storage_size: str = "5Gi"
    access_mode: str = "ReadWriteOnce"
    database_name: str = MODEL_REGISTRY_DB_SECRET_STR_DATA["database-name"]
    database_user: str = MODEL_REGISTRY_DB_SECRET_STR_DATA["database-user"]
    database_password: str = MODEL_REGISTRY_DB_SECRET_STR_DATA["database-password"]
    port: int = 3306
    ssl_config: Optional[Dict[str, Any]] = None
    labels: Optional[Dict[str, str]] = None
    annotations: Optional[Dict[str, str]] = None


@dataclass
class ModelRegistryConfig:
    """Configuration for Model Registry instance."""

    name: str = "model-registry"
    namespace: str = "default"
    use_oauth_proxy: bool = True
    use_istio: bool = False
    teardown: bool = True
    grpc_config: Optional[Dict[str, Any]] = None
    rest_config: Optional[Dict[str, Any]] = None
    mysql_config: Optional[ModelRegistryDBConfig] = None
    labels: Optional[Dict[str, str]] = None
    wait_for_conditions: bool = True
    oauth_proxy_config: Optional[Dict[str, Any]] = None
    istio_config: Optional[Dict[str, Any]] = None


@dataclass
class ModelRegistryDBBundle:
    """Bundle containing all Model Registry DB resources."""

    service: Service
    pvc: PersistentVolumeClaim
    secret: Secret
    deployment: Deployment
    config: ModelRegistryDBConfig

    def cleanup(self) -> None:
        """Clean up all resources in this bundle in reverse order of creation."""
        # Cleanup in reverse order: Deployment -> Service -> PVC -> Secret
        resources = [
            ("deployment", self.deployment),
            ("service", self.service),
            ("pvc", self.pvc),
            ("secret", self.secret),
        ]

        for resource_type, resource in resources:
            try:
                if resource and hasattr(resource, "delete"):
                    LOGGER.info(f"Cleaning up {resource_type}: {resource.name}")
                    resource.delete(wait=True)
            except Exception as e:
                LOGGER.warning(f"Failed to cleanup {resource_type} {resource.name if resource else 'unknown'}: {e}")

    def get_mysql_config(self) -> Dict[str, Any]:
        """Get MySQL configuration dictionary for Model Registry."""
        return {
            "host": f"{self.deployment.name}.{self.deployment.namespace}.svc.cluster.local",
            "database": self.config.database_name,
            "passwordSecret": {"key": "database-password", "name": self.secret.name},
            "port": self.config.port,
            "skipDBCreation": False,
            "username": self.config.database_user,
        }


@dataclass
class ModelRegistryInstanceBundle:
    """Bundle containing Model Registry instance and related resources."""

    instance: ModelRegistry
    db_bundle: Optional[ModelRegistryDBBundle]
    service: Optional[Service]
    config: ModelRegistryConfig
    rest_endpoint: Optional[str] = None
    grpc_endpoint: Optional[str] = None

    def cleanup(self) -> None:
        """Clean up all resources in this bundle."""
        # Cleanup Model Registry instance first
        try:
            if self.instance and hasattr(self.instance, "delete"):
                LOGGER.info(f"Cleaning up Model Registry instance: {self.instance.name}")
                self.instance.delete(wait=True)
        except Exception as e:
            LOGGER.warning(
                f"Failed to cleanup Model Registry instance {self.instance.name if self.instance else 'unknown'}: {e}"
            )

        # Note: DB bundle cleanup is handled by the cleanup registry to ensure proper teardown order
        # We don't call self.db_bundle.cleanup() here to avoid double-cleanup
