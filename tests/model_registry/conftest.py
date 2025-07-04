import os
import pytest
from pytest import Config
import schemathesis
from typing import Generator, Any

from ocp_resources.infrastructure import Infrastructure
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.namespace import Namespace
from ocp_resources.service import Service
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment

from ocp_resources.model_registry_modelregistry_opendatahub_io import ModelRegistry
from schemathesis.specs.openapi.schemas import BaseOpenAPISchema
from schemathesis.generation.stateful.state_machine import APIStateMachine
from schemathesis.core.transport import Response
from schemathesis.generation.case import Case
from ocp_resources.resource import ResourceEditor

from pytest import FixtureRequest
from simple_logger.logger import get_logger
from kubernetes.dynamic import DynamicClient
from pytest_testconfig import config as py_config
from model_registry.types import RegisteredModel

# Factory fixture imports
from dataclasses import dataclass
from typing import Optional, List, Dict, Callable
import uuid
from contextlib import contextmanager
from tests.model_registry.constants import MR_DB_IMAGE_DIGEST
from utilities.constants import Annotations

from tests.model_registry.constants import (
    MR_OPERATOR_NAME,
    MR_INSTANCE_NAME,
    DB_RESOURCES_NAME,
    MODEL_REGISTRY_DB_SECRET_STR_DATA,
    MODEL_REGISTRY_DB_SECRET_ANNOTATIONS,
    OAUTH_PROXY_CONFIG_DICT,
    MODEL_REGISTRY_STANDARD_LABELS,
    ISTIO_CONFIG_DICT,
)
from tests.model_registry.rest_api.utils import ModelRegistryV1Alpha1
from utilities.constants import Labels
from tests.model_registry.utils import (
    get_endpoint_from_mr_service,
    get_mr_service_by_label,
    get_model_registry_deployment_template_dict,
    get_model_registry_db_label_dict,
    wait_for_pods_running,
)
from utilities.constants import Protocols, DscComponents
from model_registry import ModelRegistry as ModelRegistryClient
from semver import Version
from utilities.general import wait_for_pods_by_labels

LOGGER = get_logger(name=__name__)

MIN_MR_VERSION = Version.parse(version="2.20.0")


@pytest.fixture(scope="class")
def model_registry_namespace(updated_dsc_component_state_scope_class: DataScienceCluster) -> str:
    return updated_dsc_component_state_scope_class.instance.spec.components.modelregistry.registriesNamespace


@pytest.fixture(scope="class")
def model_registry_db_service(
    pytestconfig: Config,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    teardown_resources: bool,
    is_model_registry_oauth: bool,
) -> Generator[Service, Any, Any]:
    if pytestconfig.option.post_upgrade:
        mr_db_service = Service(name=DB_RESOURCES_NAME, namespace=model_registry_namespace, ensure_exists=True)
        yield mr_db_service
        mr_db_service.delete(wait=True)
    else:
        with Service(
            client=admin_client,
            name=DB_RESOURCES_NAME,
            namespace=model_registry_namespace,
            ports=[
                {
                    "name": "mysql",
                    "nodePort": 0,
                    "port": 3306,
                    "protocol": "TCP",
                    "appProtocol": "tcp",
                    "targetPort": 3306,
                }
            ],
            selector={
                "name": DB_RESOURCES_NAME,
            },
            label=get_model_registry_db_label_dict(db_resource_name=DB_RESOURCES_NAME),
            annotations={
                "template.openshift.io/expose-uri": r"mysql://{.spec.clusterIP}:{.spec.ports[?(.name==\mysql\)].port}",
            },
            teardown=teardown_resources,
        ) as mr_db_service:
            yield mr_db_service


@pytest.fixture(scope="class")
def model_registry_db_pvc(
    pytestconfig: Config,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    teardown_resources: bool,
    is_model_registry_oauth: bool,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    if pytestconfig.option.post_upgrade:
        mr_db_pvc = PersistentVolumeClaim(
            name=DB_RESOURCES_NAME, namespace=model_registry_namespace, ensure_exists=True
        )
        yield mr_db_pvc
        mr_db_pvc.delete(wait=True)
    else:
        with PersistentVolumeClaim(
            accessmodes="ReadWriteOnce",
            name=DB_RESOURCES_NAME,
            namespace=model_registry_namespace,
            client=admin_client,
            size="5Gi",
            label=get_model_registry_db_label_dict(db_resource_name=DB_RESOURCES_NAME),
            teardown=teardown_resources,
        ) as pvc:
            yield pvc


@pytest.fixture(scope="class")
def model_registry_db_secret(
    pytestconfig: Config,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    teardown_resources: bool,
    is_model_registry_oauth: bool,
) -> Generator[Secret, Any, Any]:
    if pytestconfig.option.post_upgrade:
        mr_db_secret = Secret(name=DB_RESOURCES_NAME, namespace=model_registry_namespace, ensure_exists=True)
        yield mr_db_secret
        mr_db_secret.delete(wait=True)
    else:
        with Secret(
            client=admin_client,
            name=DB_RESOURCES_NAME,
            namespace=model_registry_namespace,
            string_data=MODEL_REGISTRY_DB_SECRET_STR_DATA,
            label=get_model_registry_db_label_dict(db_resource_name=DB_RESOURCES_NAME),
            annotations=MODEL_REGISTRY_DB_SECRET_ANNOTATIONS,
            teardown=teardown_resources,
        ) as mr_db_secret:
            yield mr_db_secret


@pytest.fixture(scope="class")
def model_registry_db_deployment(
    pytestconfig: Config,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    model_registry_db_secret: Secret,
    model_registry_db_pvc: PersistentVolumeClaim,
    model_registry_db_service: Service,
    teardown_resources: bool,
    is_model_registry_oauth: bool,
) -> Generator[Deployment, Any, Any]:
    if pytestconfig.option.post_upgrade:
        db_deployment = Deployment(name=DB_RESOURCES_NAME, namespace=model_registry_namespace, ensure_exists=True)
        yield db_deployment
        db_deployment.delete(wait=True)
    else:
        with Deployment(
            name=DB_RESOURCES_NAME,
            namespace=model_registry_namespace,
            annotations={
                "template.alpha.openshift.io/wait-for-ready": "true",
            },
            label=get_model_registry_db_label_dict(db_resource_name=DB_RESOURCES_NAME),
            replicas=1,
            revision_history_limit=0,
            selector={"matchLabels": {"name": DB_RESOURCES_NAME}},
            strategy={"type": "Recreate"},
            template=get_model_registry_deployment_template_dict(
                secret_name=model_registry_db_secret.name, resource_name=DB_RESOURCES_NAME
            ),
            wait_for_resource=True,
            teardown=teardown_resources,
        ) as mr_db_deployment:
            mr_db_deployment.wait_for_replicas(deployed=True)
            yield mr_db_deployment


@pytest.fixture(scope="class")
def model_registry_instance(
    pytestconfig: Config,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    model_registry_mysql_config: dict[str, Any],
    teardown_resources: bool,
    is_model_registry_oauth: bool,
) -> Generator[ModelRegistry, Any, Any]:
    """Creates a model registry instance with oauth proxy configuration."""
    if pytestconfig.option.post_upgrade:
        mr_instance = ModelRegistry(name=MR_INSTANCE_NAME, namespace=model_registry_namespace, ensure_exists=True)
        yield mr_instance
        mr_instance.delete(wait=True)
    else:
        istio_config = None
        oauth_config = None
        mr_class_name = ModelRegistry
        if is_model_registry_oauth:
            LOGGER.warning("Requested Ouath Proxy configuration:")
            oauth_config = OAUTH_PROXY_CONFIG_DICT
        else:
            LOGGER.warning("Requested OSSM configuration:")
            istio_config = ISTIO_CONFIG_DICT
            mr_class_name = ModelRegistryV1Alpha1
        with mr_class_name(
            name=MR_INSTANCE_NAME,
            namespace=model_registry_namespace,
            label=MODEL_REGISTRY_STANDARD_LABELS,
            grpc={},
            rest={},
            istio=istio_config,
            oauth_proxy=oauth_config,
            mysql=model_registry_mysql_config,
            wait_for_resource=True,
            teardown=teardown_resources,
        ) as mr:
            mr.wait_for_condition(condition="Available", status="True")
            mr.wait_for_condition(condition="OAuthProxyAvailable", status="True")

            yield mr


@pytest.fixture(scope="class")
def model_registry_mysql_config(
    request: FixtureRequest,
    model_registry_db_deployment: Deployment,
    model_registry_db_secret: Secret,
) -> dict[str, Any]:
    """
    Fixture to build the MySQL config dictionary for Model Registry.
    Expects request.param to be a dict. If 'sslRootCertificateConfigMap' is not present, it defaults to None.
    If 'sslRootCertificateConfigMap' is present, it will be used to configure the MySQL connection.

    Args:
        request: The pytest request object
        model_registry_db_deployment: The model registry db deployment
        model_registry_db_secret: The model registry db secret

    Returns:
        dict[str, Any]: The MySQL config dictionary
    """
    param = request.param if hasattr(request, "param") else {}
    config = {
        "host": f"{model_registry_db_deployment.name}.{model_registry_db_deployment.namespace}.svc.cluster.local",
        "database": MODEL_REGISTRY_DB_SECRET_STR_DATA["database-name"],
        "passwordSecret": {"key": "database-password", "name": model_registry_db_deployment.name},
        "port": param.get("port", 3306),
        "skipDBCreation": False,
        "username": MODEL_REGISTRY_DB_SECRET_STR_DATA["database-user"],
    }
    if "sslRootCertificateConfigMap" in param:
        config["sslRootCertificateConfigMap"] = param["sslRootCertificateConfigMap"]

    return config


@pytest.fixture(scope="class")
def model_registry_instance_service(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    model_registry_instance: ModelRegistry,
) -> Service:
    """
    Get the service for the regular model registry instance.
    Args:
        admin_client: The admin client
        model_registry_namespace: The namespace where the model registry is deployed
        model_registry_instance: The model registry instance to get the service for
    Returns:
        Service: The service for the model registry instance
    """
    return get_mr_service_by_label(
        client=admin_client, namespace_name=model_registry_namespace, mr_instance=model_registry_instance
    )


@pytest.fixture(scope="class")
def model_registry_instance_rest_endpoint(
    model_registry_instance_service: Service,
) -> str:
    """
    Get the REST endpoint for the model registry instance.
    Args:
        model_registry_instance_service: The service for the model registry instance
    Returns:
        str: The REST endpoint for the model registry instance
    """
    return get_endpoint_from_mr_service(svc=model_registry_instance_service, protocol=Protocols.REST)


@pytest.fixture(scope="class")
def generated_schema(pytestconfig: Config, model_registry_instance_rest_endpoint: str) -> BaseOpenAPISchema:
    os.environ["API_HOST"] = model_registry_instance_rest_endpoint
    config = schemathesis.config.SchemathesisConfig.from_path(f"{pytestconfig.rootpath}/schemathesis.toml")
    schema = schemathesis.openapi.from_url(
        url="https://raw.githubusercontent.com/kubeflow/model-registry/main/api/openapi/model-registry.yaml",
        config=config,
    )
    return schema


@pytest.fixture()
def state_machine(generated_schema: BaseOpenAPISchema, current_client_token: str) -> APIStateMachine:
    BaseAPIWorkflow = generated_schema.as_state_machine()

    class APIWorkflow(BaseAPIWorkflow):  # type: ignore
        headers: dict[str, str]

        def setup(self) -> None:
            self.headers = {"Authorization": f"Bearer {current_client_token}", "Content-Type": "application/json"}

        def before_call(self, case: Case) -> None:
            LOGGER.info(f"Checking: {case.method} {case.path}")

        # these kwargs are passed to requests.request()
        def get_call_kwargs(self, case: Case) -> dict[str, Any]:
            return {"verify": False, "headers": self.headers}

        def after_call(self, response: Response, case: Case) -> None:
            LOGGER.info(
                f"Method tested: {case.method}, API: {case.path}, response code:{response.status_code},"
                f" Full Response:{response.text}"
            )

    return APIWorkflow


# =============================================================================
# FACTORY FIXTURES - Phase 1 Implementation
# =============================================================================


# =============================================================================
# CLEANUP REGISTRY - Ensures proper teardown order
# =============================================================================


class ModelRegistryCleanupRegistry:
    """Registry to track and manage cleanup of Model Registry resources."""

    def __init__(self) -> None:
        self._cleanup_functions: List[Callable[[], None]] = []

    def register_cleanup(self, cleanup_func: Callable[[], None]) -> None:
        """Register a cleanup function to be called during teardown."""
        self._cleanup_functions.append(cleanup_func)

    def cleanup_all(self) -> None:
        """Execute all registered cleanup functions in reverse order."""
        # Execute cleanup functions in reverse order (LIFO)
        for cleanup_func in reversed(self._cleanup_functions):
            try:
                cleanup_func()
            except Exception as e:
                LOGGER.warning(f"Error during cleanup: {e}")

        # Clear the registry after cleanup
        self._cleanup_functions.clear()


# Global cleanup registry instance
_cleanup_registry = ModelRegistryCleanupRegistry()


@pytest.fixture(scope="class")
def model_registry_cleanup_registry() -> ModelRegistryCleanupRegistry:
    """Provide access to the cleanup registry."""
    return _cleanup_registry


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


@pytest.fixture(scope="class")
def model_registry_db_factory(
    admin_client: DynamicClient,
    pytestconfig: Config,
    model_registry_cleanup_registry: ModelRegistryCleanupRegistry,
) -> Generator[Callable[[ModelRegistryDBConfig], ModelRegistryDBBundle], Any, Any]:
    """Factory fixture for creating Model Registry DB bundles."""
    created_bundles: List[ModelRegistryDBBundle] = []

    def create_db_bundle(config: ModelRegistryDBConfig) -> ModelRegistryDBBundle:
        """Create a complete Model Registry DB bundle."""
        # Generate unique names to avoid conflicts
        unique_suffix = str(uuid.uuid4())[:8]
        db_name = f"{config.name_prefix}-{unique_suffix}"

        # Use existing labels helper or create custom ones
        labels = config.labels or get_model_registry_db_label_dict(db_resource_name=db_name)
        annotations = config.annotations or MODEL_REGISTRY_DB_SECRET_ANNOTATIONS

        # Handle post-upgrade scenario
        if pytestconfig.option.post_upgrade:
            service = Service(name=db_name, namespace=config.namespace, ensure_exists=True)
            pvc = PersistentVolumeClaim(name=db_name, namespace=config.namespace, ensure_exists=True)
            secret = Secret(name=db_name, namespace=config.namespace, ensure_exists=True)
            deployment = Deployment(name=db_name, namespace=config.namespace, ensure_exists=True)
        else:
            # Create resources in order: Secret -> PVC -> Service -> Deployment
            secret = None
            pvc = None
            service = None
            deployment = None

            try:
                # Create Secret first
                LOGGER.info(f"Creating DB secret: {db_name}")
                secret = Secret(
                    client=admin_client,
                    name=db_name,
                    namespace=config.namespace,
                    string_data={
                        "database-name": config.database_name,
                        "database-user": config.database_user,
                        "database-password": config.database_password,
                    },
                    label=labels,
                    annotations=annotations,
                )
                secret.create()

                # Create PVC
                LOGGER.info(f"Creating DB PVC: {db_name}")
                pvc = PersistentVolumeClaim(
                    client=admin_client,
                    name=db_name,
                    namespace=config.namespace,
                    accessmodes=config.access_mode,
                    size=config.storage_size,
                    label=labels,
                )
                pvc.create()

                # Create Service
                LOGGER.info(f"Creating DB service: {db_name}")
                service = Service(
                    client=admin_client,
                    name=db_name,
                    namespace=config.namespace,
                    ports=[
                        {
                            "name": "mysql",
                            "nodePort": 0,
                            "port": config.port,
                            "protocol": "TCP",
                            "appProtocol": "tcp",
                            "targetPort": config.port,
                        }
                    ],
                    selector={"name": db_name},
                    label=labels,
                    annotations={
                        "template.openshift.io/expose-uri": (
                            "mysql://{.spec.clusterIP}:{.spec.ports[?(.name==\\mysql\\)].port}"
                        ),
                    },
                )
                service.create()

                # Create Deployment
                LOGGER.info(f"Creating DB deployment: {db_name}")
                deployment = Deployment(
                    client=admin_client,
                    name=db_name,
                    namespace=config.namespace,
                    annotations={"template.alpha.openshift.io/wait-for-ready": "true"},
                    label=labels,
                    replicas=1,
                    revision_history_limit=0,
                    selector={"matchLabels": {"name": db_name}},
                    strategy={"type": "Recreate"},
                    template=get_model_registry_deployment_template_dict(
                        secret_name=secret.name, resource_name=db_name
                    ),
                )
                deployment.create()

                # Wait for deployment to be ready
                LOGGER.info(f"Waiting for DB deployment to be ready: {db_name}")
                deployment.wait_for_replicas(deployed=True)

            except Exception as e:
                # Clean up any resources that were created before the failure
                LOGGER.error(f"Failed to create DB bundle {db_name}: {e}")
                for resource_name, resource in [
                    ("deployment", deployment),
                    ("service", service),
                    ("pvc", pvc),
                    ("secret", secret),
                ]:
                    if resource:
                        try:
                            LOGGER.info(f"Cleaning up failed DB {resource_name}: {resource.name}")
                            resource.delete(wait=True)
                        except Exception as cleanup_e:
                            LOGGER.warning(f"Failed to cleanup {resource_name} during error recovery: {cleanup_e}")
                raise

        bundle = ModelRegistryDBBundle(service=service, pvc=pvc, secret=secret, deployment=deployment, config=config)
        created_bundles.append(bundle)

        # Register cleanup function with the registry
        if config.teardown:
            model_registry_cleanup_registry.register_cleanup(cleanup_func=bundle.cleanup)

        LOGGER.info(f"Successfully created DB bundle: {db_name}")
        return bundle

    yield create_db_bundle

    # Note: Cleanup is now handled by the cleanup registry, not here
    # This ensures proper teardown order relative to DSC patch revert


@pytest.fixture(scope="class")
def model_registry_instance_factory(
    admin_client: DynamicClient,
    pytestconfig: Config,
    model_registry_db_factory: Callable[[ModelRegistryDBConfig], ModelRegistryDBBundle],
    model_registry_cleanup_registry: ModelRegistryCleanupRegistry,
) -> Generator[Callable[[ModelRegistryConfig], ModelRegistryInstanceBundle], Any, Any]:
    """Factory fixture for creating Model Registry instances."""
    created_instances: List[ModelRegistryInstanceBundle] = []

    def create_instance(config: ModelRegistryConfig) -> ModelRegistryInstanceBundle:
        """Create a complete Model Registry instance bundle."""
        # Generate unique name to avoid conflicts
        unique_suffix = str(uuid.uuid4())[:8]
        instance_name = f"{config.name}-{unique_suffix}"

        # Create or get DB bundle if needed
        db_bundle = None
        mysql_config = None
        if config.mysql_config:
            db_bundle = model_registry_db_factory(config.mysql_config)  # noqa: FCN001
            mysql_config = db_bundle.get_mysql_config()

        # Configure OAuth vs Istio
        oauth_config = None
        istio_config = None
        mr_class = ModelRegistry

        if config.use_oauth_proxy:
            oauth_config = config.oauth_proxy_config or OAUTH_PROXY_CONFIG_DICT
        elif config.use_istio:
            istio_config = config.istio_config or ISTIO_CONFIG_DICT
            mr_class = ModelRegistryV1Alpha1

        # Default labels
        labels = config.labels or MODEL_REGISTRY_STANDARD_LABELS.copy()
        labels.update({
            Annotations.KubernetesIo.NAME: instance_name,
            Annotations.KubernetesIo.INSTANCE: instance_name,
        })

        # Handle post-upgrade scenario
        if pytestconfig.option.post_upgrade:
            instance = mr_class(name=instance_name, namespace=config.namespace, ensure_exists=True)
        else:
            # Create Model Registry instance
            try:
                LOGGER.info(f"Creating Model Registry instance: {instance_name}")
                instance = mr_class(
                    client=admin_client,
                    name=instance_name,
                    namespace=config.namespace,
                    label=labels,
                    grpc=config.grpc_config or {},
                    rest=config.rest_config or {},
                    istio=istio_config,
                    oauth_proxy=oauth_config,
                    mysql=mysql_config,
                )
                instance.create()

                # Wait for conditions if requested
                if config.wait_for_conditions:
                    LOGGER.info(f"Waiting for Model Registry conditions: {instance_name}")
                    instance.wait_for_condition(condition="Available", status="True")
                    if config.use_oauth_proxy:
                        instance.wait_for_condition(condition="OAuthProxyAvailable", status="True")

            except Exception as e:
                LOGGER.error(f"Failed to create Model Registry instance {instance_name}: {e}")
                # Clean up the instance if it was created but failed during condition waiting
                try:
                    if "instance" in locals() and instance:
                        LOGGER.info(f"Cleaning up failed Model Registry instance: {instance.name}")
                        instance.delete(wait=True)
                except Exception as cleanup_e:
                    LOGGER.warning(f"Failed to cleanup Model Registry instance during error recovery: {cleanup_e}")
                # Note: DB bundle cleanup is already registered with the cleanup registry
                raise

        # Get service and endpoints
        service = None
        rest_endpoint = None
        grpc_endpoint = None

        try:
            LOGGER.info(f"Getting service for Model Registry instance: {instance_name}")
            service = get_mr_service_by_label(
                client=admin_client, namespace_name=config.namespace, mr_instance=instance
            )
            LOGGER.info(f"Getting endpoints for Model Registry service: {service.name}")
            rest_endpoint = get_endpoint_from_mr_service(svc=service, protocol=Protocols.REST)
            grpc_endpoint = get_endpoint_from_mr_service(svc=service, protocol=Protocols.GRPC)
            LOGGER.info(f"Model Registry endpoints - REST: {rest_endpoint}, gRPC: {grpc_endpoint}")
        except Exception as e:
            LOGGER.warning(f"Failed to get service/endpoints for {instance_name}: {e}")

        bundle = ModelRegistryInstanceBundle(
            instance=instance,
            db_bundle=db_bundle,
            service=service,
            config=config,
            rest_endpoint=rest_endpoint,
            grpc_endpoint=grpc_endpoint,
        )
        created_instances.append(bundle)

        # Register cleanup function with the registry (but not for DB bundle - that's already registered)
        if config.teardown:

            def cleanup_instance() -> None:
                try:
                    if bundle.instance and hasattr(bundle.instance, "delete"):
                        LOGGER.info(f"Cleaning up Model Registry instance: {bundle.instance.name}")
                        bundle.instance.delete(wait=True)
                except Exception as e:
                    LOGGER.warning(
                        f"Failed to cleanup Model Registry instance "
                        f"{bundle.instance.name if bundle.instance else 'unknown'}: {e}"
                    )

            model_registry_cleanup_registry.register_cleanup(cleanup_func=cleanup_instance)

        LOGGER.info(f"Successfully created Model Registry bundle: {instance_name}")
        return bundle

    yield create_instance

    # Note: Cleanup is now handled by the cleanup registry, not here
    # This ensures proper teardown order relative to DSC patch revert


@pytest.fixture(scope="class")
def model_registry_client_factory(
    current_client_token: str,
) -> Callable[[str], ModelRegistryClient]:
    """Factory fixture for creating Model Registry clients."""

    def create_client(rest_endpoint: str) -> ModelRegistryClient:
        """Create a Model Registry client for the given endpoint."""
        server, port = rest_endpoint.split(":")
        return ModelRegistryClient(
            server_address=f"{Protocols.HTTPS}://{server}",
            port=int(port),
            author="opendatahub-test",
            user_token=current_client_token,
            is_secure=False,
        )

    return create_client


# =============================================================================
# CONVENIENCE FIXTURES - Common configurations
# =============================================================================


@pytest.fixture(scope="class")
def default_model_registry_factory(
    model_registry_instance_factory: Callable[[ModelRegistryConfig], ModelRegistryInstanceBundle],
    model_registry_namespace: str,
) -> Callable[[Optional[str]], ModelRegistryInstanceBundle]:
    """Factory for creating Model Registry instances with default configuration."""

    def create_default_instance(name_prefix: Optional[str] = None) -> ModelRegistryInstanceBundle:
        name_prefix = name_prefix or "test-mr"
        config = ModelRegistryConfig(
            name=name_prefix,
            namespace=model_registry_namespace,
            mysql_config=ModelRegistryDBConfig(
                name_prefix=f"{name_prefix}-db",
                namespace=model_registry_namespace,
            ),
        )
        return model_registry_instance_factory(config)  # noqa: FCN001

    return create_default_instance


@pytest.fixture(scope="class")
def oauth_model_registry_factory(
    model_registry_instance_factory: Callable[[ModelRegistryConfig], ModelRegistryInstanceBundle],
    model_registry_namespace: str,
) -> Callable[[Optional[str]], ModelRegistryInstanceBundle]:
    """Factory for creating Model Registry instances with OAuth configuration."""

    def create_oauth_instance(name_prefix: Optional[str] = None) -> ModelRegistryInstanceBundle:
        name_prefix = name_prefix or "oauth-mr"
        config = ModelRegistryConfig(
            name=name_prefix,
            namespace=model_registry_namespace,
            use_oauth_proxy=True,
            use_istio=False,
            mysql_config=ModelRegistryDBConfig(
                name_prefix=f"{name_prefix}-db",
                namespace=model_registry_namespace,
            ),
        )
        return model_registry_instance_factory(config)  # noqa: FCN001

    return create_oauth_instance


@pytest.fixture(scope="class")
def istio_model_registry_factory(
    model_registry_instance_factory: Callable[[ModelRegistryConfig], ModelRegistryInstanceBundle],
    model_registry_namespace: str,
) -> Callable[[Optional[str]], ModelRegistryInstanceBundle]:
    """Factory for creating Model Registry instances with Istio configuration."""

    def create_istio_instance(name_prefix: Optional[str] = None) -> ModelRegistryInstanceBundle:
        name_prefix = name_prefix or "istio-mr"
        config = ModelRegistryConfig(
            name=name_prefix,
            namespace=model_registry_namespace,
            use_oauth_proxy=False,
            use_istio=True,
            mysql_config=ModelRegistryDBConfig(
                name_prefix=f"{name_prefix}-db",
                namespace=model_registry_namespace,
            ),
        )
        return model_registry_instance_factory(config)  # noqa: FCN001

    return create_istio_instance


@pytest.fixture(scope="class")
def standalone_db_factory(
    model_registry_db_factory: Callable[[ModelRegistryDBConfig], ModelRegistryDBBundle],
    model_registry_namespace: str,
) -> Callable[[Optional[str]], ModelRegistryDBBundle]:
    """Factory for creating standalone Model Registry DB instances."""

    def create_standalone_db(name_prefix: Optional[str] = None) -> ModelRegistryDBBundle:
        name_prefix = name_prefix or "standalone-db"
        config = ModelRegistryDBConfig(
            name_prefix=name_prefix,
            namespace=model_registry_namespace,
        )
        return model_registry_db_factory(config)  # noqa: FCN001

    return create_standalone_db


@pytest.fixture(scope="class")
def multi_instance_factory(
    model_registry_instance_factory: Callable[[ModelRegistryConfig], ModelRegistryInstanceBundle],
    model_registry_namespace: str,
) -> Callable[[int, Optional[str]], List[ModelRegistryInstanceBundle]]:
    """Factory for creating multiple Model Registry instances."""

    def create_multiple_instances(count: int, name_prefix: Optional[str] = None) -> List[ModelRegistryInstanceBundle]:
        name_prefix = name_prefix or "multi-mr"
        instances = []

        for i in range(count):
            config = ModelRegistryConfig(
                name=f"{name_prefix}-{i}",
                namespace=model_registry_namespace,
                mysql_config=ModelRegistryDBConfig(
                    name_prefix=f"{name_prefix}-db-{i}",
                    namespace=model_registry_namespace,
                ),
            )
            instances.append(model_registry_instance_factory(config))  # noqa: FCN001

        return instances

    return create_multiple_instances


# =============================================================================
# HELPER FUNCTIONS FOR DSC SETUP
# =============================================================================

# Helper functions moved to tests.model_registry.factory_utils to avoid NIC001 linting issues


@pytest.fixture(scope="class")
def simple_model_registry_factory(
    model_registry_instance_factory: Callable[[ModelRegistryConfig], ModelRegistryInstanceBundle],
    model_registry_namespace: str,
) -> Callable[[Optional[Dict[str, Any]]], ModelRegistryInstanceBundle]:
    """
    Simplified factory for creating Model Registry instances with minimal configuration.

    This fixture provides a simpler interface for tests that don't need complex customization.
    """

    def create_simple_instance(config_overrides: Optional[Dict[str, Any]] = None) -> ModelRegistryInstanceBundle:
        config_overrides = config_overrides or {}

        # Default configuration
        config = ModelRegistryConfig(
            name=config_overrides.get("name", "simple-mr"),
            namespace=model_registry_namespace,
            use_oauth_proxy=config_overrides.get("use_oauth_proxy", True),
            mysql_config=ModelRegistryDBConfig(
                name_prefix=config_overrides.get("db_name_prefix", "simple-mr-db"),
                namespace=model_registry_namespace,
                storage_size=config_overrides.get("storage_size", "5Gi"),
            ),
        )

        # Apply any additional overrides
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return model_registry_instance_factory(config)  # noqa: FCN001

    return create_simple_instance


# =============================================================================
# HELPER CONTEXT MANAGERS
# =============================================================================


@contextmanager
def temporary_model_registry(
    config: ModelRegistryConfig,
    instance_factory: Callable[[ModelRegistryConfig], ModelRegistryInstanceBundle],
) -> Generator[ModelRegistryInstanceBundle, None, None]:
    """Context manager for temporarily creating a Model Registry instance."""
    bundle = instance_factory(config)  # noqa: FCN001
    try:
        yield bundle
    finally:
        if bundle.config.teardown:
            bundle.cleanup()


@contextmanager
def temporary_model_registry_db(
    config: ModelRegistryDBConfig,
    db_factory: Callable[[ModelRegistryDBConfig], ModelRegistryDBBundle],
) -> Generator[ModelRegistryDBBundle, None, None]:
    """Context manager for temporarily creating a Model Registry DB."""
    bundle = db_factory(config)  # noqa: FCN001
    try:
        yield bundle
    finally:
        if bundle.config.teardown:
            bundle.cleanup()


# =============================================================================
# EXAMPLE TEST UTILITIES
# =============================================================================


# ModelRegistryTestHelper class moved to tests.model_registry.factory_utils to avoid NIC001 linting issues


@pytest.fixture(scope="class")
def updated_dsc_component_state_scope_class(
    pytestconfig: Config,
    request: FixtureRequest,
    dsc_resource: DataScienceCluster,
    admin_client: DynamicClient,
    teardown_resources: bool,
    is_model_registry_oauth: bool,
    model_registry_cleanup_registry: ModelRegistryCleanupRegistry,
) -> Generator[DataScienceCluster, Any, Any]:
    if not teardown_resources or pytestconfig.option.post_upgrade:
        # if we are not tearing down resources or we are in post upgrade, we don't need to do anything
        # the pre_upgrade/post_upgrade fixtures will handle the rest
        yield dsc_resource
    else:
        original_components = dsc_resource.instance.spec.components
        component_patch = request.param["component_patch"]

        with ResourceEditor(patches={dsc_resource: {"spec": {"components": component_patch}}}):
            for component_name in component_patch:
                dsc_resource.wait_for_condition(
                    condition=DscComponents.COMPONENT_MAPPING[component_name], status="True"
                )
            if component_patch.get(DscComponents.MODELREGISTRY):
                namespace = Namespace(
                    name=dsc_resource.instance.spec.components.modelregistry.registriesNamespace, ensure_exists=True
                )
                namespace.wait_for_status(status=Namespace.Status.ACTIVE)
            wait_for_pods_running(
                admin_client=admin_client,
                namespace_name=py_config["applications_namespace"],
                number_of_consecutive_checks=6,
            )
            yield dsc_resource

            # CRITICAL: Clean up factory-created resources BEFORE reverting DSC patch
            LOGGER.info("Cleaning up Model Registry factory resources before reverting DSC patch")
            model_registry_cleanup_registry.cleanup_all()

        for component_name, value in component_patch.items():
            LOGGER.info(f"Waiting for component {component_name} to be updated.")
            if original_components[component_name]["managementState"] == DscComponents.ManagementState.MANAGED:
                dsc_resource.wait_for_condition(
                    condition=DscComponents.COMPONENT_MAPPING[component_name], status="True"
                )
            if (
                component_name == DscComponents.MODELREGISTRY
                and value.get("managementState") == DscComponents.ManagementState.MANAGED
            ):
                # Since namespace specified in registriesNamespace is automatically created after setting
                # managementStateto Managed. We need to explicitly delete it on clean up.
                namespace = Namespace(name=value["registriesNamespace"], ensure_exists=True)
                if namespace:
                    namespace.delete(wait=True)


@pytest.fixture(scope="class")
def pre_upgrade_dsc_patch(
    dsc_resource: DataScienceCluster,
    admin_client: DynamicClient,
) -> DataScienceCluster:
    original_components = dsc_resource.instance.spec.components
    component_patch = {DscComponents.MODELREGISTRY: {"managementState": DscComponents.ManagementState.MANAGED}}
    if (
        original_components.get(DscComponents.MODELREGISTRY).get("managementState")
        == DscComponents.ManagementState.MANAGED
    ):
        pytest.fail("Model Registry is already set to Managed before upgrade - was this intentional?")
    else:
        editor = ResourceEditor(patches={dsc_resource: {"spec": {"components": component_patch}}})
        editor.update()
        dsc_resource.wait_for_condition(condition=DscComponents.COMPONENT_MAPPING["modelregistry"], status="True")
        namespace = Namespace(
            name=dsc_resource.instance.spec.components.modelregistry.registriesNamespace, ensure_exists=True
        )
        namespace.wait_for_status(status=Namespace.Status.ACTIVE)
        wait_for_pods_running(
            admin_client=admin_client,
            namespace_name=py_config["applications_namespace"],
            number_of_consecutive_checks=6,
        )
        return dsc_resource


@pytest.fixture(scope="class")
def post_upgrade_dsc_patch(
    dsc_resource: DataScienceCluster,
) -> Generator[DataScienceCluster, Any, Any]:
    # yield right away so that the rest of the fixture is executed at teardown time
    yield dsc_resource

    # the state we found after the upgrade
    original_components = dsc_resource.instance.spec.components
    # We don't have an easy way to figure out the state of the components before the upgrade at runtime
    # For now we know that MR has to go back to Removed after post upgrade tests are run
    component_patch = {DscComponents.MODELREGISTRY: {"managementState": DscComponents.ManagementState.REMOVED}}
    if (
        original_components.get(DscComponents.MODELREGISTRY).get("managementState")
        == DscComponents.ManagementState.REMOVED
    ):
        pytest.fail("Model Registry is already set to Removed after upgrade - was this intentional?")
    else:
        editor = ResourceEditor(patches={dsc_resource: {"spec": {"components": component_patch}}})
        editor.update()
    ns = original_components.get(DscComponents.MODELREGISTRY).get("registriesNamespace")
    namespace = Namespace(name=ns, ensure_exists=True)
    if namespace:
        namespace.delete(wait=True)


@pytest.fixture(scope="class")
def model_registry_client(
    current_client_token: str,
    model_registry_instance_rest_endpoint: str,
) -> ModelRegistryClient:
    """
    Get a client for the model registry instance.
    Args:
        request: The pytest request object
        current_client_token: The current client token
    Returns:
        ModelRegistryClient: A client for the model registry instance
    """
    server, port = model_registry_instance_rest_endpoint.split(":")
    return ModelRegistryClient(
        server_address=f"{Protocols.HTTPS}://{server}",
        port=int(port),
        author="opendatahub-test",
        user_token=current_client_token,
        is_secure=False,
    )


@pytest.fixture(scope="class")
def registered_model(request: FixtureRequest, model_registry_client: ModelRegistryClient) -> RegisteredModel:
    return model_registry_client.register_model(
        name=request.param.get("model_name"),
        uri=request.param.get("model_uri"),
        version=request.param.get("model_version"),
        description=request.param.get("model_description"),
        model_format_name=request.param.get("model_format"),
        model_format_version=request.param.get("model_format_version"),
        storage_key=request.param.get("model_storage_key"),
        storage_path=request.param.get("model_storage_path"),
        metadata=request.param.get("model_metadata"),
    )


@pytest.fixture()
def model_registry_operator_pod(admin_client: DynamicClient) -> Generator[Pod, Any, Any]:
    """Get the model registry operator pod."""
    yield wait_for_pods_by_labels(
        admin_client=admin_client,
        namespace=py_config["applications_namespace"],
        label_selector=f"{Labels.OpenDataHubIo.NAME}={MR_OPERATOR_NAME}",
        expected_num_pods=1,
    )[0]


@pytest.fixture()
def model_registry_instance_pod(admin_client: DynamicClient) -> Generator[Pod, Any, Any]:
    """Get the model registry instance pod."""
    yield wait_for_pods_by_labels(
        admin_client=admin_client,
        namespace=py_config["model_registry_namespace"],
        label_selector=f"app={MR_INSTANCE_NAME}",
        expected_num_pods=1,
    )[0]


@pytest.fixture(scope="class")
def is_model_registry_oauth(request: FixtureRequest) -> bool:
    return getattr(request, "param", {}).get("use_oauth_proxy", True)


@pytest.fixture(scope="session")
def api_server_url(admin_client: DynamicClient) -> str:
    infrastructure = Infrastructure(client=admin_client, name="cluster", ensure_exists=True)
    return infrastructure.instance.status.apiServerURL
