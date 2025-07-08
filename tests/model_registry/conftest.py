import pytest
from pytest import Config
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
from ocp_resources.resource import ResourceEditor

from pytest import FixtureRequest
from simple_logger.logger import get_logger
from kubernetes.dynamic import DynamicClient
from pytest_testconfig import config as py_config
from model_registry.types import RegisteredModel

# Factory fixture imports
from typing import List, Callable
import uuid
from utilities.constants import Annotations
from tests.model_registry.factory_utils import (
    ModelRegistryDBConfig,
    ModelRegistryConfig,
    ModelRegistryDBBundle,
    ModelRegistryInstanceBundle,
    SecretConfig,
    PVCConfig,
    ServiceConfig,
    DeploymentConfig,
    ModelRegistryResourceConfig,
)

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


# =============================================================================
# FACTORY FIXTURES
# =============================================================================


@pytest.fixture(scope="class")
def resource_creation_factory(
    admin_client: DynamicClient,
    model_registry_cleanup_registry: ModelRegistryCleanupRegistry,
) -> Callable[[Any, Any], Any]:
    """A generic factory for creating and cleaning up Kubernetes resources."""

    def _create_resource(resource_class: Any, config: Any, **kwargs: Any) -> Any:
        """Helper to instantiate, create, and register cleanup for a resource."""
        resource_name = config.name
        LOGGER.info(f"Creating {resource_class.__name__}: {resource_name}")

        try:
            resource = resource_class(client=admin_client, name=resource_name, **kwargs)
            resource.create()

            if config.teardown:

                def cleanup() -> None:
                    try:
                        LOGGER.info(f"Cleaning up {resource_class.__name__}: {resource.name}")
                        resource.delete(wait=True)
                    except Exception as e:
                        LOGGER.warning(f"Failed to cleanup {resource_class.__name__} {resource.name}: {e}")

                model_registry_cleanup_registry.register_cleanup(cleanup_func=cleanup)

            return resource
        except Exception as e:
            LOGGER.error(f"Failed to create {resource_class.__name__} {resource_name}: {e}")
            raise

    return _create_resource


@pytest.fixture(scope="class")
def secret_factory(resource_creation_factory: Callable[..., Any]) -> Callable[[SecretConfig], Secret]:
    """Factory for creating Secret resources."""

    def _create_secret(config: SecretConfig) -> Secret:
        return resource_creation_factory(
            resource_class=Secret,
            config=config,
            namespace=config.namespace,
            string_data=config.string_data,
            label=config.labels,
            annotations=config.annotations,
        )

    return _create_secret


@pytest.fixture(scope="class")
def pvc_factory(resource_creation_factory: Callable[..., Any]) -> Callable[[PVCConfig], PersistentVolumeClaim]:
    """Factory for creating PersistentVolumeClaim resources."""

    def _create_pvc(config: PVCConfig) -> PersistentVolumeClaim:
        return resource_creation_factory(
            resource_class=PersistentVolumeClaim,
            config=config,
            namespace=config.namespace,
            accessmodes=config.access_modes,
            size=config.size,
            label=config.labels,
        )

    return _create_pvc


@pytest.fixture(scope="class")
def service_factory(resource_creation_factory: Callable[..., Any]) -> Callable[[ServiceConfig], Service]:
    """Factory for creating Service resources."""

    def _create_service(config: ServiceConfig) -> Service:
        return resource_creation_factory(
            resource_class=Service,
            config=config,
            namespace=config.namespace,
            ports=config.ports,
            selector=config.selector,
            label=config.labels,
            annotations=config.annotations,
        )

    return _create_service


@pytest.fixture(scope="class")
def deployment_factory(resource_creation_factory: Callable[..., Any]) -> Callable[[DeploymentConfig], Deployment]:
    """Factory for creating Deployment resources."""

    def _create_deployment(config: DeploymentConfig) -> Deployment:
        deployment = resource_creation_factory(
            resource_class=Deployment,
            config=config,
            namespace=config.namespace,
            annotations=config.annotations,
            label=config.labels,
            replicas=config.replicas,
            revision_history_limit=config.revision_history_limit,
            selector=config.selector or {"matchLabels": {"name": config.name}},
            strategy=config.strategy or {"type": "Recreate"},
            template=config.template,
        )
        LOGGER.info(f"Waiting for DB deployment to be ready: {config.name}")
        deployment.wait_for_replicas(deployed=True)
        return deployment

    return _create_deployment


@pytest.fixture(scope="class")
def model_registry_resource_factory(
    resource_creation_factory: Callable[..., Any],
) -> Callable[[ModelRegistryResourceConfig], ModelRegistry]:
    """Factory for creating ModelRegistry resources."""

    def _create_model_registry(config: ModelRegistryResourceConfig) -> ModelRegistry:
        mr_class = ModelRegistryV1Alpha1 if config.use_istio else ModelRegistry
        instance = resource_creation_factory(
            resource_class=mr_class,
            config=config,
            namespace=config.namespace,
            label=config.labels,
            grpc=config.grpc_config or {},
            rest=config.rest_config or {},
            istio=config.istio_config,
            oauth_proxy=config.oauth_proxy_config,
            mysql=config.mysql_config,
        )

        if config.wait_for_conditions:
            LOGGER.info(f"Waiting for Model Registry conditions: {config.name}")
            instance.wait_for_condition(condition="Available", status="True")
            if config.use_oauth_proxy:
                instance.wait_for_condition(condition="OAuthProxyAvailable", status="True")

        return instance

    return _create_model_registry


@pytest.fixture(scope="class")
def model_registry_db_factory(
    pytestconfig: Config,
    model_registry_cleanup_registry: ModelRegistryCleanupRegistry,
    secret_factory: Callable[[SecretConfig], Secret],
    pvc_factory: Callable[[PVCConfig], PersistentVolumeClaim],
    service_factory: Callable[[ServiceConfig], Service],
    deployment_factory: Callable[[DeploymentConfig], Deployment],
) -> Generator[Callable[[ModelRegistryDBConfig], ModelRegistryDBBundle], Any, Any]:
    """Factory fixture for creating Model Registry DB bundles."""
    created_bundles: List[ModelRegistryDBBundle] = []

    def create_db_bundle(config: ModelRegistryDBConfig) -> ModelRegistryDBBundle:
        """Create a complete Model Registry DB bundle."""
        unique_suffix = str(uuid.uuid4())[:8]
        db_name = f"{config.name_prefix}-{unique_suffix}"
        labels = config.labels or get_model_registry_db_label_dict(db_resource_name=db_name)

        if pytestconfig.option.post_upgrade:
            # In post-upgrade, we assume resources exist and just wrap them
            service = Service(name=db_name, namespace=config.namespace, ensure_exists=True)
            pvc = PersistentVolumeClaim(name=db_name, namespace=config.namespace, ensure_exists=True)
            secret = Secret(name=db_name, namespace=config.namespace, ensure_exists=True)
            deployment = Deployment(name=db_name, namespace=config.namespace, ensure_exists=True)
        else:
            # Create resources using individual factories
            secret_config = SecretConfig(
                name=db_name,
                namespace=config.namespace,
                string_data={
                    "database-name": config.database_name,
                    "database-user": config.database_user,
                    "database-password": config.database_password,
                },
                labels=labels,
                annotations=config.annotations or MODEL_REGISTRY_DB_SECRET_ANNOTATIONS,
                teardown=config.teardown,
            )
            secret = secret_factory(secret_config)  # noqa: FCN001

            pvc_config = PVCConfig(
                name=db_name,
                namespace=config.namespace,
                access_modes=config.access_mode,
                size=config.storage_size,
                labels=labels,
                teardown=config.teardown,
            )
            pvc = pvc_factory(pvc_config)  # noqa: FCN001

            service_config = ServiceConfig(
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
                labels=labels,
                annotations={
                    "template.openshift.io/expose-uri": (
                        "mysql://{.spec.clusterIP}:{.spec.ports[?(.name==\\mysql\\)].port}"
                    ),
                },
                teardown=config.teardown,
            )
            service = service_factory(service_config)  # noqa: FCN001

            deployment_config = DeploymentConfig(
                name=db_name,
                namespace=config.namespace,
                template=get_model_registry_deployment_template_dict(secret_name=secret.name, resource_name=db_name),
                labels=labels,
                annotations={"template.alpha.openshift.io/wait-for-ready": "true"},
                teardown=config.teardown,
            )
            deployment = deployment_factory(deployment_config)  # noqa: FCN001

        bundle = ModelRegistryDBBundle(service=service, pvc=pvc, secret=secret, deployment=deployment, config=config)
        created_bundles.append(bundle)

        # Note: Cleanup is now handled by the individual resource factories.
        # We don't need to register a cleanup for the bundle itself.

        LOGGER.info(f"Successfully created DB bundle: {db_name}")
        return bundle

    yield create_db_bundle

    # Note: Cleanup is handled by the cleanup registry, not here
    # This ensures proper teardown order relative to DSC patch revert


@pytest.fixture(scope="class")
def model_registry_instance_factory(
    admin_client: DynamicClient,
    pytestconfig: Config,
    model_registry_db_factory: Callable[[ModelRegistryDBConfig], ModelRegistryDBBundle],
    model_registry_resource_factory: Callable[[ModelRegistryResourceConfig], ModelRegistry],
    model_registry_cleanup_registry: ModelRegistryCleanupRegistry,
) -> Generator[Callable[[ModelRegistryConfig], ModelRegistryInstanceBundle], Any, Any]:
    """Factory fixture for creating Model Registry instances."""
    created_instances: List[ModelRegistryInstanceBundle] = []

    def create_instance(config: ModelRegistryConfig) -> ModelRegistryInstanceBundle:
        """Create a complete Model Registry instance bundle."""
        unique_suffix = str(uuid.uuid4())[:8]
        instance_name = f"{config.name}-{unique_suffix}"

        db_bundle = None
        mysql_config_dict = None
        if config.mysql_config:
            db_bundle = model_registry_db_factory(config.mysql_config)  # noqa: FCN001
            mysql_config_dict = db_bundle.get_mysql_config()

        labels = config.labels or MODEL_REGISTRY_STANDARD_LABELS.copy()
        labels.update({
            Annotations.KubernetesIo.NAME: instance_name,
            Annotations.KubernetesIo.INSTANCE: instance_name,
        })

        if pytestconfig.option.post_upgrade:
            mr_class = ModelRegistryV1Alpha1 if config.use_istio else ModelRegistry
            instance = mr_class(name=instance_name, namespace=config.namespace, ensure_exists=True)
        else:
            mr_resource_config = ModelRegistryResourceConfig(
                name=instance_name,
                namespace=config.namespace,
                mysql_config=mysql_config_dict,
                labels=labels,
                grpc_config=config.grpc_config,
                rest_config=config.rest_config,
                oauth_proxy_config=config.oauth_proxy_config or OAUTH_PROXY_CONFIG_DICT,
                istio_config=config.istio_config or ISTIO_CONFIG_DICT,
                use_oauth_proxy=config.use_oauth_proxy,
                use_istio=config.use_istio,
                teardown=config.teardown,
                wait_for_conditions=config.wait_for_conditions,
            )
            instance = model_registry_resource_factory(mr_resource_config)  # noqa: FCN001

        service = None
        rest_endpoint = None
        grpc_endpoint = None
        try:
            service = get_mr_service_by_label(
                client=admin_client, namespace_name=config.namespace, mr_instance=instance
            )
            rest_endpoint = get_endpoint_from_mr_service(svc=service, protocol=Protocols.REST)
            grpc_endpoint = get_endpoint_from_mr_service(svc=service, protocol=Protocols.GRPC)
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

        # Note: Cleanup is handled by the model_registry_resource_factory.

        LOGGER.info(f"Successfully created Model Registry bundle: {instance_name}")
        return bundle

    yield create_instance

    # Note: Cleanup is handled by the cleanup registry.


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


@pytest.fixture(scope="class")
def default_model_registry(
    model_registry_instance_factory: Callable[[ModelRegistryConfig], ModelRegistryInstanceBundle],
    model_registry_namespace: str,
) -> ModelRegistryInstanceBundle:
    """Creates a Model Registry instance with default configuration."""
    config = ModelRegistryConfig(
        name="default-mr",
        namespace=model_registry_namespace,
        mysql_config=ModelRegistryDBConfig(
            name_prefix="default-mr-db",
            namespace=model_registry_namespace,
        ),
    )
    return model_registry_instance_factory(config)  # noqa: FCN001


@pytest.fixture(scope="class")
def oauth_model_registry(
    model_registry_instance_factory: Callable[[ModelRegistryConfig], ModelRegistryInstanceBundle],
    model_registry_namespace: str,
) -> ModelRegistryInstanceBundle:
    """Creates a Model Registry instance with OAuth configuration."""
    config = ModelRegistryConfig(
        name="oauth-mr",
        namespace=model_registry_namespace,
        use_oauth_proxy=True,
        use_istio=False,
        mysql_config=ModelRegistryDBConfig(
            name_prefix="oauth-mr-db",
            namespace=model_registry_namespace,
        ),
    )
    return model_registry_instance_factory(config)  # noqa: FCN001


@pytest.fixture(scope="class")
def istio_model_registry(
    model_registry_instance_factory: Callable[[ModelRegistryConfig], ModelRegistryInstanceBundle],
    model_registry_namespace: str,
) -> ModelRegistryInstanceBundle:
    """Creates a Model Registry instance with Istio configuration."""
    config = ModelRegistryConfig(
        name="istio-mr",
        namespace=model_registry_namespace,
        use_oauth_proxy=False,
        use_istio=True,
        mysql_config=ModelRegistryDBConfig(
            name_prefix="istio-mr-db",
            namespace=model_registry_namespace,
        ),
    )
    return model_registry_instance_factory(config)  # noqa: FCN001


@pytest.fixture(scope="class")
def standalone_db(
    model_registry_db_factory: Callable[[ModelRegistryDBConfig], ModelRegistryDBBundle],
    model_registry_namespace: str,
) -> ModelRegistryDBBundle:
    """Creates a standalone Model Registry DB instance."""
    config = ModelRegistryDBConfig(
        name_prefix="standalone-db",
        namespace=model_registry_namespace,
    )
    return model_registry_db_factory(config)  # noqa: FCN001


@pytest.fixture(scope="class")
def simple_model_registry(
    model_registry_instance_factory: Callable[[ModelRegistryConfig], ModelRegistryInstanceBundle],
    model_registry_namespace: str,
) -> ModelRegistryInstanceBundle:
    """
    Creates a Model Registry instance with minimal configuration.
    """
    config = ModelRegistryConfig(
        name="simple-mr",
        namespace=model_registry_namespace,
        mysql_config=ModelRegistryDBConfig(
            name_prefix="simple-mr-db",
            namespace=model_registry_namespace,
        ),
    )
    return model_registry_instance_factory(config)  # noqa: FCN001


# =============================================================================
# SIMPLE RESOURCE FIXTURES
# =============================================================================


@pytest.fixture(scope="class")
def db_name_1() -> str:
    """Generate a unique name for the first DB instance."""
    return f"db-instance-1-{str(uuid.uuid4())[:8]}"


@pytest.fixture(scope="class")
def db_secret_1(
    secret_factory: Callable[[SecretConfig], Secret], model_registry_namespace: str, db_name_1: str
) -> Secret:
    """Create the first DB secret."""
    labels = get_model_registry_db_label_dict(db_resource_name=db_name_1)
    config = SecretConfig(
        name=db_name_1,
        namespace=model_registry_namespace,
        string_data=MODEL_REGISTRY_DB_SECRET_STR_DATA,
        labels=labels,
        annotations=MODEL_REGISTRY_DB_SECRET_ANNOTATIONS,
    )
    return secret_factory(config)  # noqa: FCN001


@pytest.fixture(scope="class")
def db_pvc_1(
    pvc_factory: Callable[[PVCConfig], PersistentVolumeClaim], model_registry_namespace: str, db_name_1: str
) -> PersistentVolumeClaim:
    """Create the first DB PVC."""
    labels = get_model_registry_db_label_dict(db_resource_name=db_name_1)
    config = PVCConfig(
        name=db_name_1,
        namespace=model_registry_namespace,
        access_modes="ReadWriteOnce",
        size="5Gi",
        labels=labels,
    )
    return pvc_factory(config)  # noqa: FCN001


@pytest.fixture(scope="class")
def db_service_1(
    service_factory: Callable[[ServiceConfig], Service], model_registry_namespace: str, db_name_1: str
) -> Service:
    """Create the first DB service."""
    labels = get_model_registry_db_label_dict(db_resource_name=db_name_1)
    config = ServiceConfig(
        name=db_name_1,
        namespace=model_registry_namespace,
        ports=[{"name": "mysql", "port": 3306, "protocol": "TCP", "targetPort": 3306}],
        selector={"name": db_name_1},
        labels=labels,
    )
    return service_factory(config)  # noqa: FCN001


@pytest.fixture(scope="class")
def db_deployment_1(
    deployment_factory: Callable[[DeploymentConfig], Deployment],
    model_registry_namespace: str,
    db_name_1: str,
    db_secret_1: Secret,
) -> Deployment:
    """Create the first DB deployment."""
    labels = get_model_registry_db_label_dict(db_resource_name=db_name_1)
    config = DeploymentConfig(
        name=db_name_1,
        namespace=model_registry_namespace,
        template=get_model_registry_deployment_template_dict(secret_name=db_secret_1.name, resource_name=db_name_1),
        labels=labels,
    )
    return deployment_factory(config)  # noqa: FCN001


@pytest.fixture(scope="class")
def model_registry_instance_1(
    model_registry_resource_factory: Callable[[ModelRegistryResourceConfig], ModelRegistry],
    model_registry_namespace: str,
    db_deployment_1: Deployment,
    db_secret_1: Secret,
) -> ModelRegistry:
    """Create the first Model Registry instance (default/oauth)."""
    mysql_config = {
        "host": f"{db_deployment_1.name}.{model_registry_namespace}.svc.cluster.local",
        "database": MODEL_REGISTRY_DB_SECRET_STR_DATA["database-name"],
        "passwordSecret": {"key": "database-password", "name": db_secret_1.name},
        "port": 3306,
        "username": MODEL_REGISTRY_DB_SECRET_STR_DATA["database-user"],
    }
    mr_name = f"mr-instance-1-{str(uuid.uuid4())[:8]}"
    labels = MODEL_REGISTRY_STANDARD_LABELS.copy()
    labels.update({Annotations.KubernetesIo.NAME: mr_name, Annotations.KubernetesIo.INSTANCE: mr_name})
    config = ModelRegistryResourceConfig(
        name=mr_name,
        namespace=model_registry_namespace,
        mysql_config=mysql_config,
        labels=labels,
        use_oauth_proxy=True,
        oauth_proxy_config=OAUTH_PROXY_CONFIG_DICT,
    )
    return model_registry_resource_factory(config)  # noqa: FCN001


# --- Instance 2 Resources ---


@pytest.fixture(scope="class")
def db_name_2() -> str:
    """Generate a unique name for the second DB instance."""
    return f"db-instance-2-{str(uuid.uuid4())[:8]}"


@pytest.fixture(scope="class")
def db_secret_2(
    secret_factory: Callable[[SecretConfig], Secret], model_registry_namespace: str, db_name_2: str
) -> Secret:
    """Create the second DB secret."""
    labels = get_model_registry_db_label_dict(db_resource_name=db_name_2)
    config = SecretConfig(
        name=db_name_2,
        namespace=model_registry_namespace,
        string_data=MODEL_REGISTRY_DB_SECRET_STR_DATA,
        labels=labels,
        annotations=MODEL_REGISTRY_DB_SECRET_ANNOTATIONS,
    )
    return secret_factory(config)  # noqa: FCN001


@pytest.fixture(scope="class")
def db_pvc_2(
    pvc_factory: Callable[[PVCConfig], PersistentVolumeClaim], model_registry_namespace: str, db_name_2: str
) -> PersistentVolumeClaim:
    """Create the second DB PVC."""
    labels = get_model_registry_db_label_dict(db_resource_name=db_name_2)
    config = PVCConfig(
        name=db_name_2,
        namespace=model_registry_namespace,
        access_modes="ReadWriteOnce",
        size="5Gi",
        labels=labels,
    )
    return pvc_factory(config)  # noqa: FCN001


@pytest.fixture(scope="class")
def db_service_2(
    service_factory: Callable[[ServiceConfig], Service], model_registry_namespace: str, db_name_2: str
) -> Service:
    """Create the second DB service."""
    labels = get_model_registry_db_label_dict(db_resource_name=db_name_2)
    config = ServiceConfig(
        name=db_name_2,
        namespace=model_registry_namespace,
        ports=[{"name": "mysql", "port": 3306, "protocol": "TCP", "targetPort": 3306}],
        selector={"name": db_name_2},
        labels=labels,
    )
    return service_factory(config)  # noqa: FCN001


@pytest.fixture(scope="class")
def db_deployment_2(
    deployment_factory: Callable[[DeploymentConfig], Deployment],
    model_registry_namespace: str,
    db_name_2: str,
    db_secret_2: Secret,
) -> Deployment:
    """Create the second DB deployment."""
    labels = get_model_registry_db_label_dict(db_resource_name=db_name_2)
    config = DeploymentConfig(
        name=db_name_2,
        namespace=model_registry_namespace,
        template=get_model_registry_deployment_template_dict(secret_name=db_secret_2.name, resource_name=db_name_2),
        labels=labels,
    )
    return deployment_factory(config)  # noqa: FCN001


@pytest.fixture(scope="class")
def model_registry_instance_2(
    model_registry_resource_factory: Callable[[ModelRegistryResourceConfig], ModelRegistry],
    model_registry_namespace: str,
    db_deployment_2: Deployment,
    db_secret_2: Secret,
) -> ModelRegistry:
    """Create the second Model Registry instance (istio)."""
    mysql_config = {
        "host": f"{db_deployment_2.name}.{model_registry_namespace}.svc.cluster.local",
        "database": MODEL_REGISTRY_DB_SECRET_STR_DATA["database-name"],
        "passwordSecret": {"key": "database-password", "name": db_secret_2.name},
        "port": 3306,
        "username": MODEL_REGISTRY_DB_SECRET_STR_DATA["database-user"],
    }
    mr_name = f"mr-instance-2-{str(uuid.uuid4())[:8]}"
    labels = MODEL_REGISTRY_STANDARD_LABELS.copy()
    labels.update({Annotations.KubernetesIo.NAME: mr_name, Annotations.KubernetesIo.INSTANCE: mr_name})
    config = ModelRegistryResourceConfig(
        name=mr_name,
        namespace=model_registry_namespace,
        mysql_config=mysql_config,
        labels=labels,
        use_oauth_proxy=True,
        oauth_proxy_config=OAUTH_PROXY_CONFIG_DICT,
    )
    return model_registry_resource_factory(config)  # noqa: FCN001
