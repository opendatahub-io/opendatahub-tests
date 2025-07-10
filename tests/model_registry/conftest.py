from contextlib import ExitStack

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

from tests.model_registry.constants import (
    MR_OPERATOR_NAME,
    MR_INSTANCE_BASE_NAME,
    DB_BASE_RESOURCES_NAME,
    MODEL_REGISTRY_DB_SECRET_STR_DATA,
    OAUTH_PROXY_CONFIG_DICT,
    ISTIO_CONFIG_DICT,
    DB_RESOURCE_NAME,
    MR_INSTANCE_NAME,
)
from tests.model_registry.rest_api.utils import ModelRegistryV1Alpha1
from utilities.constants import Labels, Protocols
from tests.model_registry.utils import (
    get_endpoint_from_mr_service,
    get_mr_service_by_label,
    wait_for_pods_running,
    get_mr_service_objects,
    get_mr_pvc_objects,
    get_mr_secret_objects,
    get_mr_deployment_objects,
    get_model_registry_objects,
)
from utilities.constants import DscComponents
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
    request: pytest.FixtureRequest,
    pytestconfig: Config,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    teardown_resources: bool,
    is_model_registry_oauth: bool,
) -> Generator[list[Service], Any, Any]:
    num_resources = getattr(request, "param", {}).get("num", 1)
    if pytestconfig.option.post_upgrade:
        mr_db_service = Service(name=DB_RESOURCE_NAME, namespace=model_registry_namespace, ensure_exists=True)
        yield [mr_db_service]
        mr_db_service.delete(wait=True)
    else:
        services = get_mr_service_objects(
            client=admin_client,
            namespace=model_registry_namespace,
            base_name=DB_BASE_RESOURCES_NAME,
            num=num_resources,
            teardown_resources=teardown_resources,
        )
        with ExitStack() as stack:
            mr_services = [stack.enter_context(service) for service in services]
            yield mr_services


@pytest.fixture(scope="class")
def model_registry_db_pvc(
    request: pytest.FixtureRequest,
    pytestconfig: Config,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    teardown_resources: bool,
    is_model_registry_oauth: bool,
) -> Generator[list[PersistentVolumeClaim], Any, Any]:
    num_resources = getattr(request, "param", {}).get("num", 1)
    if pytestconfig.option.post_upgrade:
        mr_db_pvc = PersistentVolumeClaim(
            name=DB_BASE_RESOURCES_NAME, namespace=model_registry_namespace, ensure_exists=True
        )
        yield [mr_db_pvc]
        mr_db_pvc.delete(wait=True)
    else:
        pvcs = get_mr_pvc_objects(
            client=admin_client,
            namespace=model_registry_namespace,
            base_name=DB_BASE_RESOURCES_NAME,
            num=num_resources,
            teardown_resources=teardown_resources,
        )
        with ExitStack() as stack:
            mr_pvcs = [stack.enter_context(pvc) for pvc in pvcs]
            yield mr_pvcs


@pytest.fixture(scope="class")
def model_registry_db_secret(
    request: pytest.FixtureRequest,
    pytestconfig: Config,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    teardown_resources: bool,
    is_model_registry_oauth: bool,
) -> Generator[list[Secret], Any, Any]:
    num_resources = getattr(request, "param", {}).get("num", 1)
    if pytestconfig.option.post_upgrade:
        mr_db_secret = Secret(name=DB_BASE_RESOURCES_NAME, namespace=model_registry_namespace, ensure_exists=True)
        yield [mr_db_secret]
        mr_db_secret.delete(wait=True)
    else:
        secrets = get_mr_secret_objects(
            client=admin_client,
            namespace=model_registry_namespace,
            base_name=DB_BASE_RESOURCES_NAME,
            num=num_resources,
            teardown_resources=teardown_resources,
        )
        with ExitStack() as stack:
            mr_secrets = [stack.enter_context(secret) for secret in secrets]
            yield mr_secrets


@pytest.fixture(scope="class")
def model_registry_db_deployment(
    request: pytest.FixtureRequest,
    pytestconfig: Config,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    model_registry_db_secret: list[Secret],
    model_registry_db_pvc: list[PersistentVolumeClaim],
    model_registry_db_service: list[Service],
    teardown_resources: bool,
    is_model_registry_oauth: bool,
) -> Generator[list[Deployment], Any, Any]:
    num_resources = getattr(request, "param", {}).get("num", 1)
    if pytestconfig.option.post_upgrade:
        db_deployment = Deployment(name=DB_BASE_RESOURCES_NAME, namespace=model_registry_namespace, ensure_exists=True)
        yield [db_deployment]
        db_deployment.delete(wait=True)
    else:
        deployments = get_mr_deployment_objects(
            client=admin_client,
            namespace=model_registry_namespace,
            base_name=DB_BASE_RESOURCES_NAME,
            num=num_resources,
            teardown_resources=teardown_resources,
            mr_secrets=model_registry_db_secret,
        )
        with ExitStack() as stack:
            mr_deployments = [stack.enter_context(deployment) for deployment in deployments]
            for deployment in mr_deployments:
                deployment.wait_for_replicas(deployed=True)
            yield deployments


@pytest.fixture(scope="class")
def model_registry_instance(
    request: pytest.FixtureRequest,
    pytestconfig: Config,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    model_registry_db_deployment: list[Deployment],
    teardown_resources: bool,
    is_model_registry_oauth: bool,
) -> Generator[list[Any], Any, Any]:
    """Creates a model registry instance with oauth proxy configuration."""
    param = getattr(request, "param", {})
    num_resources = param.get("num", 1)

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
        mr_objects = get_model_registry_objects(
            client=admin_client,
            namespace=model_registry_namespace,
            base_name=MR_INSTANCE_BASE_NAME,
            num=num_resources,
            mr_class_name=mr_class_name,
            teardown_resources=teardown_resources,
            istio_config=istio_config,
            oauth_config=oauth_config,
            params=param,
            deployments=model_registry_db_deployment,
        )
        with ExitStack() as stack:
            mr_instances = [stack.enter_context(mr_obj) for mr_obj in mr_objects]
            for mr_instance in mr_instances:
                mr_instance.wait_for_condition(condition="Available", status="True")
                mr_instance.wait_for_condition(condition="OAuthProxyAvailable", status="True")
                wait_for_pods_running(
                    admin_client=admin_client, namespace_name=model_registry_namespace, number_of_consecutive_checks=6
                )
            yield mr_instances


@pytest.fixture(scope="class")
def model_registry_instance_service(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    model_registry_instance: list[ModelRegistry],
) -> list[Service]:
    """
    Get the service for the regular model registry instance.
    Args:
        admin_client: The admin client
        model_registry_namespace: The namespace where the model registry is deployed
        model_registry_instance: The model registry instance to get the service for
    Returns:
        Service: The service for the model registry instance
    """
    return [
        get_mr_service_by_label(
            client=admin_client, namespace_name=model_registry_namespace, mr_instance=model_registry
        )
        for model_registry in model_registry_instance
    ]


@pytest.fixture(scope="class")
def model_registry_instance_rest_endpoint(
    model_registry_instance_service: list[Service],
) -> list[str]:
    """
    Get the REST endpoint for the model registry instance.
    Args:
        model_registry_instance_service: The service for the model registry instance
    Returns:
        str: The REST endpoint for the model registry instance
    """
    return [get_endpoint_from_mr_service(svc=svc, protocol=Protocols.REST) for svc in model_registry_instance_service]


@pytest.fixture(scope="class")
def updated_dsc_component_state_scope_class(
    pytestconfig: Config,
    request: FixtureRequest,
    dsc_resource: DataScienceCluster,
    admin_client: DynamicClient,
    teardown_resources: bool,
    is_model_registry_oauth: bool,
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
) -> list[ModelRegistryClient]:
    """
    Get a client for the model registry instance.
    Args:
        request: The pytest request object
        current_client_token: The current client token
    Returns:
        ModelRegistryClient: A client for the model registry instance
    """
    mr_clients = []
    for rest_endpoint in model_registry_instance_rest_endpoint:
        server, port = rest_endpoint.split(":")
        mr_clients.append(
            ModelRegistryClient(
                server_address=f"{Protocols.HTTPS}://{server}",
                port=int(port),
                author="opendatahub-test",
                user_token=current_client_token,
                is_secure=False,
            )
        )
    return mr_clients


@pytest.fixture(scope="class")
def registered_model(request: FixtureRequest, model_registry_client: list[ModelRegistryClient]) -> RegisteredModel:
    return model_registry_client[0].register_model(
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
        label_selector=f"app={MR_INSTANCE_BASE_NAME}",
        expected_num_pods=1,
    )[0]


@pytest.fixture()
def model_registry_db_instance_pod(admin_client: DynamicClient) -> Generator[Pod, Any, Any]:
    """Get the model registry instance pod."""
    yield wait_for_pods_by_labels(
        admin_client=admin_client,
        namespace=py_config["model_registry_namespace"],
        label_selector=f"name={DB_RESOURCE_NAME}",
        expected_num_pods=1,
    )[0]


@pytest.fixture()
def set_mr_db_dirty(model_registry_db_instance_pod: Pod) -> int:
    """Set the model registry database dirty and return the latest migration version"""
    output = model_registry_db_instance_pod.execute(
        command=[
            "mysql",
            "-u",
            MODEL_REGISTRY_DB_SECRET_STR_DATA["database-user"],
            f"-p{MODEL_REGISTRY_DB_SECRET_STR_DATA['database-password']}",
            "-e",
            "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1;",
            MODEL_REGISTRY_DB_SECRET_STR_DATA["database-name"],
        ]
    )
    latest_migration_version = int(output.strip().split()[1])
    model_registry_db_instance_pod.execute(
        command=[
            "mysql",
            "-u",
            MODEL_REGISTRY_DB_SECRET_STR_DATA["database-user"],
            f"-p{MODEL_REGISTRY_DB_SECRET_STR_DATA['database-password']}",
            "-e",
            f"UPDATE schema_migrations SET dirty = 1 WHERE version = {latest_migration_version};",
            MODEL_REGISTRY_DB_SECRET_STR_DATA["database-name"],
        ]
    )
    return latest_migration_version


@pytest.fixture()
def delete_mr_deployment() -> None:
    """Delete the model registry deployment"""
    mr_deployment = Deployment(
        name=MR_INSTANCE_NAME, namespace=py_config["model_registry_namespace"], ensure_exists=True
    )
    mr_deployment.delete(wait=True)


@pytest.fixture(scope="class")
def is_model_registry_oauth(request: FixtureRequest) -> bool:
    return getattr(request, "param", {}).get("use_oauth_proxy", True)


@pytest.fixture(scope="session")
def api_server_url(admin_client: DynamicClient) -> str:
    infrastructure = Infrastructure(client=admin_client, name="cluster", ensure_exists=True)
    return infrastructure.instance.status.apiServerURL


@pytest.fixture(scope="class")
def model_registry_rest_url(model_registry_instance_rest_endpoint: list[str]) -> list[str]:
    # address and port need to be split in the client instantiation
    return [f"{Protocols.HTTPS}://{rest_url}" for rest_url in model_registry_instance_rest_endpoint]


@pytest.fixture(scope="class")
def model_registry_rest_headers(current_client_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {current_client_token}",
        "accept": "application/json",
        "Content-Type": "application/json",
    }
