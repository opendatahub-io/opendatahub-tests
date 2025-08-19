from contextlib import ExitStack

import pytest
import shlex
import subprocess
import os
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from pytest import Config
from pyhelper_utils.shell import run_command
from typing import Generator, Any, List, Dict

from ocp_resources.config_map import ConfigMap
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.namespace import Namespace
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
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
from tests.model_registry.utils import generate_namespace_name
from utilities.general import generate_random_name

from tests.model_registry.constants import (
    MR_OPERATOR_NAME,
    MR_INSTANCE_BASE_NAME,
    DB_BASE_RESOURCES_NAME,
    DB_RESOURCE_NAME,
    MR_INSTANCE_NAME,
)
from utilities.constants import Labels, Protocols
from tests.model_registry.utils import (
    get_endpoint_from_mr_service,
    get_mr_service_by_label,
    wait_for_pods_running,
    get_model_registry_objects,
    get_model_registry_metadata_resources,
)
from utilities.constants import DscComponents
from model_registry import ModelRegistry as ModelRegistryClient
from utilities.general import wait_for_pods_by_labels
from utilities.infra import get_data_science_cluster

DEFAULT_TOKEN_DURATION = "10m"
LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="session")
def model_registry_namespace(updated_dsc_component_state_scope_session: DataScienceCluster) -> str:
    return updated_dsc_component_state_scope_session.instance.spec.components.modelregistry.registriesNamespace


@pytest.fixture(scope="class")
def model_registry_instance(
    request: pytest.FixtureRequest,
    pytestconfig: Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
    model_registry_metadata_db_resources: dict[Any, Any],
    model_registry_namespace: str,
) -> Generator[list[Any], Any, Any]:
    """Creates a model registry instance with oauth proxy configuration."""
    param = getattr(request, "param", {})
    if pytestconfig.option.post_upgrade:
        mr_instance = ModelRegistry(name=MR_INSTANCE_NAME, namespace=model_registry_namespace, ensure_exists=True)
        yield [mr_instance]
        mr_instance.delete(wait=True)
    else:
        LOGGER.warning("Requested Oauth Proxy configuration:")
        mr_objects = get_model_registry_objects(
            client=admin_client,
            namespace=model_registry_namespace,
            base_name=MR_INSTANCE_BASE_NAME,
            num=param.get("num_resources", 1),
            teardown_resources=teardown_resources,
            params=param,
            db_backend=param.get("db_name", "mysql"),
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
def model_registry_metadata_db_resources(
    request: FixtureRequest,
    admin_client: DynamicClient,
    pytestconfig: Config,
    teardown_resources: bool,
    model_registry_namespace: str,
) -> Generator[dict[Any, Any], None, None]:
    num_resources = getattr(request, "param", {}).get("num_resources", 1)
    db_backend = getattr(request, "param", {}).get("db_name", "mysql")

    if pytestconfig.option.post_upgrade:
        resources = {
            Secret: [Secret(name=DB_RESOURCE_NAME, namespace=model_registry_namespace, ensure_exists=True)],
            PersistentVolumeClaim: [
                PersistentVolumeClaim(name=DB_RESOURCE_NAME, namespace=model_registry_namespace, ensure_exists=True)
            ],
            Service: [Service(name=DB_RESOURCE_NAME, namespace=model_registry_namespace, ensure_exists=True)],
            ConfigMap: [ConfigMap(name=DB_RESOURCE_NAME, namespace=model_registry_namespace, ensure_exists=True)]
            if db_backend == "mariadb"
            else [],
            Deployment: [Deployment(name=DB_RESOURCE_NAME, namespace=model_registry_namespace, ensure_exists=True)],
        }
        yield resources
        for kind in [Deployment, ConfigMap, Service, PersistentVolumeClaim, Secret]:
            for resource in resources[kind]:
                resource.delete(wait=True)
    else:
        resources_instances = {}
        resources = get_model_registry_metadata_resources(
            base_name=DB_BASE_RESOURCES_NAME,
            namespace=model_registry_namespace,
            num_resources=num_resources,
            db_backend=db_backend,
            teardown_resources=teardown_resources,
            client=admin_client,
        )
        with ExitStack() as stack:
            for kind_name in [Secret, PersistentVolumeClaim, Service, ConfigMap, Deployment]:
                if resources[kind_name]:
                    LOGGER.info(f"Creating {num_resources} {kind_name} resources")
                    resources_instances[kind_name] = [
                        stack.enter_context(resource_obj) for resource_obj in resources[kind_name]
                    ]
            for deployment in resources_instances[Deployment]:
                deployment.wait_for_replicas(deployed=True)
            yield resources_instances


@pytest.fixture(scope="class")
def model_registry_instance_rest_endpoint(admin_client: DynamicClient, model_registry_instance) -> list[str]:
    """
    Get the REST endpoint(s) for the model registry instance.
    """
    # get all the services:
    mr_services = [
        get_mr_service_by_label(client=admin_client, namespace_name=mr_instance.namespace, mr_instance=mr_instance)
        for mr_instance in model_registry_instance
    ]
    if not mr_services:
        raise ResourceNotFoundError("No model registry services found")
    return [get_endpoint_from_mr_service(svc=svc, protocol=Protocols.REST) for svc in mr_services]


@pytest.fixture(scope="session")
def updated_dsc_component_state_scope_session(
    pytestconfig: Config,
    request: FixtureRequest,
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[DataScienceCluster, Any, Any]:
    dsc_resource = get_data_science_cluster(client=admin_client)
    if not teardown_resources or pytestconfig.option.post_upgrade:
        # if we are not tearing down resources or we are in post upgrade, we don't need to do anything
        # the pre_upgrade/post_upgrade fixtures will handle the rest
        yield dsc_resource
    else:
        original_components = dsc_resource.instance.spec.components
        component_patch = {
            DscComponents.MODELREGISTRY: {
                "managementState": DscComponents.ManagementState.MANAGED,
                "registriesNamespace": py_config["model_registry_namespace"],
            },
        }
        LOGGER.info(f"Applying patch {component_patch}")

        with ResourceEditor(patches={dsc_resource: {"spec": {"components": component_patch}}}):
            for component_name in component_patch:
                dsc_resource.wait_for_condition(
                    condition=DscComponents.COMPONENT_MAPPING[component_name], status="True"
                )
            namespace = Namespace(name=py_config["model_registry_namespace"], wait_for_resource=True)
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
                namespace = Namespace(name=py_config["model_registry_namespace"], ensure_exists=True)
                if namespace:
                    namespace.delete(wait=True)


@pytest.fixture(scope="class")
def model_registry_client(
    current_client_token: str,
    model_registry_instance_rest_endpoint: list[str],
) -> list[ModelRegistryClient]:
    """
    Get a client for the model registry instance.
    Args:
        current_client_token: The current client token
        model_registry_instance_rest_endpoint: list of model registry endpoints
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
    if not mr_clients:
        raise ResourceNotFoundError("No model registry clients created")
    return mr_clients


@pytest.fixture(scope="class")
def registered_model(
    request: FixtureRequest, model_registry_client: list[ModelRegistryClient]
) -> Generator[RegisteredModel, None, None]:
    yield model_registry_client[0].register_model(
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


@pytest.fixture(scope="class")
def model_registry_deployment_containers(model_registry_namespace: str) -> list[dict[str, Any]]:
    return Deployment(
        name=MR_INSTANCE_NAME, namespace=model_registry_namespace, ensure_exists=True
    ).instance.spec.template.spec.containers


@pytest.fixture(scope="class")
def model_registry_pod(admin_client: DynamicClient, model_registry_namespace: str) -> Pod:
    mr_pod = list(
        Pod.get(
            dyn_client=admin_client,
            namespace=model_registry_namespace,
            label_selector=f"app={MR_INSTANCE_NAME}",
        )
    )
    assert len(mr_pod) == 1
    return mr_pod[0]


@pytest.fixture(scope="function")
def sa_namespace(request: pytest.FixtureRequest, admin_client: DynamicClient) -> Generator[Namespace, None, None]:
    """
    Creates a namespace
    """
    test_file = os.path.relpath(request.fspath.strpath, start=os.path.dirname(__file__))
    ns_name = generate_namespace_name(file_path=test_file)
    LOGGER.info(f"Creating temporary namespace: {ns_name}")
    with Namespace(client=admin_client, name=ns_name) as ns:
        ns.wait_for_status(status=Namespace.Status.ACTIVE, timeout=120)
        yield ns


@pytest.fixture(scope="function")
def service_account(admin_client: DynamicClient, sa_namespace: Namespace) -> Generator[ServiceAccount, None, None]:
    """
    Creates a ServiceAccount.
    """
    sa_name = generate_random_name(prefix="mr-test-user")
    LOGGER.info(f"Creating ServiceAccount: {sa_name} in namespace {sa_namespace.name}")
    with ServiceAccount(client=admin_client, name=sa_name, namespace=sa_namespace.name, wait_for_resource=True) as sa:
        yield sa


@pytest.fixture(scope="function")
def sa_token(service_account: ServiceAccount) -> str:
    """
    Retrieves a short-lived token for the ServiceAccount using 'oc create token'.
    """
    sa_name = service_account.name
    namespace = service_account.namespace
    LOGGER.info(f"Retrieving token for ServiceAccount: {sa_name} in namespace {namespace}")
    try:
        cmd = f"oc create token {sa_name} -n {namespace} --duration={DEFAULT_TOKEN_DURATION}"
        LOGGER.debug(f"Executing command: {cmd}")
        res, out, err = run_command(command=shlex.split(cmd), verify_stderr=False, check=True, timeout=30)
        token = out.strip()
        if not token:
            raise ValueError("Retrieved token is empty after successful command execution.")

        LOGGER.info(f"Successfully retrieved token for SA '{sa_name}'")
        return token

    except Exception as e:  # Catch all exceptions from the try block
        error_type = type(e).__name__
        log_message = (
            f"Failed during token retrieval for SA '{sa_name}' in namespace '{namespace}'. "
            f"Error Type: {error_type}, Message: {str(e)}"
        )
        if isinstance(e, subprocess.CalledProcessError):
            # Add specific details for CalledProcessError
            # run_command already logs the error if log_errors=True and returncode !=0,
            # but we can add context here.
            stderr_from_exception = e.stderr.strip() if e.stderr else "N/A"
            log_message += f". Exit Code: {e.returncode}. Stderr from exception: {stderr_from_exception}"
        elif isinstance(e, subprocess.TimeoutExpired):
            timeout_value = getattr(e, "timeout", "N/A")
            log_message += f". Command timed out after {timeout_value} seconds."
        elif isinstance(e, FileNotFoundError):
            # This occurs if 'oc' is not found.
            # e.filename usually holds the name of the file that was not found.
            command_not_found = e.filename if hasattr(e, "filename") and e.filename else shlex.split(cmd)[0]
            log_message += f". Command '{command_not_found}' not found. Is it installed and in PATH?"

        LOGGER.error(log_message, exc_info=True)  # exc_info=True adds stack trace to the log
        raise


@pytest.fixture(scope="function")
def mr_access_role(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    sa_namespace: Namespace,
) -> Generator[Role, None, None]:
    """
    Creates the MR Access Role using direct constructor parameters and a context manager.
    """
    role_name = f"registry-user-{MR_INSTANCE_NAME}-{sa_namespace.name[:8]}"
    LOGGER.info(f"Defining Role: {role_name} in namespace {model_registry_namespace}")

    role_rules: List[Dict[str, Any]] = [
        {
            "apiGroups": [""],
            "resources": ["services"],
            "resourceNames": [MR_INSTANCE_NAME],  # Grant access only to the specific MR service object
            "verbs": ["get"],
        }
    ]
    role_labels = {
        "app.kubernetes.io/component": "model-registry-test-rbac",
        "test.opendatahub.io/namespace": sa_namespace.name,
    }

    with Role(
        client=admin_client,
        name=role_name,
        namespace=model_registry_namespace,
        rules=role_rules,
        label=role_labels,
        wait_for_resource=True,
    ) as role:
        yield role


@pytest.fixture(scope="function")
def mr_access_role_binding(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    mr_access_role: Role,
    sa_namespace: Namespace,
) -> Generator[RoleBinding, None, None]:
    """
    Creates the MR Access RoleBinding using direct constructor parameters and a context manager.
    """
    binding_name = f"{mr_access_role.name}-binding"

    LOGGER.info(
        f"Defining RoleBinding: {binding_name} linking Group 'system:serviceaccounts:{sa_namespace.name}' "
        f"to Role '{mr_access_role.name}' in namespace {model_registry_namespace}"
    )
    binding_labels = {
        "app.kubernetes.io/component": "model-registry-test-rbac",
        "test.opendatahub.io/namespace": sa_namespace.name,
    }

    with RoleBinding(
        client=admin_client,
        name=binding_name,
        namespace=model_registry_namespace,
        # Subject parameters
        subjects_kind="Group",
        subjects_name=f"system:serviceaccounts:{sa_namespace.name}",
        subjects_api_group="rbac.authorization.k8s.io",  # This is the default apiGroup for Group kind
        # Role reference parameters
        role_ref_kind=mr_access_role.kind,
        role_ref_name=mr_access_role.name,
        label=binding_labels,
        wait_for_resource=True,
    ) as binding:
        LOGGER.info(f"RoleBinding {binding.name} created successfully.")
        yield binding
        LOGGER.info(f"RoleBinding {binding.name} deletion initiated by context manager.")
