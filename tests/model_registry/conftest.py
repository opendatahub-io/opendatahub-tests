import pytest
import re
import random
import string
import subprocess
import schemathesis
import shlex
from typing import Generator, Any, List, Dict
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.namespace import Namespace
from ocp_resources.service import Service
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from ocp_resources.service_account import ServiceAccount
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding

from ocp_resources.model_registry import ModelRegistry
import schemathesis.schemas
from schemathesis.specs.openapi.schemas import BaseOpenAPISchema
from schemathesis.generation.stateful.state_machine import APIStateMachine
from schemathesis.core.transport import Response
from schemathesis.generation.case import Case
from ocp_resources.resource import ResourceEditor

from pytest import FixtureRequest
from simple_logger.logger import get_logger
from kubernetes.dynamic import DynamicClient
from pytest_testconfig import config as py_config
from pyhelper_utils.shell import run_command
from model_registry.types import RegisteredModel
from tests.model_registry.constants import (
    MR_OPERATOR_NAME,
    MR_INSTANCE_NAME,
    ISTIO_CONFIG_DICT,
    DB_RESOURCES_NAME,
    MODEL_REGISTRY_DB_SECRET_STR_DATA,
    MODEL_REGISTRY_DB_SECRET_ANNOTATIONS,
)
from tests.model_registry.utils import (
    get_endpoint_from_mr_service,
    get_mr_service_by_label,
    get_model_registry_deployment_template_dict,
    get_model_registry_db_label_dict,
    wait_for_pods_running,
)
from utilities.constants import Annotations, Protocols, DscComponents
from model_registry import ModelRegistry as ModelRegistryClient


LOGGER = get_logger(name=__name__)
DEFAULT_TOKEN_DURATION = "10m"


@pytest.fixture(scope="class")
def model_registry_namespace(updated_dsc_component_state_scope_class: DataScienceCluster) -> str:
    return updated_dsc_component_state_scope_class.instance.spec.components.modelregistry.registriesNamespace


@pytest.fixture(scope="class")
def model_registry_db_service(
    admin_client: DynamicClient, model_registry_namespace: str
) -> Generator[Service, Any, Any]:
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
    ) as mr_db_service:
        yield mr_db_service


@pytest.fixture(scope="class")
def model_registry_db_pvc(
    admin_client: DynamicClient, model_registry_namespace: str
) -> Generator[PersistentVolumeClaim, Any, Any]:
    with PersistentVolumeClaim(
        accessmodes="ReadWriteOnce",
        name=DB_RESOURCES_NAME,
        namespace=model_registry_namespace,
        client=admin_client,
        size="5Gi",
        label=get_model_registry_db_label_dict(db_resource_name=DB_RESOURCES_NAME),
    ) as pvc:
        yield pvc


@pytest.fixture(scope="class")
def model_registry_db_secret(
    admin_client: DynamicClient,
    model_registry_namespace: str,
) -> Generator[Secret, Any, Any]:
    with Secret(
        client=admin_client,
        name=DB_RESOURCES_NAME,
        namespace=model_registry_namespace,
        string_data=MODEL_REGISTRY_DB_SECRET_STR_DATA,
        label=get_model_registry_db_label_dict(db_resource_name=DB_RESOURCES_NAME),
        annotations=MODEL_REGISTRY_DB_SECRET_ANNOTATIONS,
    ) as mr_db_secret:
        yield mr_db_secret


@pytest.fixture(scope="class")
def model_registry_db_deployment(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    model_registry_db_secret: Secret,
    model_registry_db_pvc: PersistentVolumeClaim,
    model_registry_db_service: Service,
) -> Generator[Deployment, Any, Any]:
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
    ) as mr_db_deployment:
        mr_db_deployment.wait_for_replicas(deployed=True)
        yield mr_db_deployment


@pytest.fixture(scope="class")
def model_registry_instance(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    model_registry_db_deployment: Deployment,
    model_registry_db_secret: Secret,
    model_registry_db_service: Service,
) -> Generator[ModelRegistry, Any, Any]:
    with ModelRegistry(
        name=MR_INSTANCE_NAME,
        namespace=model_registry_namespace,
        label={
            Annotations.KubernetesIo.NAME: MR_INSTANCE_NAME,
            Annotations.KubernetesIo.INSTANCE: MR_INSTANCE_NAME,
            Annotations.KubernetesIo.PART_OF: MR_OPERATOR_NAME,
            Annotations.KubernetesIo.CREATED_BY: MR_OPERATOR_NAME,
        },
        grpc={},
        rest={},
        istio=ISTIO_CONFIG_DICT,
        mysql={
            "host": f"{model_registry_db_deployment.name}.{model_registry_db_deployment.namespace}.svc.cluster.local",
            "database": model_registry_db_secret.string_data["database-name"],
            "passwordSecret": {"key": "database-password", "name": DB_RESOURCES_NAME},
            "port": 3306,
            "skipDBCreation": False,
            "username": model_registry_db_secret.string_data["database-user"],
        },
        wait_for_resource=True,
    ) as mr:
        mr.wait_for_condition(condition="Available", status="True")
        yield mr


@pytest.fixture(scope="class")
def model_registry_instance_service(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    model_registry_instance: ModelRegistry,
) -> Service:
    return get_mr_service_by_label(
        client=admin_client, ns=Namespace(name=model_registry_namespace), mr_instance=model_registry_instance
    )


@pytest.fixture(scope="class")
def model_registry_instance_rest_endpoint(
    model_registry_instance_service: Service,
) -> str:
    return get_endpoint_from_mr_service(svc=model_registry_instance_service, protocol=Protocols.REST)


@pytest.fixture(scope="class")
def generated_schema(model_registry_instance_rest_endpoint: str) -> BaseOpenAPISchema:
    schema = schemathesis.openapi.from_url(
        url="https://raw.githubusercontent.com/kubeflow/model-registry/main/api/openapi/model-registry.yaml"
    )
    schema.configure(base_url=f"https://{model_registry_instance_rest_endpoint}/")
    return schema


@pytest.fixture
def state_machine(generated_schema: BaseOpenAPISchema, current_client_token: str) -> APIStateMachine:
    BaseAPIWorkflow = generated_schema.as_state_machine()

    class APIWorkflow(BaseAPIWorkflow):  # type: ignore
        headers: dict[str, str]

        def setup(self) -> None:
            self.headers = {"Authorization": f"Bearer {current_client_token}", "Content-Type": "application/json"}

        # these kwargs are passed to requests.request()
        def get_call_kwargs(self, case: Case) -> dict[str, Any]:
            return {"verify": False, "headers": self.headers}

        def after_call(self, response: Response, case: Case) -> None:
            LOGGER.info(f"{case.method} {case.path} -> {response.status_code}")

    return APIWorkflow


@pytest.fixture(scope="class")
def updated_dsc_component_state_scope_class(
    request: FixtureRequest, dsc_resource: DataScienceCluster, admin_client: DynamicClient
) -> Generator[DataScienceCluster, Any, Any]:
    original_components = dsc_resource.instance.spec.components
    with ResourceEditor(patches={dsc_resource: {"spec": {"components": request.param["component_patch"]}}}):
        for component_name in request.param["component_patch"]:
            dsc_resource.wait_for_condition(condition=DscComponents.COMPONENT_MAPPING[component_name], status="True")
        if request.param["component_patch"].get(DscComponents.MODELREGISTRY):
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

    for component_name, value in request.param["component_patch"].items():
        LOGGER.info(f"Waiting for component {component_name} to be updated.")
        if original_components[component_name]["managementState"] == DscComponents.ManagementState.MANAGED:
            dsc_resource.wait_for_condition(condition=DscComponents.COMPONENT_MAPPING[component_name], status="True")
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
def model_registry_client(current_client_token: str, model_registry_instance_rest_endpoint: str) -> ModelRegistryClient:
    # address and port need to be split in the client instantiation
    server, port = model_registry_instance_rest_endpoint.split(":")
    return ModelRegistryClient(
        server_address=f"{Protocols.HTTPS}://{server}",
        port=port,
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
def model_registry_operator_pod(admin_client: DynamicClient) -> Pod:
    model_registry_operator_pods = [
        pod
        for pod in Pod.get(dyn_client=admin_client, namespace=py_config["applications_namespace"])
        if re.match(MR_OPERATOR_NAME, pod.name)
    ]
    if not model_registry_operator_pods:
        raise ResourceNotFoundError("Model registry operator pod not found")
    return model_registry_operator_pods[0]


# --- Fixture Helper Function ---


def generate_random_name(prefix: str = "test", length: int = 8) -> str:
    """Generates a random string for resource names."""
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=length))
    return f"{prefix}-{suffix}"


# --- Service Account and Namespace Fixtures (Function Scoped for Isolation) ---


@pytest.fixture(scope="function")
def sa_namespace(admin_client: DynamicClient) -> Generator[Namespace, None, None]:
    """
    Creates a temporary namespace using a context manager for automatic cleanup.
    Function scope ensures a fresh namespace for each test needing it.
    """
    ns_name = generate_random_name(prefix="mr-rbac-test-ns")
    LOGGER.info(f"Creating temporary namespace: {ns_name}")
    # Use context manager for creation and deletion
    with Namespace(client=admin_client, name=ns_name) as ns:
        try:
            ns.wait_for_status(status=Namespace.Status.ACTIVE, timeout=120)
            LOGGER.info(f"Namespace {ns_name} is active.")
            yield ns
            # Cleanup happens automatically when exiting 'with' block
            LOGGER.info(f"Namespace {ns_name} deletion initiated by context manager.")
            # Add a final wait within the fixture if immediate confirmation is needed,
            # but context manager handles the delete call. Let's rely on the manager.
            # Consider adding ns.wait_deleted(timeout=180) here if needed AFTER yield returns.
        except Exception:
            LOGGER.error(f"Timeout waiting for namespace {ns_name} to become active.")
            pytest.fail(f"Namespace {ns_name} failed to become active.")


@pytest.fixture(scope="function")
def service_account(admin_client: DynamicClient, sa_namespace: Namespace) -> Generator[ServiceAccount, None, None]:
    """
    Creates a ServiceAccount within the temporary namespace using a context manager.
    Function scope ensures it's tied to the lifetime of sa_namespace for that test.
    """
    sa_name = generate_random_name(prefix="mr-test-user")
    LOGGER.info(f"Creating ServiceAccount: {sa_name} in namespace {sa_namespace.name}")
    # Use context manager for creation and deletion
    with ServiceAccount(client=admin_client, name=sa_name, namespace=sa_namespace.name) as sa:
        try:
            sa.wait(timeout=60)  # Wait for SA object to exist
            LOGGER.info(f"ServiceAccount {sa_name} created.")
            yield sa
            # Cleanup happens automatically when exiting 'with' block
            LOGGER.info(f"ServiceAccount {sa_name} deletion initiated by context manager.")
        except Exception:
            LOGGER.error(f"Timeout waiting for ServiceAccount {sa_name} to be created.")
            pytest.fail(f"ServiceAccount {sa_name} failed to be created.")


@pytest.fixture(scope="function")
def sa_token(service_account: ServiceAccount) -> str:  # type: ignore[return]
    """
    Retrieves a short-lived token for the ServiceAccount using 'oc create token'.
    Function scope because token is temporary and tied to the SA for that test.
    """
    sa_name = service_account.name
    namespace = service_account.namespace  # Get namespace name from SA object
    LOGGER.info(f"Retrieving token for ServiceAccount: {sa_name} in namespace {namespace}")
    # (Keep the subprocess logic from previous version - it's appropriate here)
    try:
        cmd = f"oc create token {sa_name} -n {namespace} --duration={DEFAULT_TOKEN_DURATION}"
        LOGGER.debug(f"Executing command: {cmd}")
        res, out, err = run_command(command=shlex.split(cmd), verify_stderr=False, check=True)
        token = out.strip()
        if not token:
            pytest.fail(f"Retrieved token is empty for SA {sa_name} in {namespace}.")
        LOGGER.info(f"Successfully retrieved token for SA {sa_name}")
        return token
    except subprocess.CalledProcessError as e:
        LOGGER.error(f"Failed to create token for SA {sa_name} ns {namespace}: {e.stderr}")
        pytest.fail(f"Failed to create token for SA {sa_name}: {e.stderr}")
    except subprocess.TimeoutExpired:
        LOGGER.error(f"Timeout creating token for SA {sa_name} ns {namespace}")
        pytest.fail(f"Timeout creating token for SA {sa_name}")
    except Exception as e:
        LOGGER.error(
            f"An unexpected error occurred during token retrieval for SA {sa_name} ns {namespace}: {e}", exc_info=True
        )
        pytest.fail(f"Unexpected error getting token for SA {sa_name}: {e}")


# --- RBAC Fixtures (Using Context Managers, Function Scoped) ---


@pytest.fixture(scope="function")
def mr_access_role(
    admin_client: DynamicClient,
    model_registry_namespace: str,  # Existing fixture from main conftest
    sa_namespace: Namespace,  # Used for unique naming
) -> Generator[Role, None, None]:
    """
    Creates the MR Access Role using direct constructor parameters and a context manager.
    """
    role_name = f"registry-user-{MR_INSTANCE_NAME}-{sa_namespace.name[:8]}"
    LOGGER.info(f"Defining Role: {role_name} in namespace {model_registry_namespace}")

    # Define rules directly as required by the Role constructor's 'rules' parameter
    role_rules: List[Dict[str, Any]] = [
        {
            "apiGroups": [""],  # Core API group
            "resources": ["services"],  # As per last refinement for REST access
            "resourceNames": [MR_INSTANCE_NAME],  # Grant access only to the specific MR service object
            "verbs": ["get"],
        }
    ]

    # Define labels, to be passed via **kwargs
    role_labels = {
        "app.kubernetes.io/component": "model-registry-test-rbac",
        "test.opendatahub.io/namespace": sa_namespace.name,
    }

    LOGGER.info(f"Attempting to create Role: {role_name} with rules and labels.")
    # Use context manager for creation and deletion
    # Pass rules and labels directly
    with Role(
        client=admin_client,
        name=role_name,
        namespace=model_registry_namespace,
        rules=role_rules,
        label=role_labels,  # Pass labels via kwargs
    ) as role:
        try:
            role.wait(timeout=60)  # Wait for role object to exist
            LOGGER.info(f"Role {role.name} created successfully.")
            yield role
            LOGGER.info(f"Role {role.name} deletion initiated by context manager.")
        except Exception as e:  # Catch other potential errors during Role instantiation or wait
            LOGGER.error(f"Error during Role {role_name} creation or wait: {e}", exc_info=True)
            pytest.fail(f"Failed during Role {role_name} creation: {e}")


@pytest.fixture(scope="function")
def mr_access_role_binding(
    admin_client: DynamicClient,
    model_registry_namespace: str,  # Existing fixture from main conftest
    mr_access_role: Role,  # Depend on the role fixture to get its name
    sa_namespace: Namespace,  # The namespace containing the test SA
) -> Generator[RoleBinding, None, None]:
    """
    Creates the MR Access RoleBinding using direct constructor parameters and a context manager.
    """
    binding_name = f"{mr_access_role.name}-binding"  # Simplify name slightly, role name is already unique
    role_name_ref = mr_access_role.name  # Get the actual name from the created Role object

    LOGGER.info(
        f"Defining RoleBinding: {binding_name} linking Group 'system:serviceaccounts:{sa_namespace.name}' "
        f"to Role '{role_name_ref}' in namespace {model_registry_namespace}"
    )

    # Define labels, to be passed via **kwargs
    binding_labels = {
        "app.kubernetes.io/component": "model-registry-test-rbac",
        "test.opendatahub.io/namespace": sa_namespace.name,
    }

    LOGGER.info(f"Attempting to create RoleBinding: {binding_name} with labels.")
    # Use context manager for creation and deletion
    # Pass subject and role_ref details directly to constructor
    with RoleBinding(
        client=admin_client,
        name=binding_name,
        namespace=model_registry_namespace,
        # Subject parameters
        subjects_kind="Group",
        subjects_name=f"system:serviceaccounts:{sa_namespace.name}",
        subjects_api_group="rbac.authorization.k8s.io",  # This is the default apiGroup for Group kind
        # Role reference parameters
        role_ref_kind="Role",
        role_ref_name=role_name_ref,
        # role_ref_api_group="rbac.authorization.k8s.io", # This is automatically set by the class
        label=binding_labels,  # Pass labels via kwargs
    ) as binding:
        try:
            binding.wait(timeout=60)  # Wait for binding object to exist
            LOGGER.info(f"RoleBinding {binding.name} created successfully.")
            yield binding
            LOGGER.info(f"RoleBinding {binding.name} deletion initiated by context manager.")
        except Exception as e:  # Catch other potential errors
            LOGGER.error(f"Error during RoleBinding {binding_name} creation or wait: {e}", exc_info=True)
            pytest.fail(f"Failed during RoleBinding {binding_name} creation: {e}")
