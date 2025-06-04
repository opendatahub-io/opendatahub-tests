import os
import pytest
from pytest import Config
import schemathesis
from typing import Generator, Any
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

from tests.model_registry.constants import (
    MR_OPERATOR_NAME,
    MR_INSTANCE_NAME,
    ISTIO_CONFIG_DICT,
    DB_RESOURCES_NAME,
    MODEL_REGISTRY_DB_SECRET_STR_DATA,
    MODEL_REGISTRY_DB_SECRET_ANNOTATIONS,
    OAUTH_PROXY_CONFIG_DICT,
    MODEL_REGISTRY_STANDARD_LABELS,
)
from utilities.constants import Labels
from tests.model_registry.utils import (
    get_endpoint_from_mr_service,
    get_mr_service_by_label,
    get_model_registry_deployment_template_dict,
    get_model_registry_db_label_dict,
    wait_for_pods_running,
    create_model_registry_instance,
)
from utilities.constants import Protocols, DscComponents
from model_registry import ModelRegistry as ModelRegistryClient
from semver import Version
from utilities.infra import get_product_version
from utilities.operator_utils import get_cluster_service_version, validate_operator_subscription_channel
from utilities.general import wait_for_pods_by_labels

LOGGER = get_logger(name=__name__)

MIN_MR_VERSION = Version.parse(version="2.20.0")


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
    model_registry_namespace: str,
    model_registry_mysql_config: dict[str, Any],
) -> Generator[ModelRegistry, Any, Any]:
    """Creates a model registry instance with service mesh configuration."""
    with create_model_registry_instance(
        namespace=model_registry_namespace,
        name=MR_INSTANCE_NAME,
        labels=MODEL_REGISTRY_STANDARD_LABELS,
        grpc={},
        rest={},
        istio=ISTIO_CONFIG_DICT,
        mysql=model_registry_mysql_config,
        wait_for_resource=True,
    ) as mr:
        mr.wait_for_condition(condition="Available", status="True")
        yield mr


@pytest.fixture(scope="class")
def model_registry_instance_oauth_proxy(
    model_registry_namespace: str,
    model_registry_mysql_config: dict[str, Any],
) -> Generator[ModelRegistry, Any, Any]:
    """Creates a model registry instance with oauth proxy configuration."""
    with create_model_registry_instance(
        namespace=model_registry_namespace,
        name=MR_INSTANCE_NAME,
        labels=MODEL_REGISTRY_STANDARD_LABELS,
        grpc={},
        rest={},
        oauth_proxy=OAUTH_PROXY_CONFIG_DICT,
        mysql=model_registry_mysql_config,
        wait_for_resource=True,
    ) as mr:
        mr.wait_for_condition(condition="Available", status="True")
        yield mr


@pytest.fixture(scope="class")
def model_registry_mysql_config(
    model_registry_db_deployment: Deployment, model_registry_db_secret: Secret
) -> dict[str, Any]:
    return {
        "host": f"{model_registry_db_deployment.name}.{model_registry_db_deployment.namespace}.svc.cluster.local",
        "database": model_registry_db_secret.string_data["database-name"],
        "passwordSecret": {"key": "database-password", "name": model_registry_db_deployment.name},
        "port": 3306,
        "skipDBCreation": False,
        "username": model_registry_db_secret.string_data["database-user"],
    }


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
        client=admin_client,
        ns=Namespace(name=model_registry_namespace),
        mr_instance=model_registry_instance,
    )


@pytest.fixture(scope="class")
def model_registry_instance_oauth_service(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    model_registry_instance_oauth_proxy: ModelRegistry,
) -> Service:
    """
    Get the service for the OAuth proxy model registry instance.
    Args:
        admin_client: The admin client
        model_registry_namespace: The namespace where the model registry is deployed
        model_registry_instance_oauth_proxy: The OAuth proxy model registry instance
    Returns:
        Service: The service for the OAuth proxy model registry instance
    """
    return get_mr_service_by_label(
        client=admin_client,
        ns=Namespace(name=model_registry_namespace),
        mr_instance=model_registry_instance_oauth_proxy,
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


@pytest.fixture(scope="class")
def updated_dsc_component_state_scope_class(
    request: FixtureRequest, dsc_resource: DataScienceCluster, admin_client: DynamicClient
) -> Generator[DataScienceCluster, Any, Any]:
    original_components = dsc_resource.instance.spec.components
    component_patch = request.param["component_patch"]

    with ResourceEditor(patches={dsc_resource: {"spec": {"components": component_patch}}}):
        for component_name in component_patch:
            dsc_resource.wait_for_condition(condition=DscComponents.COMPONENT_MAPPING[component_name], status="True")
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
def model_registry_client(
    request: FixtureRequest,
    current_client_token: str,
) -> ModelRegistryClient:
    """
    Get a client for the model registry instance.
    Args:
        request: The pytest request object
        current_client_token: The current client token
    Returns:
        ModelRegistryClient: A client for the model registry instance
    """
    # Get the service fixture name from the request
    service_fixture = getattr(request, "param", {}).get("service_fixture", "model_registry_instance_service")

    # Get the service from the fixture
    service = request.getfixturevalue(argname=service_fixture)

    # Get the REST endpoint
    rest_endpoint = get_endpoint_from_mr_service(svc=service, protocol=Protocols.REST)

    # Create the client
    server, port = rest_endpoint.split(":")
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


@pytest.fixture(scope="package", autouse=True)
def validate_authorino_operator_version_channel(admin_client: DynamicClient) -> None:
    """Check if Authorino operator is installed with required version and channel.

    This fixture is automatically used for all tests in the model_registry directory.
    It verifies that:
    1. For OpenShift AI: The product version is >= 2.20
    2. The Authorino operator is installed
    3. The Authorino operator is using the required channel (stable)
    4. The Authorino operator is at least version 1.2.1
    """
    distribution = py_config["distribution"]
    if distribution == "upstream":
        # TODO: figure out minimum version for ODH
        LOGGER.info(f"Skipping Authorino operator check for {distribution} distribution")
        return
    # Only check product version for OpenShift AI
    if distribution == "downstream":
        product_version = get_product_version(admin_client=admin_client)
        if product_version < MIN_MR_VERSION:
            LOGGER.info(
                "Skipping Authorino operator check - product version "
                f"{product_version} is below required {MIN_MR_VERSION}"
            )
            return
        operator_name = "authorino-operator"
        # Find the CSV for the operator
        authorino_csv = get_cluster_service_version(
            client=admin_client, prefix=operator_name, namespace=py_config["applications_namespace"]
        )
        current_authorino_version = authorino_csv.instance.spec.version
        if Version.parse(version="1.2.1") > Version.parse(version=current_authorino_version):
            pytest.exit(
                f"Authorino operator is not at least version 1.2.1. Current version: {current_authorino_version}"
            )

        validate_operator_subscription_channel(
            client=admin_client, namespace="openshift-operators", operator_name=operator_name, channel_name="stable"
        )
