import pytest
import re
import schemathesis
from typing import Generator, Any
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.namespace import Namespace
from ocp_resources.service import Service
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment

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
from model_registry.types import RegisteredModel
from tests.model_registry.constants import (
    MR_NAMESPACE,
    MR_OPERATOR_NAME,
    MR_INSTANCE_NAME,
    ISTIO_CONFIG_DICT,
    DB_RESOURCES_NAME,
    MR_DB_IMAGE_DIGEST,
)
from tests.model_registry.utils import get_endpoint_from_mr_service, get_mr_service_by_label
from utilities.infra import create_ns
from utilities.constants import Annotations, Protocols, DscComponents
from model_registry import ModelRegistry as ModelRegistryClient


LOGGER = get_logger(name=__name__)

DEFAULT_LABEL_DICT_DB: dict[str, str] = {
    Annotations.KubernetesIo.NAME: DB_RESOURCES_NAME,
    Annotations.KubernetesIo.INSTANCE: DB_RESOURCES_NAME,
    Annotations.KubernetesIo.PART_OF: DB_RESOURCES_NAME,
}


@pytest.fixture(scope="class")
def model_registry_namespace(request: FixtureRequest, admin_client: DynamicClient) -> Generator[Namespace, Any, Any]:
    # TODO: model_registry_namespace fixture should basically be doing this 1) check the ns exists, 2) it is in ACTIVE
    # state and return. But it should not create the ns. If DSC manages registriesNamespace, and namespace was not
    # created when mr is updated to Managed state, it would be a bug and we should catch it
    # To be handled in upcoming PR.
    with create_ns(
        name=request.param.get("namespace_name", MR_NAMESPACE),
        admin_client=admin_client,
    ) as ns:
        yield ns


@pytest.fixture(scope="class")
def model_registry_db_service(
    admin_client: DynamicClient, model_registry_namespace: Namespace
) -> Generator[Service, Any, Any]:
    with Service(
        client=admin_client,
        name=DB_RESOURCES_NAME,
        namespace=model_registry_namespace.name,
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
        label=DEFAULT_LABEL_DICT_DB,
        annotations={
            "template.openshift.io/expose-uri": r"mysql://{.spec.clusterIP}:{.spec.ports[?(.name==\mysql\)].port}",
        },
    ) as mr_db_service:
        yield mr_db_service


@pytest.fixture(scope="class")
def model_registry_db_pvc(
    admin_client: DynamicClient,
    model_registry_namespace: Namespace,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    with PersistentVolumeClaim(
        accessmodes="ReadWriteOnce",
        name=DB_RESOURCES_NAME,
        namespace=model_registry_namespace.name,
        client=admin_client,
        size="5Gi",
        label=DEFAULT_LABEL_DICT_DB,
    ) as pvc:
        yield pvc


@pytest.fixture(scope="class")
def model_registry_db_secret(
    admin_client: DynamicClient,
    model_registry_namespace: Namespace,
) -> Generator[Secret, Any, Any]:
    with Secret(
        client=admin_client,
        name=DB_RESOURCES_NAME,
        namespace=model_registry_namespace.name,
        string_data={
            "database-name": "model_registry",
            "database-password": "TheBlurstOfTimes",  # pragma: allowlist secret
            "database-user": "mlmduser",  # pragma: allowlist secret
        },
        label=DEFAULT_LABEL_DICT_DB,
        annotations={
            "template.openshift.io/expose-database_name": "'{.data[''database-name'']}'",
            "template.openshift.io/expose-password": "'{.data[''database-password'']}'",
            "template.openshift.io/expose-username": "'{.data[''database-user'']}'",
        },
    ) as mr_db_secret:
        yield mr_db_secret


@pytest.fixture(scope="class")
def model_registry_db_deployment(
    admin_client: DynamicClient,
    model_registry_namespace: Namespace,
    model_registry_db_secret: Secret,
    model_registry_db_pvc: PersistentVolumeClaim,
    model_registry_db_service: Service,
) -> Generator[Deployment, Any, Any]:
    with Deployment(
        name=DB_RESOURCES_NAME,
        namespace=model_registry_namespace.name,
        annotations={
            "template.alpha.openshift.io/wait-for-ready": "true",
        },
        label=DEFAULT_LABEL_DICT_DB,
        replicas=1,
        revision_history_limit=0,
        selector={"matchLabels": {"name": DB_RESOURCES_NAME}},
        strategy={"type": "Recreate"},
        template={
            "metadata": {
                "labels": {
                    "name": DB_RESOURCES_NAME,
                    "sidecar.istio.io/inject": "false",
                }
            },
            "spec": {
                "containers": [
                    {
                        "env": [
                            {
                                "name": "MYSQL_USER",
                                "valueFrom": {
                                    "secretKeyRef": {
                                        "key": "database-user",
                                        "name": f"{model_registry_db_secret.name}",
                                    }
                                },
                            },
                            {
                                "name": "MYSQL_PASSWORD",
                                "valueFrom": {
                                    "secretKeyRef": {
                                        "key": "database-password",
                                        "name": f"{model_registry_db_secret.name}",
                                    }
                                },
                            },
                            {
                                "name": "MYSQL_ROOT_PASSWORD",
                                "valueFrom": {
                                    "secretKeyRef": {
                                        "key": "database-password",
                                        "name": f"{model_registry_db_secret.name}",
                                    }
                                },
                            },
                            {
                                "name": "MYSQL_DATABASE",
                                "valueFrom": {
                                    "secretKeyRef": {
                                        "key": "database-name",
                                        "name": f"{model_registry_db_secret.name}",
                                    }
                                },
                            },
                        ],
                        "args": [
                            "--datadir",
                            "/var/lib/mysql/datadir",
                            "--default-authentication-plugin=mysql_native_password",
                        ],
                        "image": MR_DB_IMAGE_DIGEST,
                        "imagePullPolicy": "IfNotPresent",
                        "livenessProbe": {
                            "exec": {
                                "command": [
                                    "/bin/bash",
                                    "-c",
                                    "mysqladmin -u${MYSQL_USER} -p${MYSQL_ROOT_PASSWORD} ping",
                                ]
                            },
                            "initialDelaySeconds": 15,
                            "periodSeconds": 10,
                            "timeoutSeconds": 5,
                        },
                        "name": "mysql",
                        "ports": [{"containerPort": 3306, "protocol": "TCP"}],
                        "readinessProbe": {
                            "exec": {
                                "command": [
                                    "/bin/bash",
                                    "-c",
                                    'mysql -D ${MYSQL_DATABASE} -u${MYSQL_USER} -p${MYSQL_ROOT_PASSWORD} -e "SELECT 1"',
                                ]
                            },
                            "initialDelaySeconds": 10,
                            "timeoutSeconds": 5,
                        },
                        "securityContext": {"capabilities": {}, "privileged": False},
                        "terminationMessagePath": "/dev/termination-log",
                        "volumeMounts": [
                            {
                                "mountPath": "/var/lib/mysql",
                                "name": f"{DB_RESOURCES_NAME}-data",
                            }
                        ],
                    }
                ],
                "dnsPolicy": "ClusterFirst",
                "restartPolicy": "Always",
                "volumes": [
                    {
                        "name": f"{DB_RESOURCES_NAME}-data",
                        "persistentVolumeClaim": {"claimName": DB_RESOURCES_NAME},
                    }
                ],
            },
        },
        wait_for_resource=True,
    ) as mr_db_deployment:
        mr_db_deployment.wait_for_replicas(deployed=True)
        yield mr_db_deployment


@pytest.fixture(scope="class")
def model_registry_instance(
    admin_client: DynamicClient,
    model_registry_namespace: Namespace,
    model_registry_db_deployment: Deployment,
    model_registry_db_secret: Secret,
    model_registry_db_service: Service,
) -> Generator[ModelRegistry, Any, Any]:
    with ModelRegistry(
        name=MR_INSTANCE_NAME,
        namespace=model_registry_namespace.name,
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
    model_registry_namespace: Namespace,
    model_registry_instance: ModelRegistry,
) -> Service:
    return get_mr_service_by_label(
        client=admin_client, ns=model_registry_namespace, mr_instance=model_registry_instance
    )


@pytest.fixture(scope="class")
def model_registry_instance_rest_endpoint(
    admin_client: DynamicClient,
    model_registry_instance_service: Service,
) -> str:
    return get_endpoint_from_mr_service(
        client=admin_client, svc=model_registry_instance_service, protocol=Protocols.REST
    )


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
    request: FixtureRequest,
    model_registry_namespace: Namespace,
    dsc_resource: DataScienceCluster,
) -> Generator[DataScienceCluster, Any, Any]:
    original_components = dsc_resource.instance.spec.components
    with ResourceEditor(patches={dsc_resource: {"spec": {"components": request.param["component_patch"]}}}):
        for component_name in request.param["component_patch"]:
            dsc_resource.wait_for_condition(condition=DscComponents.COMPONENT_MAPPING[component_name], status="True")
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
