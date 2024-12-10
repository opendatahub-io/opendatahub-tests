import pytest
import schemathesis
from typing import Generator, Any
from ocp_resources.secret import Secret
from ocp_resources.namespace import Namespace
from ocp_resources.service import Service
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.deployment import Deployment
from ocp_resources.model_registry import ModelRegistry
from simple_logger.logger import get_logger
from kubernetes.dynamic import DynamicClient

from tests.model_registry.utils import get_endpoint_from_mr_service, get_mr_service_by_label
from utilities.infra import create_ns
from utilities.constants import Protocols


LOGGER = get_logger(name=__name__)

DB_RESOURCES_NAME = "model-registry-db"


@pytest.fixture(scope="class")
def model_registry_namespace(admin_client: DynamicClient) -> Generator[Namespace, Any, Any]:
    # This namespace should exist after Model Registry is enabled, but it can also be deleted
    # from the cluster and does not get reconciled. Fetch if it exists, create otherwise.
    ns = Namespace(name="rhoai-model-registries", client=admin_client)
    if ns.exists:
        yield ns
    else:
        LOGGER.warning("rhoai-model-registries namespace was not present, creating it")
        with create_ns(
            name="rhoai-model-registries",
            admin_client=admin_client,
            teardown=False,
            ensure_exists=True,
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
        label={
            "app.kubernetes.io/name": DB_RESOURCES_NAME,
            "app.kubernetes.io/instance": DB_RESOURCES_NAME,
            "app.kubernetes.io/part-of": DB_RESOURCES_NAME,
        },
        annotations={
            "template.openshift.io/expose-uri": "mysql://{.spec.clusterIP}:{.spec.ports[?(.name==\mysql\)].port}",
        },
    ) as mr_db_service:
        yield mr_db_service


@pytest.fixture(scope="class")
def model_registry_db_pvc(
    admin_client: DynamicClient,
    model_registry_namespace: Namespace,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    pvc_kwargs = {
        "accessmodes": "ReadWriteOnce",
        "name": DB_RESOURCES_NAME,
        "namespace": model_registry_namespace.name,
        "client": admin_client,
        "size": "5Gi",
        "label": {
            "app.kubernetes.io/name": DB_RESOURCES_NAME,
            "app.kubernetes.io/instance": DB_RESOURCES_NAME,
            "app.kubernetes.io/part-of": DB_RESOURCES_NAME,
        },
    }

    with PersistentVolumeClaim(**pvc_kwargs) as pvc:
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
        label={
            "app.kubernetes.io/name": DB_RESOURCES_NAME,
            "app.kubernetes.io/instance": DB_RESOURCES_NAME,
            "app.kubernetes.io/part-of": DB_RESOURCES_NAME,
        },
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
        label={
            "app.kubernetes.io/name": DB_RESOURCES_NAME,
            "app.kubernetes.io/instance": DB_RESOURCES_NAME,
            "app.kubernetes.io/part-of": DB_RESOURCES_NAME,
        },
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
                        "image": "mysql:8.3.0",
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
        name="model-registry",
        namespace=model_registry_namespace.name,
        label={
            "app.kubernetes.io/name": "model-registry",
            "app.kubernetes.io/instance": "model-registry",
            "app.kubernetes.io/part-of": "model-registry-operator",
            "app.kubernetes.io/created-by": "model-registry-operator",
        },
        grpc={},
        rest={},
        istio={
            "authProvider": "redhat-ods-applications-auth-provider",
            "gateway": {"grpc": {"tls": {}}, "rest": {"tls": {}}},
        },
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
    return get_mr_service_by_label(admin_client, model_registry_namespace, model_registry_instance)


@pytest.fixture(scope="class")
def model_registry_instance_rest_endpoint(
    admin_client: DynamicClient,
    model_registry_instance_service: Service,
) -> str:
    return get_endpoint_from_mr_service(admin_client, model_registry_instance_service, Protocols.REST)


@pytest.fixture(scope="class")
def generate_schema(model_registry_instance_rest_endpoint):
    # host = model_registry_instance_rest_endpoint.split(":")[0]
    # port = model_registry_instance_rest_endpoint.split(":")[1]
    schema = schemathesis.from_uri(
        uri="https://raw.githubusercontent.com/kubeflow/model-registry/main/api/openapi/model-registry.yaml",
        base_url=f"https://{model_registry_instance_rest_endpoint}/",
    )
    return schema
