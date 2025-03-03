import os
import tempfile
import pytest
import schemathesis
import shutil
from typing import Generator, Any
from ocp_resources.secret import Secret
from ocp_resources.namespace import Namespace
from ocp_resources.service import Service
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.deployment import Deployment
from ocp_resources.job import Job
from ocp_resources.model_registry import ModelRegistry
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger
from kubernetes.dynamic import DynamicClient
from model_registry import ModelRegistry as ModelRegistryClient, __version__ as mr_client_version
from utilities.general import fetch_external_tests_from_github

from tests.model_registry.utils import get_endpoint_from_mr_service, get_mr_service_by_label
from utilities.infra import create_ns
from utilities.constants import Annotations, Protocols


LOGGER = get_logger(name=__name__)

DB_RESOURCES_NAME: str = "model-registry-db"
MR_INSTANCE_NAME: str = "model-registry"
MR_OPERATOR_NAME: str = "model-registry-operator"
MR_NAMESPACE: str = "rhoai-model-registries"
DEFAULT_LABEL_DICT_DB: dict[str, str] = {
    Annotations.KubernetesIo.NAME: DB_RESOURCES_NAME,
    Annotations.KubernetesIo.INSTANCE: DB_RESOURCES_NAME,
    Annotations.KubernetesIo.PART_OF: DB_RESOURCES_NAME,
}
# Define upstream repository details
REPO_API_URL = "https://raw.githubusercontent.com/kubeflow/model-registry"
REPO_BRANCH = f"refs/heads/release/v{mr_client_version}"
FILE_PATH = "clients/python/tests"
# "test_core.py" excluded for relative imports
# test_patch_model_artifacts_artifact_type fails because of a relative import
FILES = ["test_client.py", "regression_test.py"]
PROJECT_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))  # Path of `conftest.py`
UPSTREAM_TESTS_DIR = os.path.join(PROJECT_TESTS_DIR, "upstream_tests")


@pytest.fixture(scope="class")
def model_registry_namespace(admin_client: DynamicClient) -> Generator[Namespace, Any, Any]:
    # This namespace should exist after Model Registry is enabled, but it can also be deleted
    # from the cluster and does not get reconciled. Fetch if it exists, create otherwise.
    ns = Namespace(name=MR_NAMESPACE, client=admin_client)
    if ns.exists:
        yield ns
    else:
        LOGGER.warning(f"{MR_NAMESPACE} namespace was not present, creating it")
        with create_ns(
            name=MR_NAMESPACE,
            admin_client=admin_client,
            teardown=False,
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
def generated_schema(model_registry_instance_rest_endpoint: str) -> Any:
    return schemathesis.from_uri(
        uri="https://raw.githubusercontent.com/kubeflow/model-registry/main/api/openapi/model-registry.yaml",
        base_url=f"https://{model_registry_instance_rest_endpoint}/",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    """Downloads external tests into the local project structure before test execution."""
    if session.config.option.model_registry_upstream:
        LOGGER.info("pytest_sessionstart is running")
        fetch_external_tests_from_github(
            dir=UPSTREAM_TESTS_DIR, files=FILES, repo_api_url=REPO_API_URL, branch=REPO_BRANCH, file_path=FILE_PATH
        )
    else:
        LOGGER.info("Skipping upstream test fetching for Model registry. Enable with --model-registry-upstream=True")


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: pytest.ExitCode) -> None:
    """Clean up upstream tests from kubeflow/model-registry"""
    if session.config.option.model_registry_upstream:
        if os.path.exists(UPSTREAM_TESTS_DIR):
            shutil.rmtree(UPSTREAM_TESTS_DIR)
            LOGGER.info(f"Removed upstream model registry tests from {UPSTREAM_TESTS_DIR}")


# fixture needed for upstream tests
@pytest.fixture(scope="function")
def client(model_registry_instance_rest_endpoint: str, current_client_token: str) -> ModelRegistryClient:
    server, port = model_registry_instance_rest_endpoint.split(":")
    client = ModelRegistryClient(
        server_address=f"{Protocols.HTTPS}://{server}",
        port=port,
        author="opendatahub-test",
        user_token=current_client_token,
        is_secure=False,
    )
    return client


# fixture needed for upstream tests
# Marked with autouse to stop tox from complaining
@pytest.fixture(scope="module", autouse=True)
def setup_env_user_token() -> None:
    pass


# fixture needed for upstream tests from v0.2.15 onwards
# skip-unused-code
@pytest.fixture
def get_model_file() -> Generator[str, Any, Any]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as model_file:
        pass

    yield model_file.name

    os.remove(model_file.name)


# fixture needed for upstream tests from v0.2.15 onwards
# skip-unused-code
@pytest.fixture
def get_temp_dir_with_models() -> Generator[tuple[str, list[str]], Any, Any]:
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    for _ in range(3):
        tmp_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
            delete=False, dir=temp_dir, suffix=".onnx"
        )
        file_paths.append(tmp_file.name)
        tmp_file.close()

    yield temp_dir, file_paths

    for file in file_paths:
        if os.path.exists(file):
            os.remove(file)
    os.rmdir(temp_dir)


# fixture needed for upstream tests from v0.2.15 onwards
# skip-unused-code
@pytest.fixture
def get_temp_dir_with_nested_models() -> Generator[tuple[str, list[str]], Any, Any]:
    temp_dir = tempfile.mkdtemp()
    nested_dir = tempfile.mkdtemp(dir=temp_dir)

    file_paths = []
    for _ in range(3):
        tmp_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
            delete=False, dir=nested_dir, suffix=".onnx"
        )
        file_paths.append(tmp_file.name)
        tmp_file.close()

    yield temp_dir, file_paths

    for file in file_paths:
        if os.path.exists(file):
            os.remove(file)
    os.rmdir(nested_dir)
    os.rmdir(temp_dir)


# hook needed for upstream tests from v0.2.15 onwards
# skip-unused-code
def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Dynamically parametrize `minio_pod` used by upstream tests"""
    if "minio_pod" in metafunc.fixturenames and metafunc.config.getoption("model_registry_upstream"):
        metafunc.parametrize(
            "minio_pod",
            [
                {
                    "args": ["server", "/data"],
                    "image": "minio/minio:latest",
                    "labels": {"app": "minio"},
                    "annotations": {},
                }
            ],
            indirect=True,
        )


# fixture needed for upstream tests from v0.2.15 onwards
# skip-unused-code
@pytest.fixture
def mr_minio_init(admin_client: DynamicClient, minio_namespace: Namespace) -> Generator[Job, Any, Any]:
    with Job(
        client=admin_client,
        name="minio-init",
        namespace=minio_namespace.name,
        restart_policy="OnFailure",
        containers=[
            {
                "args": [
                    "sleep 5",
                    "mc alias set local http://minio:9000 $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD",
                    "mc mb local/default || true",
                ],
                "command": ["/bin/sh", "-c"],
                "env": [
                    {
                        "name": "MINIO_ROOT_USER",
                        "value": "THEACCESSKEY",
                    },
                    {
                        "name": "MINIO_ROOT_PASSWORD",
                        "value": "THESECRETKEY",
                    },
                ],
                "image": "minio/mc",
                "name": "mc",
            }
        ],
    ) as mr_job:
        yield mr_job


# fixture needed for upstream tests from v0.2.15 onwards
# skip-unused-code
@pytest.fixture
def patch_s3_env(monkeypatch: pytest.MonkeyPatch, minio_pod: Pod, minio_service: Service, mr_minio_init: Job) -> None:
    monkeypatch.setenv("AWS_S3_ENDPOINT", "http://minio:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "THEACCESSKEY")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "THESECRETKEY")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_S3_BUCKET", "default-bucket")
