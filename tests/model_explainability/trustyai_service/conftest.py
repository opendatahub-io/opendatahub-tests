from typing import Generator, Any

import pytest
import yaml
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.maria_db import MariaDB
from ocp_resources.mariadb_operator import MariadbOperator
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.trustyai_service import TrustyAIService

from tests.model_explainability.trustyai_service.trustyai_service_utils import TRUSTYAI_SERVICE_NAME
from tests.model_explainability.trustyai_service.utils import (
    get_cluster_service_version,
    wait_for_mariadb_operator_deployments,
    wait_for_mariadb_pods,
)

from utilities.constants import Timeout
from utilities.infra import update_configmap_data

MINIO: str = "minio"
OPENDATAHUB_IO: str = "opendatahub.io"
OPENSHIFT_OPERATORS: str = "openshift-operators"

MARIADB: str = "mariadb"
DB_CREDENTIALS_SECRET_NAME: str = "db-credentials"
DB_NAME: str = "trustyai_db"
DB_USERNAME: str = "trustyai_user"
DB_PASSWORD: str = "trustyai_password"


@pytest.fixture(scope="class")
def trustyai_service_with_pvc_storage(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    cluster_monitoring_config: ConfigMap,
    user_workload_monitoring_config: ConfigMap,
) -> Generator[TrustyAIService, Any, Any]:
    with TrustyAIService(
        client=admin_client,
        name=TRUSTYAI_SERVICE_NAME,
        namespace=model_namespace.name,
        storage={"format": "PVC", "folder": "/inputs", "size": "1Gi"},
        data={"filename": "data.csv", "format": "CSV"},
        metrics={"schedule": "5s"},
    ) as trustyai_service:
        trustyai_deployment = Deployment(
            namespace=model_namespace.name, name=TRUSTYAI_SERVICE_NAME, wait_for_resource=True
        )
        trustyai_deployment.wait_for_replicas()
        yield trustyai_service


@pytest.fixture(scope="class")
def trustyai_service_with_db_storage(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    cluster_monitoring_config: ConfigMap,
    user_workload_monitoring_config: ConfigMap,
    mariadb: MariaDB,
) -> Generator[TrustyAIService, Any, Any]:
    with TrustyAIService(
        client=admin_client,
        name=TRUSTYAI_SERVICE_NAME,
        namespace=model_namespace.name,
        storage={"format": "DATABASE", "size": "1Gi", "databaseConfigurations": "db-credentials"},
        metrics={"schedule": "5s"},
    ) as trustyai_service:
        trustyai_deployment = Deployment(
            namespace=model_namespace.name, name=TRUSTYAI_SERVICE_NAME, wait_for_resource=True
        )
        trustyai_deployment.wait_for_replicas()
        yield trustyai_service


@pytest.fixture(scope="session")
def user_workload_monitoring_config(admin_client: DynamicClient) -> Generator[ConfigMap, Any, Any]:
    data = {"config.yaml": yaml.dump({"prometheus": {"logLevel": "debug", "retention": "15d"}})}
    with update_configmap_data(
        client=admin_client,
        name="user-workload-monitoring-config",
        namespace="openshift-user-workload-monitoring",
        data=data,
    ) as cm:
        yield cm


@pytest.fixture(scope="class")
def minio_pod(admin_client: DynamicClient, model_namespace: Namespace) -> Generator[Pod, Any, Any]:
    with Pod(
        client=admin_client,
        name=MINIO,
        namespace=model_namespace.name,
        containers=[
            {
                "args": [
                    "server",
                    "/data1",
                ],
                "env": [
                    {
                        "name": "MINIO_ACCESS_KEY",
                        "value": "THEACCESSKEY",
                    },
                    {
                        "name": "MINIO_SECRET_KEY",
                        "value": "THESECRETKEY",
                    },
                ],
                "image": "quay.io/trustyai_testing/modelmesh-minio-examples"
                "@sha256:d2ccbe92abf9aa5085b594b2cae6c65de2bf06306c30ff5207956eb949bb49da",
                "name": MINIO,
            }
        ],
        label={"app": "minio", "maistra.io/expose-route": "true"},
        annotations={"sidecar.istio.io/inject": "true"},
    ) as minio_pod:
        yield minio_pod


@pytest.fixture(scope="class")
def minio_service(admin_client: DynamicClient, model_namespace: Namespace) -> Generator[Service, Any, Any]:
    with Service(
        client=admin_client,
        name=MINIO,
        namespace=model_namespace.name,
        ports=[
            {
                "name": "minio-client-port",
                "port": 9000,
                "protocol": "TCP",
                "targetPort": 9000,
            }
        ],
        selector={
            "app": MINIO,
        },
    ) as minio_service:
        yield minio_service


@pytest.fixture(scope="class")
def minio_data_connection(
    admin_client: DynamicClient, model_namespace: Namespace, minio_pod: Pod, minio_service: Service
) -> Generator[Secret, Any, Any]:
    with Secret(
        client=admin_client,
        name="aws-connection-minio-data-connection",
        namespace=model_namespace.name,
        data_dict={
            "AWS_ACCESS_KEY_ID": "VEhFQUNDRVNTS0VZ",
            "AWS_DEFAULT_REGION": "dXMtc291dGg=",
            "AWS_S3_BUCKET": "bW9kZWxtZXNoLWV4YW1wbGUtbW9kZWxz",
            "AWS_S3_ENDPOINT": "aHR0cDovL21pbmlvOjkwMDA=",
            "AWS_SECRET_ACCESS_KEY": "VEhFU0VDUkVUS0VZ",  # pragma: allowlist secret
        },
        label={
            f"{OPENDATAHUB_IO}/dashboard": "true",
            f"{OPENDATAHUB_IO}/managed": "true",
        },
        annotations={
            f"{OPENDATAHUB_IO}/connection-type": "s3",
            "openshift.io/display-name": "Minio Data Connection",
        },
    ) as minio_secret:
        yield minio_secret


@pytest.fixture(scope="class")
def db_credentials_secret(admin_client: DynamicClient, model_namespace: Namespace) -> Generator[Secret, Any, Any]:
    with Secret(
        client=admin_client,
        name=DB_CREDENTIALS_SECRET_NAME,
        namespace=model_namespace.name,
        string_data={
            "databaseKind": MARIADB,
            "databaseName": DB_NAME,
            "databaseUsername": DB_USERNAME,
            "databasePassword": DB_PASSWORD,
            "databaseService": MARIADB,
            "databasePort": "3306",
            "databaseGeneration": "update",
        },
    ) as db_credentials:
        yield db_credentials


@pytest.fixture(scope="class")
def mariadb_operator_cr(admin_client: DynamicClient) -> Generator[MariadbOperator, Any, Any]:
    mariadb_csv: ClusterServiceVersion = get_cluster_service_version(
        client=admin_client, prefix="mariadb", namespace=OPENSHIFT_OPERATORS
    )
    alm_examples: list[dict[str, Any]] = mariadb_csv.get_alm_examples()
    mariadb_operator_cr_dict: dict[str, Any] = next(
        example for example in alm_examples if example["kind"] == "MariadbOperator"
    )

    if not mariadb_operator_cr_dict:
        raise ResourceNotFoundError(f"No MariadbOperator dict found in alm_examples for CSV {mariadb_csv.name}")

    mariadb_operator_cr_dict["metadata"]["namespace"] = OPENSHIFT_OPERATORS
    with MariadbOperator(kind_dict=mariadb_operator_cr_dict) as mariadb_operator_cr:
        mariadb_operator_cr.wait_for_condition(
            condition="Deployed", status=mariadb_operator_cr.Condition.Status.TRUE, timeout=Timeout.TIMEOUT_10MIN
        )
        wait_for_mariadb_operator_deployments(mariadb_operator=mariadb_operator_cr)
        yield mariadb_operator_cr


@pytest.fixture(scope="class")
def mariadb(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    db_credentials_secret: Secret,
    mariadb_operator_cr: MariadbOperator,
) -> Generator[MariaDB, Any, Any]:
    mariadb_csv: ClusterServiceVersion = get_cluster_service_version(
        client=admin_client, prefix=MARIADB, namespace=OPENSHIFT_OPERATORS
    )
    alm_examples: list[dict[str, Any]] = mariadb_csv.get_alm_examples()
    mariadb_dict: dict[str, Any] = next(example for example in alm_examples if example["kind"] == "MariaDB")

    if not mariadb_dict:
        raise ResourceNotFoundError(f"No MariaDB dict found in alm_examples for CSV {mariadb_csv.name}")

    mariadb_dict["metadata"]["namespace"] = model_namespace.name
    mariadb_dict["spec"]["database"] = DB_NAME
    mariadb_dict["spec"]["username"] = DB_USERNAME

    mariadb_dict["spec"]["replicas"] = 1
    mariadb_dict["spec"]["galera"]["enabled"] = False
    mariadb_dict["spec"]["metrics"]["enabled"] = False

    password_secret_key_ref = {"generate": False, "key": "databasePassword", "name": DB_CREDENTIALS_SECRET_NAME}

    mariadb_dict["spec"]["rootPasswordSecretKeyRef"] = password_secret_key_ref
    mariadb_dict["spec"]["passwordSecretKeyRef"] = password_secret_key_ref

    with MariaDB(kind_dict=mariadb_dict) as mariadb:
        wait_for_mariadb_pods(client=admin_client, mariadb=mariadb)
        yield mariadb
