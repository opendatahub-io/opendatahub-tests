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
from ocp_resources.secret import Secret
from ocp_resources.subscription import Subscription
from ocp_resources.trustyai_service import TrustyAIService
from ocp_utilities.operators import install_operator, uninstall_operator

from tests.model_explainability.trustyai_service.trustyai_service_utils import TRUSTYAI_SERVICE_NAME
from tests.model_explainability.trustyai_service.utils import (
    get_cluster_service_version,
    wait_for_mariadb_operator_deployments,
    wait_for_mariadb_pods,
)

from utilities.constants import Timeout
from utilities.infra import update_configmap_data

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
    trustyai_db_ca_secret: None,
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


@pytest.fixture(scope="session")
def installed_mariadb_operator(admin_client: DynamicClient) -> Generator[None, Any, Any]:
    operator_ns = Namespace(name="openshift-operators", ensure_exists=True)
    operator_name = "mariadb-operator"

    mariadb_operator_subscription = Subscription(client=admin_client, namespace=operator_ns.name, name=operator_name)

    if not mariadb_operator_subscription.exists:
        install_operator(
            admin_client=admin_client,
            target_namespaces=[],
            name=operator_name,
            channel="alpha",
            source="community-operators",
            operator_namespace=operator_ns.name,
            timeout=Timeout.TIMEOUT_15MIN,
            install_plan_approval="Manual",
            starting_csv=f"{operator_name}.v0.37.1",
        )

        deployment = Deployment(
            client=admin_client,
            namespace=operator_ns.name,
            name=f"{operator_name}-helm-controller-manager",
            wait_for_resource=True,
        )
        deployment.wait_for_replicas()

    yield
    uninstall_operator(
        admin_client=admin_client, name=operator_name, operator_namespace=operator_ns.name, clean_up_namespace=False
    )


@pytest.fixture(scope="class")
def mariadb_operator_cr(
    admin_client: DynamicClient, installed_mariadb_operator: None
) -> Generator[MariadbOperator, Any, Any]:
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
    mariadb_dict["spec"]["tls"] = {"enabled": True, "required": True}

    password_secret_key_ref = {"generate": False, "key": "databasePassword", "name": DB_CREDENTIALS_SECRET_NAME}

    mariadb_dict["spec"]["rootPasswordSecretKeyRef"] = password_secret_key_ref
    mariadb_dict["spec"]["passwordSecretKeyRef"] = password_secret_key_ref
    with MariaDB(kind_dict=mariadb_dict) as mariadb:
        wait_for_mariadb_pods(client=admin_client, mariadb=mariadb)
        yield mariadb


@pytest.fixture(scope="class")
def trustyai_db_ca_secret(
    admin_client: DynamicClient, model_namespace: Namespace, mariadb: MariaDB
) -> Generator[None, Any, None]:
    mariadb_ca_secret = Secret(
        client=admin_client, name=f"{mariadb.name}-ca", namespace=model_namespace.name, ensure_exists=True
    )
    with Secret(
        client=admin_client,
        name=f"{TRUSTYAI_SERVICE_NAME}-db-ca",
        namespace=model_namespace.name,
        data_dict={"ca.crt": mariadb_ca_secret.instance.data["ca.crt"]},
    ):
        yield
