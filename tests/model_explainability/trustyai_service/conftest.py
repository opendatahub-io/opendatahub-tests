from typing import Generator, Any

import pytest
import yaml
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.maria_db import MariaDB
from ocp_resources.mariadb_operator import MariadbOperator
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.subscription import Subscription
from ocp_resources.trustyai_service import TrustyAIService
from ocp_utilities.operators import install_operator, uninstall_operator

from tests.model_explainability.trustyai_service.trustyai_service_utils import (
    wait_for_isvc_deployment_registered_by_trustyai_service,
)
from tests.model_explainability.trustyai_service.utils import (
    get_cluster_service_version,
    wait_for_mariadb_operator_deployments,
    create_trustyai_service,
    wait_for_mariadb_pods,
    TRUSTYAI_SERVICE_NAME,
)

from utilities.constants import Timeout, KServeDeploymentType, ApiGroups, Labels, Ports
from utilities.inference_utils import create_isvc
from utilities.infra import update_configmap_data


OPENSHIFT_OPERATORS: str = "openshift-operators"

TAI_DATA_CONFIG = {"filename": "data.csv", "format": "CSV"}
TAI_METRICS_CONFIG = {"schedule": "5s"}
TAI_DB_STORAGE_CONFIG = {"format": "DATABASE", "size": "1Gi", "databaseConfigurations": "db-credentials"}
MARIADB: str = "mariadb"
DB_CREDENTIALS_SECRET_NAME: str = "db-credentials"
DB_NAME: str = "trustyai_db"
DB_USERNAME: str = "trustyai_user"
DB_PASSWORD: str = "trustyai_password"
MLSERVER: str = "mlserver"
MLSERVER_RUNTIME_NAME: str = f"{MLSERVER}-1.x"
XGBOOST: str = "xgboost"
SKLEARN: str = "sklearn"
LIGHTGBM: str = "lightgbm"
MLFLOW: str = "mlflow"
TIMEOUT_20MIN: int = 20 * Timeout.TIMEOUT_1MIN
INVALID_TLS_CERTIFICATE: str = "LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUJnRENDQVNlZ0F3SUJBZ0lRRGtTcXVuUWRzRmZwdi8zSm\
5TS2ZoVEFLQmdncWhrak9QUVFEQWpBVk1STXcKRVFZRFZRUURFd3B0WVhKcFlXUmlMV05oTUI0WERUSTFNRFF4TkRFME1EUXhOMW9YRFRJNE1EUXhNekUx\
TURReApOMW93RlRFVE1CRUdBMVVFQXhNS2JXRnlhV0ZrWWkxallUQlpNQk1HQnlxR1NNNDlBZ0VHQ0NxR1NNNDlBd0VICkEwSUFCQ2IxQ1IwUjV1akZ1QUR\
Gd1NsazQzUUpmdDFmTFVnOWNJNyttZ0w3bVd3MmVLUXowL04ybm9KMGpJaDYKN0NnQ2syUW1jNTdWM1podkFWQzJoU2NEbWg2aldUQlhNQTRHQTFVZER3RU\
Ivd1FFQXdJQ0JEQVBCZ05WSFJNQgpBZjhFQlRBREFRSC9NQjBHQTFVZERnUVdCQlNUa2tzSU9pL1pTbCtQRlJua2NQRlJ0QTRrMERBVkJnTlZIUkVFCkRqQ\
U1nZ3B0WVhKcFlXUmlMV05oTUFvR0NDcUdTTTQ5QkFNQ0EwY0FNRVFDSUI1Q2F6VW1WWUZQYTFkS2txUGkKbitKSEQvNVZTTGd4aHVPclgzUGcxQnlzQWlB\
RmcvTXlNWW9CZUNrUVRWdS9rUkIwK2N2Qy9RMDB4NExvVGpJaQpGdCtKMGc9PQotLS0tLUVORCBDRVJUSUZJQ0FURS0t\
LS0t"  # pragma: allowlist secret


@pytest.fixture(scope="class")
def trustyai_service_with_pvc_storage(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    cluster_monitoring_config: ConfigMap,
    user_workload_monitoring_config: ConfigMap,
    teardown_resources: bool,
) -> Generator[TrustyAIService, Any, Any]:
    trustyai_service_kwargs = {"client": admin_client, "namespace": model_namespace.name, "name": TRUSTYAI_SERVICE_NAME}
    trustyai_service = TrustyAIService(**trustyai_service_kwargs)

    if pytestconfig.option.post_upgrade:
        yield trustyai_service
        trustyai_service.clean_up()

    else:
        yield from create_trustyai_service(
            **trustyai_service_kwargs,
            storage={"format": "PVC", "folder": "/inputs", "size": "1Gi"},
            metrics=TAI_METRICS_CONFIG,
            data=TAI_DATA_CONFIG,
            wait_for_replicas=True,
            teardown=teardown_resources,
        )


@pytest.fixture(scope="class")
def trustyai_service_with_db_storage(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    cluster_monitoring_config: ConfigMap,
    user_workload_monitoring_config: ConfigMap,
    mariadb: MariaDB,
    trustyai_db_ca_secret: None,
) -> Generator[TrustyAIService, Any, Any]:
    yield from create_trustyai_service(
        client=admin_client,
        namespace=model_namespace.name,
        storage=TAI_DB_STORAGE_CONFIG,
        metrics=TAI_METRICS_CONFIG,
        wait_for_replicas=True,
    )


@pytest.fixture(scope="class")
def trustyai_service_with_invalid_db_cert(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    cluster_monitoring_config: ConfigMap,
    user_workload_monitoring_config: ConfigMap,
    mariadb: MariaDB,
    trustyai_invalid_db_ca_secret: None,
) -> Generator[TrustyAIService, None, None]:
    """Create a TrustyAIService deployment with an invalid database certificate set as secret.

    Yields: A secret with invalid database certificate set.
    """
    yield from create_trustyai_service(
        client=admin_client,
        namespace=model_namespace.name,
        storage=TAI_DB_STORAGE_CONFIG,
        metrics=TAI_METRICS_CONFIG,
        wait_for_replicas=False,
    )


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
) -> Generator[Secret, Any, None]:
    mariadb_ca_secret = Secret(
        client=admin_client, name=f"{mariadb.name}-ca", namespace=model_namespace.name, ensure_exists=True
    )
    with Secret(
        client=admin_client,
        name=f"{TRUSTYAI_SERVICE_NAME}-db-ca",
        namespace=model_namespace.name,
        data_dict={"ca.crt": mariadb_ca_secret.instance.data["ca.crt"]},
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def trustyai_invalid_db_ca_secret(
    admin_client: DynamicClient, model_namespace: Namespace, mariadb: MariaDB
) -> Generator[Secret, Any, None]:
    with Secret(
        client=admin_client,
        name=f"{TRUSTYAI_SERVICE_NAME}-db-ca",
        namespace=model_namespace.name,
        data_dict={"ca.crt": INVALID_TLS_CERTIFICATE},
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def mlserver_runtime(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    minio_data_connection: Secret,
    model_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[ServingRuntime, Any, Any]:
    mlserver_runtime_kwargs = {
        "client": admin_client,
        "namespace": model_namespace.name,
        "name": "kserve-mlserver",
    }

    serving_runtime = ServingRuntime(**mlserver_runtime_kwargs)

    if pytestconfig.option.post_upgrade:
        yield serving_runtime
        serving_runtime.clean_up()

    supported_model_formats = [
        {"name": SKLEARN, "version": "0", "autoSelect": True, "priority": 2},
        {"name": SKLEARN, "version": "1", "autoSelect": True, "priority": 2},
        {"name": XGBOOST, "version": "1", "autoSelect": True, "priority": 2},
        {"name": XGBOOST, "version": "2", "autoSelect": True, "priority": 2},
        {"name": LIGHTGBM, "version": "3", "autoSelect": True, "priority": 2},
        {"name": LIGHTGBM, "version": "4", "autoSelect": True, "priority": 2},
        {"name": MLFLOW, "version": "1", "autoSelect": True, "priority": 1},
        {"name": MLFLOW, "version": "2", "autoSelect": True, "priority": 1},
    ]
    containers = [
        {
            "name": "kserve-container",
            "image": "quay.io/trustyai_testing/mlserver"
            "@sha256:68a4cd74fff40a3c4f29caddbdbdc9e54888aba54bf3c5f78c8ffd577c3a1c89",
            "env": [
                {"name": "MLSERVER_MODEL_IMPLEMENTATION", "value": "{{.Labels.modelClass}}"},
                {"name": "MLSERVER_HTTP_PORT", "value": str(Ports.REST_PORT)},
                {"name": "MLSERVER_GRPC_PORT", "value": "9000"},
                {"name": "MODELS_DIR", "value": "/mnt/models/"},
            ],
            "resources": {"requests": {"cpu": "1", "memory": "2Gi"}, "limits": {"cpu": "1", "memory": "2Gi"}},
        }
    ]

    with ServingRuntime(
        containers=containers,
        supported_model_formats=supported_model_formats,
        protocol_versions=["v2"],
        annotations={
            f"{ApiGroups.OPENDATAHUB_IO}/accelerator-name": "",
            f"{ApiGroups.OPENDATAHUB_IO}/template-display-name": "KServe MLServer",
            "prometheus.kserve.io/path": "/metrics",
            "prometheus.io/port": str(Ports.REST_PORT),
            "openshift.io/display-name": "mlserver-1.x",
        },
        label={Labels.OpenDataHub.DASHBOARD: "true"},
        teardown=teardown_resources,
        **mlserver_runtime_kwargs,
    ) as mlserver:
        yield mlserver


@pytest.fixture(scope="class")
def gaussian_credit_model(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    minio_pod: Pod,
    minio_service: Service,
    minio_data_connection: Secret,
    mlserver_runtime: ServingRuntime,
    trustyai_service_with_pvc_storage: TrustyAIService,
    teardown_resources: bool,
) -> Generator[InferenceService, Any, Any]:
    gaussian_credit_model_kwargs = {
        "client": admin_client,
        "namespace": model_namespace.name,
        "name": "gaussian-credit-model",
    }

    isvc = InferenceService(**gaussian_credit_model_kwargs)

    if pytestconfig.option.post_upgrade:
        yield isvc
        isvc.clean_up()
    else:
        with create_isvc(
            deployment_mode=KServeDeploymentType.SERVERLESS,
            model_format=XGBOOST,
            runtime=mlserver_runtime.name,
            storage_key=minio_data_connection.name,
            storage_path="sklearn/gaussian_credit_model/1",
            enable_auth=True,
            wait_for_predictor_pods=False,
            resources={"requests": {"cpu": "1", "memory": "2Gi"}, "limits": {"cpu": "1", "memory": "2Gi"}},
            teardown=teardown_resources,
            **gaussian_credit_model_kwargs,
        ) as isvc:
            wait_for_isvc_deployment_registered_by_trustyai_service(
                client=admin_client,
                isvc=isvc,
                runtime_name=mlserver_runtime.name,
            )
            yield isvc
