from typing import Generator, Any

import pytest
import yaml
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.config_map import ConfigMap
from ocp_resources.inference_service import InferenceService
from ocp_resources.maria_db import MariaDB
from ocp_resources.mariadb_operator import MariadbOperator
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.trustyai_service import TrustyAIService

from tests.model_explainability.trustyai_service.constants import (
    TAI_DATA_CONFIG,
    TAI_METRICS_CONFIG,
    TAI_PVC_STORAGE_CONFIG,
    KSERVE_MLSERVER,
    KSERVE_MLSERVER_CONTAINERS,
    KSERVE_MLSERVER_SUPPORTED_MODEL_FORMATS,
    KSERVE_MLSERVER_ANNOTATIONS,
    GAUSSIAN_CREDIT_MODEL_RESOURCES,
    GAUSSIAN_CREDIT_MODEL_STORAGE_PATH,
    XGBOOST,
    GAUSSIAN_CREDIT_MODEL,
    TAI_DB_STORAGE_CONFIG,
    ISVC_GETTER,
)
from tests.model_explainability.trustyai_service.trustyai_service_utils import (
    wait_for_isvc_deployment_registered_by_trustyai_service,
)
from tests.model_explainability.trustyai_service.utils import (
    create_trustyai_service,
    wait_for_mariadb_pods,
    TRUSTYAI_SERVICE_NAME,
    create_isvc_getter_role,
    create_isvc_getter_role_binding,
    create_isvc_getter_service_account,
    create_isvc_getter_token_secret,
)
from utilities.logger import RedactedString
from utilities.operator_utils import get_cluster_service_version
from utilities.constants import KServeDeploymentType, Labels, OPENSHIFT_OPERATORS, MARIADB
from utilities.inference_utils import create_isvc
from utilities.infra import update_configmap_data, create_inference_token

DB_CREDENTIALS_SECRET_NAME: str = "db-credentials"
DB_NAME: str = "trustyai_db"
DB_USERNAME: str = "trustyai_user"
DB_PASSWORD: str = "trustyai_password"


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

    if pytestconfig.option.post_upgrade:
        trustyai_service = TrustyAIService(**trustyai_service_kwargs)
        yield trustyai_service
        trustyai_service.clean_up()

    else:
        yield from create_trustyai_service(
            **trustyai_service_kwargs,
            storage=TAI_PVC_STORAGE_CONFIG,
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
        "name": KSERVE_MLSERVER,
    }

    if pytestconfig.option.post_upgrade:
        serving_runtime = ServingRuntime(**mlserver_runtime_kwargs)
        yield serving_runtime
        serving_runtime.clean_up()

    else:
        with ServingRuntime(
            containers=KSERVE_MLSERVER_CONTAINERS,
            supported_model_formats=KSERVE_MLSERVER_SUPPORTED_MODEL_FORMATS,
            protocol_versions=["v2"],
            annotations=KSERVE_MLSERVER_ANNOTATIONS,
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
    teardown_resources: bool,
) -> Generator[InferenceService, Any, Any]:
    gaussian_credit_model_kwargs = {
        "client": admin_client,
        "namespace": model_namespace.name,
        "name": GAUSSIAN_CREDIT_MODEL,
    }

    if pytestconfig.option.post_upgrade:
        isvc = InferenceService(**gaussian_credit_model_kwargs)
        yield isvc
        isvc.clean_up()
    else:
        with create_isvc(
            deployment_mode=KServeDeploymentType.SERVERLESS,
            model_format=XGBOOST,
            runtime=mlserver_runtime.name,
            storage_key=minio_data_connection.name,
            storage_path=GAUSSIAN_CREDIT_MODEL_STORAGE_PATH,
            enable_auth=True,
            wait_for_predictor_pods=False,
            resources=GAUSSIAN_CREDIT_MODEL_RESOURCES,
            teardown=teardown_resources,
            **gaussian_credit_model_kwargs,
        ) as isvc:
            wait_for_isvc_deployment_registered_by_trustyai_service(
                client=admin_client,
                isvc=isvc,
                runtime_name=mlserver_runtime.name,
            )
            yield isvc


@pytest.fixture(scope="class")
def isvc_getter_service_account(
    admin_client: DynamicClient, model_namespace: Namespace
) -> Generator[ServiceAccount, Any, Any]:
    yield from create_isvc_getter_service_account(client=admin_client, namespace=model_namespace, name=ISVC_GETTER)


@pytest.fixture(scope="class")
def isvc_getter_role(admin_client: DynamicClient, model_namespace: Namespace) -> Generator[Role, Any, Any]:
    yield from create_isvc_getter_role(client=admin_client, namespace=model_namespace, name=ISVC_GETTER)


@pytest.fixture(scope="class")
def isvc_getter_role_binding(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    isvc_getter_role: Role,
    isvc_getter_service_account: ServiceAccount,
) -> Generator[RoleBinding, Any, Any]:
    yield from create_isvc_getter_role_binding(
        client=admin_client,
        namespace=model_namespace,
        role=isvc_getter_role,
        service_account=isvc_getter_service_account,
        name=ISVC_GETTER,
    )


@pytest.fixture(scope="class")
def isvc_getter_token_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    isvc_getter_service_account: ServiceAccount,
    isvc_getter_role_binding: RoleBinding,
) -> Generator[Secret, Any, Any]:
    yield from create_isvc_getter_token_secret(
        client=admin_client,
        name="sa-token",
        namespace=model_namespace,
        service_account=isvc_getter_service_account,
    )


@pytest.fixture(scope="class")
def isvc_getter_token(isvc_getter_service_account: ServiceAccount, isvc_getter_token_secret: Secret) -> str:
    return RedactedString(value=create_inference_token(model_service_account=isvc_getter_service_account))
