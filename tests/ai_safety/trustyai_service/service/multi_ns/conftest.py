from collections.abc import Generator
from contextlib import ExitStack
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.trustyai_service import TrustyAIService

from tests.ai_safety.trustyai_service.constants import (
    GAUSSIAN_CREDIT_MODEL,
    GAUSSIAN_CREDIT_MODEL_RESOURCES,
    GAUSSIAN_CREDIT_MODEL_STORAGE_PATH,
    ISVC_GETTER,
    KSERVE_MLSERVER,
    KSERVE_MLSERVER_ANNOTATIONS,
    KSERVE_MLSERVER_CONTAINERS,
    KSERVE_MLSERVER_SUPPORTED_MODEL_FORMATS,
    TAI_DATA_CONFIG,
    TAI_DB_STORAGE_CONFIG,
    TAI_METRICS_CONFIG,
    TAI_PVC_STORAGE_CONFIG,
    XGBOOST,
)
from tests.ai_safety.trustyai_service.trustyai_service_utils import (
    wait_for_isvc_deployment_registered_by_trustyai_service,
)
from tests.ai_safety.trustyai_service.utils import (
    create_isvc_getter_role,
    create_isvc_getter_role_binding,
    create_isvc_getter_service_account,
    create_isvc_getter_token_secret,
    create_standalone_mariadb,
    create_trustyai_service,
)
from utilities.constants import TRUSTYAI_SERVICE_NAME, KServeDeploymentType
from utilities.inference_utils import create_isvc
from utilities.infra import create_inference_token, create_ns
from utilities.minio import create_minio_data_connection_secret

DB_CREDENTIALS_SECRET_NAME: str = "db-credentials"
DB_NAME: str = "trustyai_db"
DB_USERNAME: str = "trustyai_user"
DB_PASSWORD: str = "trustyai_password"


@pytest.fixture(scope="class")
def model_namespaces(request, admin_client) -> Generator[list[Namespace], Any]:
    with ExitStack() as stack:
        namespaces = [
            stack.enter_context(create_ns(admin_client=admin_client, name=param["name"])) for param in request.param
        ]
        yield namespaces


@pytest.fixture(scope="class")
def minio_data_connection_multi_ns(
    request, admin_client, model_namespaces, minio_service
) -> Generator[list[Secret], Any]:
    with ExitStack() as stack:
        secrets = [
            stack.enter_context(
                create_minio_data_connection_secret(
                    minio_service=minio_service,
                    model_namespace=ns.name,
                    aws_s3_bucket=param["bucket"],
                    client=admin_client,
                )
            )
            for ns, param in zip(model_namespaces, request.param)
        ]
        yield secrets


@pytest.fixture(scope="class")
def trustyai_service_with_pvc_storage_multi_ns(
    admin_client, model_namespaces, cluster_monitoring_config, user_workload_monitoring_config
) -> Generator[list[TrustyAIService], Any]:
    with ExitStack() as stack:
        services = [
            stack.enter_context(
                create_trustyai_service(
                    client=admin_client,
                    namespace=ns.name,
                    name=TRUSTYAI_SERVICE_NAME,
                    storage=TAI_PVC_STORAGE_CONFIG,
                    metrics=TAI_METRICS_CONFIG,
                    data=TAI_DATA_CONFIG,
                    wait_for_replicas=True,
                    teardown=False,
                )
            )
            for ns in model_namespaces
        ]
        yield services


@pytest.fixture(scope="class")
def kserve_logger_ca_bundle_multi_ns(
    admin_client: DynamicClient, model_namespaces: list[Namespace]
) -> Generator[list[ConfigMap], Any]:
    """Create CA certificate ConfigMaps required for KServeRaw logger in each namespace."""
    with ExitStack() as stack:
        ca_bundles = [
            stack.enter_context(
                ConfigMap(
                    client=admin_client,
                    name="kserve-logger-ca-bundle",
                    namespace=ns.name,
                    annotations={"service.beta.openshift.io/inject-cabundle": "true"},
                    data={},
                    teardown=False,
                )
            )
            for ns in model_namespaces
        ]
        yield ca_bundles


@pytest.fixture(scope="class")
def mlserver_runtime_multi_ns(admin_client, model_namespaces) -> Generator[list[ServingRuntime], Any]:
    with ExitStack() as stack:
        runtimes = [
            stack.enter_context(
                ServingRuntime(
                    client=admin_client,
                    namespace=ns.name,
                    name=KSERVE_MLSERVER,
                    containers=KSERVE_MLSERVER_CONTAINERS,
                    supported_model_formats=KSERVE_MLSERVER_SUPPORTED_MODEL_FORMATS,
                    protocol_versions=["v2"],
                    annotations=KSERVE_MLSERVER_ANNOTATIONS,
                    label={"opendatahub.io/dashboard": "true"},
                    teardown=False,
                )
            )
            for ns in model_namespaces
        ]
        yield runtimes


@pytest.fixture(scope="class")
def gaussian_credit_model_multi_ns(
    admin_client: DynamicClient,
    model_namespaces: list[Namespace],
    minio_pod: Pod,
    minio_service: Service,
    minio_data_connection_multi_ns: list[Secret],
    mlserver_runtime_multi_ns: list[ServingRuntime],
    kserve_raw_config: ConfigMap,
    kserve_logger_ca_bundle_multi_ns: list[ConfigMap],
) -> Generator[list[InferenceService], Any]:
    with ExitStack() as stack:
        models = []
        for ns, secret, runtime in zip(model_namespaces, minio_data_connection_multi_ns, mlserver_runtime_multi_ns):
            isvc_context = create_isvc(
                client=admin_client,
                namespace=ns.name,
                name=GAUSSIAN_CREDIT_MODEL,
                deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
                model_format=XGBOOST,
                runtime=runtime.name,
                storage_key=secret.name,
                storage_path=GAUSSIAN_CREDIT_MODEL_STORAGE_PATH,
                enable_auth=True,
                external_route=True,
                wait_for_predictor_pods=False,
                resources=GAUSSIAN_CREDIT_MODEL_RESOURCES,
            )
            isvc = stack.enter_context(cm=isvc_context)

            wait_for_isvc_deployment_registered_by_trustyai_service(
                client=admin_client,
                isvc=isvc,
                runtime_name=runtime.name,
            )

            models.append(isvc)

        yield models


@pytest.fixture(scope="class")
def isvc_getter_service_account_multi_ns(admin_client, model_namespaces) -> Generator[list[ServiceAccount]]:
    with ExitStack() as stack:
        sas = [
            stack.enter_context(create_isvc_getter_service_account(admin_client, ns, ISVC_GETTER))
            for ns in model_namespaces
        ]
        yield sas


@pytest.fixture(scope="class")
def isvc_getter_role_multi_ns(admin_client, model_namespaces) -> Generator[list[Role]]:
    with ExitStack() as stack:
        roles = [
            stack.enter_context(create_isvc_getter_role(admin_client, ns, f"isvc-getter-{ns.name}"))
            for ns in model_namespaces
        ]
        yield roles


@pytest.fixture(scope="class")
def isvc_getter_role_binding_multi_ns(
    admin_client,
    model_namespaces,
    isvc_getter_role_multi_ns,
    isvc_getter_service_account_multi_ns,
) -> Generator[list[RoleBinding]]:
    with ExitStack() as stack:
        bindings = [
            stack.enter_context(
                create_isvc_getter_role_binding(
                    client=admin_client,
                    namespace=ns,
                    role=role,
                    service_account=sa,
                    name=ISVC_GETTER,
                )
            )
            for ns, role, sa in zip(model_namespaces, isvc_getter_role_multi_ns, isvc_getter_service_account_multi_ns)
        ]
        yield bindings


@pytest.fixture(scope="class")
def isvc_getter_token_secret_multi_ns(
    admin_client,
    model_namespaces,
    isvc_getter_service_account_multi_ns,
    isvc_getter_role_binding_multi_ns,
) -> Generator[list[Secret]]:
    with ExitStack() as stack:
        secrets = [
            stack.enter_context(
                create_isvc_getter_token_secret(
                    client=admin_client,
                    namespace=ns,
                    name=f"sa-token-{ns.name}",
                    service_account=sa,
                )
            )
            for ns, sa in zip(model_namespaces, isvc_getter_service_account_multi_ns)
        ]
        yield secrets


@pytest.fixture(scope="class")
def isvc_getter_token_multi_ns(
    isvc_getter_service_account_multi_ns,
    isvc_getter_token_secret_multi_ns,
) -> list[str]:
    return [create_inference_token(model_service_account=sa) for sa in isvc_getter_service_account_multi_ns]


@pytest.fixture(scope="class")
def trustyai_service_with_db_storage_multi_ns(
    admin_client,
    model_namespaces,
    cluster_monitoring_config,
    user_workload_monitoring_config,
    mariadb_multi_ns,
    trustyai_db_ca_secret_multi_ns: None,
) -> Generator[list[TrustyAIService], Any]:
    with ExitStack() as stack:
        services = [
            stack.enter_context(
                create_trustyai_service(
                    client=admin_client,
                    namespace=ns.name,
                    name=TRUSTYAI_SERVICE_NAME,
                    storage=TAI_DB_STORAGE_CONFIG,
                    metrics=TAI_METRICS_CONFIG,
                    data=TAI_DATA_CONFIG,
                    wait_for_replicas=True,
                )
            )
            for ns in model_namespaces
        ]
        yield services


@pytest.fixture(scope="class")
def trustyai_db_ca_secret_multi_ns(
    admin_client,
    model_namespaces: list[Namespace],
    mariadb_multi_ns: list[Deployment],
) -> Generator[list[Secret]]:
    """Creates one trustyai-db-ca secret per namespace, using the corresponding MariaDB CA cert."""
    with ExitStack() as stack:
        secrets = []

        for ns in model_namespaces:
            mariadb_ca_secret = Secret(
                client=admin_client,
                name="mariadb-ca",
                namespace=ns.name,
                ensure_exists=True,
            )
            ca_cert = mariadb_ca_secret.instance.data["ca.crt"]

            secret = stack.enter_context(
                cm=Secret(
                    client=admin_client,
                    name=f"{TRUSTYAI_SERVICE_NAME}-db-ca",
                    namespace=ns.name,
                    data_dict={"ca.crt": ca_cert},
                    teardown=True,
                )
            )
            secrets.append(secret)
        yield secrets


@pytest.fixture(scope="class")
def db_credentials_secret_multi_ns(admin_client, model_namespaces: list[Namespace]) -> Generator[list[Secret]]:
    """Creates DB credentials Secret in each model namespace."""
    with ExitStack() as stack:
        secrets = []

        for ns in model_namespaces:
            secret = stack.enter_context(
                cm=Secret(
                    client=admin_client,
                    name=DB_CREDENTIALS_SECRET_NAME,
                    namespace=ns.name,
                    string_data={
                        "databaseKind": "mariadb",
                        "databaseName": DB_NAME,
                        "databaseUsername": DB_USERNAME,
                        "databasePassword": DB_PASSWORD,
                        "databaseService": "mariadb",
                        "databasePort": "3306",
                        "databaseGeneration": "update",
                    },
                    teardown=True,
                )
            )
            secrets.append(secret)
        yield secrets


@pytest.fixture(scope="class")
def mariadb_multi_ns(
    admin_client: DynamicClient,
    model_namespaces: list[Namespace],
    db_credentials_secret_multi_ns: list[Secret],
) -> Generator[list[Deployment], Any, Any]:
    """Provides MariaDB instances using standalone Deployments with TLS in each namespace."""
    mariadb_instances: list[Deployment] = []

    with ExitStack() as stack:
        for ns in model_namespaces:
            deployment = stack.enter_context(
                cm=create_standalone_mariadb(
                    client=admin_client,
                    namespace_name=ns.name,
                    name="mariadb",
                    db_credentials_secret_name=DB_CREDENTIALS_SECRET_NAME,
                    teardown=True,
                )
            )
            mariadb_instances.append(deployment)
        yield mariadb_instances
