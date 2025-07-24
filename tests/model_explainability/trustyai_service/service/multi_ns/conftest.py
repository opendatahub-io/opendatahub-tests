import copy
from contextlib import ExitStack
from typing import Generator, Any, List
import pytest
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.inference_service import InferenceService
from ocp_resources.trustyai_service import TrustyAIService
from ocp_resources.service_account import ServiceAccount
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from utilities.constants import KServeDeploymentType, Labels, OPENSHIFT_OPERATORS, MARIADB
from ocp_resources.maria_db import MariaDB
from tests.model_explainability.trustyai_service.constants import (
    TAI_METRICS_CONFIG,
    TAI_DATA_CONFIG,
    TAI_PVC_STORAGE_CONFIG,
    GAUSSIAN_CREDIT_MODEL_STORAGE_PATH,
    GAUSSIAN_CREDIT_MODEL_RESOURCES,
    KSERVE_MLSERVER,
    KSERVE_MLSERVER_CONTAINERS,
    KSERVE_MLSERVER_SUPPORTED_MODEL_FORMATS,
    KSERVE_MLSERVER_ANNOTATIONS,
    XGBOOST,
    ISVC_GETTER,
    GAUSSIAN_CREDIT_MODEL, TAI_DB_STORAGE_CONFIG,
)
from tests.model_explainability.trustyai_service.trustyai_service_utils import (
    TRUSTYAI_SERVICE_NAME,
    wait_for_isvc_deployment_registered_by_trustyai_service,
)
from tests.model_explainability.trustyai_service.utils import (
    create_trustyai_service,
    create_isvc_getter_service_account,
    create_isvc_getter_role,
    create_isvc_getter_role_binding,
    create_isvc_getter_token_secret,
    wait_for_mariadb_pods,
)
from utilities.constants import KServeDeploymentType
from utilities.inference_utils import create_isvc
from utilities.infra import create_inference_token
from utilities.minio import create_minio_data_connection_secret
from utilities.operator_utils import get_cluster_service_version
from ocp_resources.cluster_service_version import ClusterServiceVersion
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from kubernetes.dynamic import DynamicClient

DB_CREDENTIALS_SECRET_NAME: str = "db-credentials"
DB_NAME: str = "trustyai_db"
DB_USERNAME: str = "trustyai_user"
DB_PASSWORD: str = "trustyai_password"


@pytest.fixture(scope="class")
def model_namespaces(request, admin_client, teardown_resources) -> Generator[List[Namespace], Any, None]:
    # if any(param.get("modelmesh-enabled") for param in request.param):
    #     request.getfixturevalue(argname="enabled_modelmesh_in_dsc")

    namespaces = []
    for param in request.param:
        ns = Namespace(client=admin_client, name=param["name"], teardown=teardown_resources)
        ns.deploy()
        namespaces.append(ns)

    yield namespaces

    for ns in namespaces:
        ns.clean_up()


@pytest.fixture(scope="class")
def minio_data_connection_multi_ns(
    request, admin_client, model_namespaces, minio_service
) -> Generator[List[Secret], Any, None]:
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
) -> Generator[List[TrustyAIService], Any, None]:
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
def mlserver_runtime_multi_ns(admin_client, model_namespaces) -> Generator[List[ServingRuntime], Any, None]:
    runtimes = []
    for ns in model_namespaces:
        runtime = ServingRuntime(
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
        runtime.deploy()
        runtimes.append(runtime)

    yield runtimes
    for runtime in runtimes:
        runtime.clean_up()


@pytest.fixture(scope="class")
def gaussian_credit_model_multi_ns(
    admin_client,
    model_namespaces,
    minio_pod,
    minio_service,
    minio_data_connection_multi_ns,
    mlserver_runtime_multi_ns,
    trustyai_service_with_db_storage_multi_ns, #check this and remove if not needed
) -> Generator[List[InferenceService], Any, None]:
    with ExitStack() as stack:
        models = []
        for ns, secret, runtime in zip(model_namespaces, minio_data_connection_multi_ns, mlserver_runtime_multi_ns):
            isvc_context = create_isvc(
                client=admin_client,
                namespace=ns.name,
                name=GAUSSIAN_CREDIT_MODEL,
                deployment_mode=KServeDeploymentType.SERVERLESS,
                model_format=XGBOOST,
                runtime=runtime.name,
                storage_key=secret.name,
                storage_path=GAUSSIAN_CREDIT_MODEL_STORAGE_PATH,
                enable_auth=True,
                wait_for_predictor_pods=False,
                resources=GAUSSIAN_CREDIT_MODEL_RESOURCES,
            )
            isvc = stack.enter_context(isvc_context)  # noqa: FCN001

            wait_for_isvc_deployment_registered_by_trustyai_service(
                client=admin_client,
                isvc=isvc,
                runtime_name=runtime.name,
            )

            models.append(isvc)

        yield models


@pytest.fixture(scope="class")
def trustyai_service_with_db_storage_multi_ns(
    admin_client,
    model_namespaces,
    cluster_monitoring_config,
    user_workload_monitoring_config,
    mariadb_multi_ns,
    trustyai_db_ca_secret_multi_ns: None,
) -> Generator[List[TrustyAIService], Any, None]:
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
    model_namespaces: List[Namespace],
    mariadb_multi_ns: List,  # Fix: get the list of per-NS MariaDBs, not a single 'mariadb'
) -> Generator[List[Secret], Any, None]:
    """
    Creates one trustyai-db-ca secret per namespace, using the corresponding MariaDB CA cert.
    """
    secrets = []

    for ns, mariadb_ns in zip(model_namespaces, mariadb_multi_ns):
        # Ensure the MariaDB CA secret exists in this namespace
        mariadb_ca_secret = Secret(
            client=admin_client,
            name=f"{mariadb_ns.name}-ca",
            namespace=ns.name,
            ensure_exists=True,
        )

        # Create the trustyai-db-ca secret using the CA cert
        secret = Secret(
            client=admin_client,
            name=f"{TRUSTYAI_SERVICE_NAME}-db-ca",
            namespace=ns.name,
            data_dict={"ca.crt": mariadb_ca_secret.instance.data["ca.crt"]},
            teardown=True,
        )
        secret.deploy()
        secrets.append(secret)

    yield secrets

    for secret in secrets:
        secret.clean_up()

# @pytest.fixture(scope="class")
# def trustyai_db_secret_multi_ns(
#     admin_client,
#     model_namespaces: List[Namespace],
#     mariadb
# ) -> Generator[List[Secret], Any, None]:
#     """Creates one trustyai-db secret per namespace with DB credentials for connecting to MariaDB."""
#     secrets = []
#
#     for ns in model_namespaces:
#         secret = Secret(
#             client=admin_client,
#             name=f"{TRUSTYAI_SERVICE_NAME}-db",
#             namespace=ns.name,
#             data_dict={
#                 "db.name": mariadb.db_name,
#                 "db.user": mariadb.db_user,
#                 "db.password": mariadb.db_password,
#                 "db.host": mariadb.host,
#                 "db.port": str(mariadb.port),
#                 "db.protocol": "mariadb",
#             },
#             teardown=True,
#         )
#         secret.deploy()
#         secrets.append(secret)
#
#     yield secrets
#
#     for secret in secrets:
#         secret.clean_up()


@pytest.fixture(scope="class")
def isvc_getter_service_account_multi_ns(admin_client, model_namespaces) -> Generator[List[ServiceAccount], None, None]:
    with ExitStack() as stack:
        sas = [
            stack.enter_context(create_isvc_getter_service_account(admin_client, ns, ISVC_GETTER))
            for ns in model_namespaces
        ]
        yield sas


@pytest.fixture(scope="class")
def isvc_getter_role_multi_ns(admin_client, model_namespaces) -> Generator[List[Role], None, None]:
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
) -> Generator[List[RoleBinding], None, None]:
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
) -> Generator[List[Secret], None, None]:
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
) -> List[str]:
    return [create_inference_token(model_service_account=sa) for sa in isvc_getter_service_account_multi_ns]

@pytest.fixture(scope="class")
def db_credentials_secret_multi_ns(admin_client, model_namespaces: List[Namespace]) -> Generator[List[Secret], Any, None]:
    """Creates DB credentials Secret in each model namespace."""
    secrets = []

    for ns in model_namespaces:
        secret = Secret(
            client=admin_client,
            name=DB_CREDENTIALS_SECRET_NAME,
            namespace=ns.name,
            string_data={
                "databaseKind": "mariadb",
                "databaseName": DB_NAME,
                "databaseUsername": DB_USERNAME,
                "databasePassword": DB_PASSWORD,
                "databaseService": f"trustyai-db-{ns.name}",
                "databasePort": "3306",
                "databaseGeneration": "update",
            },
        )
        secret.deploy()
        secrets.append(secret)

    yield secrets


@pytest.fixture(scope="class")
def mariadb_multi_ns(
    admin_client,
    model_namespaces: List[Namespace],
    db_credentials_secret_multi_ns: List[Secret],
    mariadb_operator_cr,
) -> Generator[List[MariaDB], Any, None]:
    """Creates one MariaDB instance per namespace using matching DB credentials secrets."""
    mariadb_csv: ClusterServiceVersion = get_cluster_service_version(
        client=admin_client, prefix=MARIADB, namespace=OPENSHIFT_OPERATORS
    )
    alm_examples: list[dict[str, Any]] = mariadb_csv.get_alm_examples()
    mariadb_template: dict[str, Any] = next(example for example in alm_examples if example["kind"] == "MariaDB")

    if not mariadb_template:
        raise ResourceNotFoundError(f"No MariaDB dict found in alm_examples for CSV {mariadb_csv.name}")

    mariadb_instances: List[MariaDB] = []
    errors: List[str] = []

    for ns, secret in zip(model_namespaces, db_credentials_secret_multi_ns):
        try:
            mariadb_spec = copy.deepcopy(mariadb_template)
            mariadb_spec["metadata"]["namespace"] = ns.name
            mariadb_spec["metadata"]["name"] = f"trustyai-db-{ns.name}"
            mariadb_spec["spec"]["database"] = DB_NAME
            mariadb_spec["spec"]["username"] = DB_USERNAME
            mariadb_spec["spec"]["replicas"] = 1
            mariadb_spec["spec"]["storage"] = {"size": "1Gi"}
            mariadb_spec["spec"]["galera"]["enabled"] = False
            mariadb_spec["spec"]["metrics"]["enabled"] = False
            mariadb_spec["spec"]["tls"] = {"enabled": True, "required": True}

            password_secret_key_ref = {
                "generate": False,
                "key": "databasePassword",
                "name": DB_CREDENTIALS_SECRET_NAME,
            }
            mariadb_spec["spec"]["rootPasswordSecretKeyRef"] = password_secret_key_ref
            mariadb_spec["spec"]["passwordSecretKeyRef"] = password_secret_key_ref

            mariadb_ns = MariaDB(kind_dict=mariadb_spec)
            mariadb_ns.deploy()
            print(f"Waiting for MariaDB pods in namespace {ns.name} ...")
            wait_for_mariadb_pods(client=admin_client, mariadb=mariadb_ns)
            mariadb_instances.append(mariadb_ns)
        except Exception as e:
            errors.append(f"{ns.name}: {e}")
            # Optionally: still append so cleanup happens
            mariadb_instances.append(None)

    # If any errors accumulated, raise them here after all attempts
    if errors:
        raise RuntimeError("MariaDB bringup errors:\n" + "\n".join(errors))

    yield mariadb_instances

    for mdb in mariadb_instances:
        if mdb:
            mdb.clean_up()