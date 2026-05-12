import json
import secrets
from collections.abc import Generator
from typing import Any

import pytest
import yaml
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.config_map import ConfigMap
from ocp_resources.inference_service import InferenceService
from ocp_resources.maria_db import MariaDB
from ocp_resources.mariadb_operator import MariadbOperator
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.trustyai_service import TrustyAIService
from pytest_testconfig import py_config

from tests.model_explainability.trustyai_service.constants import (
    GAUSSIAN_CREDIT_MODEL,
    GAUSSIAN_CREDIT_MODEL_RESOURCES,
    GAUSSIAN_CREDIT_MODEL_STORAGE_URI,
    ISVC_GETTER,
    KSERVE_MLSERVER,
    TAI_DATA_CONFIG,
    TAI_DB_STORAGE_CONFIG,
    TAI_METRICS_CONFIG,
    TAI_PVC_STORAGE_CONFIG,
    XGBOOST,
)
from tests.model_explainability.trustyai_service.trustyai_service_utils import (
    wait_for_isvc_deployment_registered_by_trustyai_service,
)
from tests.model_explainability.trustyai_service.utils import (
    create_isvc_getter_role,
    create_isvc_getter_role_binding,
    create_isvc_getter_service_account,
    create_isvc_getter_token_secret,
    create_trustyai_service,
    wait_for_mariadb_pods,
)
from utilities.constants import (
    MARIADB,
    OPENSHIFT_OPERATORS,
    TRUSTYAI_SERVICE_NAME,
    Annotations,
    KServeDeploymentType,
    RuntimeTemplates,
)
from utilities.inference_utils import create_isvc
from utilities.infra import create_inference_token, get_kserve_storage_initialize_image, update_configmap_data
from utilities.logger import RedactedString
from utilities.operator_utils import get_cluster_service_version
from utilities.serving_runtime import ServingRuntimeFromTemplate

DB_CREDENTIALS_SECRET_NAME: str = "db-credentials"
DB_NAME: str = "trustyai_db"
DB_USERNAME: str = "trustyai_user"
DB_PASSWORD: str = "trustyai_password"


@pytest.fixture(scope="class")
def trustyai_service(
    request: FixtureRequest,
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    cluster_monitoring_config: ConfigMap,
    user_workload_monitoring_config: ConfigMap,
    teardown_resources: bool,
) -> Generator[TrustyAIService, Any, Any]:
    """Provides a TrustyAIService instance for testing.

    In post-upgrade mode, references the existing TrustyAIService created during pre-upgrade tests
    and cleans it up after post-upgrade tests complete.

    In pre-upgrade mode (or when no upgrade flag is set), creates a new TrustyAIService with either
    PVC or DB storage based on the test parametrization, and manages cleanup via teardown_resources.
    """
    tais_kwargs = {"client": admin_client, "namespace": model_namespace.name, "name": TRUSTYAI_SERVICE_NAME}

    if pytestconfig.option.post_upgrade:
        trustyai_service = TrustyAIService(**tais_kwargs)
        yield trustyai_service
        trustyai_service.clean_up()
    else:
        if request.param["storage"] == "pvc":
            tais_kwargs["storage"] = TAI_PVC_STORAGE_CONFIG
            tais_kwargs["data"] = TAI_DATA_CONFIG
        elif request.param["storage"] == "db":
            request.getfixturevalue("mariadb")
            request.getfixturevalue("trustyai_db_ca_secret")
            tais_kwargs["storage"] = TAI_DB_STORAGE_CONFIG
        else:
            raise ValueError("TrustyAI storage can only be 'pvc' or 'db'")

        with create_trustyai_service(
            **tais_kwargs,
            metrics=TAI_METRICS_CONFIG,
            wait_for_replicas=True,
            teardown=teardown_resources,
        ) as trustyai_service:
            yield trustyai_service


@pytest.fixture(scope="session")
def kserve_raw_config(admin_client: DynamicClient) -> Generator[ConfigMap, Any, Any]:
    """Configure KServe for KServeRaw support by adding logger configuration."""

    storage_initializer_image = get_kserve_storage_initialize_image(client=admin_client)
    logger_config = {
        "image": storage_initializer_image,
        "memoryRequest": "100Mi",
        "memoryLimit": "1Gi",
        "cpuRequest": "100m",
        "cpuLimit": "1",
        "defaultUrl": "http://default-broker",
        "caBundle": "kserve-logger-ca-bundle",
        "caCertFile": "service-ca.crt",
        "tlsSkipVerify": False,
    }

    data = {"logger": json.dumps(obj=logger_config)}

    cm = ConfigMap(
        client=admin_client,
        name="inferenceservice-config",
        namespace=py_config["applications_namespace"],
        ensure_exists=True,
    )

    with ResourceEditor(
        patches={
            cm: {
                "metadata": {"annotations": {Annotations.OpenDataHubIo.MANAGED: "false"}},
                "data": data,
            }
        }
    ):
        yield cm


@pytest.fixture(scope="class")
def kserve_logger_ca_bundle(admin_client: DynamicClient, model_namespace: Namespace) -> Generator[ConfigMap, Any, Any]:
    """Create CA certificate ConfigMap required for KServeRaw logger."""
    with ConfigMap(
        client=admin_client,
        name="kserve-logger-ca-bundle",
        namespace=model_namespace.name,
        annotations={"service.beta.openshift.io/inject-cabundle": "true"},
        data={},
    ) as ca_bundle:
        yield ca_bundle


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
def db_credentials_secret(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[Secret, Any, Any]:
    """Provides database credentials secret for MariaDB connection.

    In post-upgrade mode, references the existing secret created during pre-upgrade tests
    and cleans it up after post-upgrade tests complete.

    In pre-upgrade mode (or when no upgrade flag is set), creates a new secret with MariaDB
    connection details and manages cleanup via teardown_resources.
    """
    if pytestconfig.option.post_upgrade:
        secret = Secret(
            client=admin_client,
            name=DB_CREDENTIALS_SECRET_NAME,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        yield secret
        secret.clean_up()
    else:
        db_password = secrets.token_urlsafe(nbytes=24)
        with Secret(
            client=admin_client,
            name=DB_CREDENTIALS_SECRET_NAME,
            namespace=model_namespace.name,
            string_data={
                "databaseKind": MARIADB,
                "databaseName": DB_NAME,
                "databaseUsername": DB_USERNAME,
                "databasePassword": db_password,
                "databaseService": MARIADB,
                "databasePort": "3306",
                "databaseGeneration": "update",
            },
            teardown=teardown_resources,
        ) as db_credentials:
            yield db_credentials


@pytest.fixture(scope="class")
def mariadb(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    db_credentials_secret: Secret,
    mariadb_operator_cr: MariadbOperator,
    teardown_resources: bool,
) -> Generator[MariaDB, Any, Any]:
    """Provides a MariaDB instance for TrustyAI database storage.

    In post-upgrade mode, references the existing MariaDB created during pre-upgrade tests
    and cleans it up after post-upgrade tests complete.

    In pre-upgrade mode (or when no upgrade flag is set), creates a new MariaDB instance,
    waits for pods to be ready, and cleans up.
    """
    if pytestconfig.option.post_upgrade:
        mariadb = MariaDB(
            client=admin_client,
            name="mariadb",
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        yield mariadb
        mariadb.clean_up()
    else:
        mariadb_csv: ClusterServiceVersion = get_cluster_service_version(
            client=admin_client, prefix=MARIADB, namespace=OPENSHIFT_OPERATORS
        )
        alm_examples: list[dict[str, Any]] = mariadb_csv.get_alm_examples()
        mariadb_dict: dict[str, Any] = next((example for example in alm_examples if example["kind"] == "MariaDB"), None)

        if not mariadb_dict:
            raise ResourceNotFoundError(f"No MariaDB dict found in alm_examples for CSV {mariadb_csv.name}")

        mariadb_dict["metadata"]["namespace"] = model_namespace.name
        mariadb_dict["spec"]["database"] = DB_NAME
        mariadb_dict["spec"]["username"] = DB_USERNAME

        mariadb_dict["spec"]["replicas"] = 1

        # Need to fix MariaDB version due to an issue with the default version in certain environments
        # Using the same registry and image used by the MariaDB operator
        # --just changing the tag to point to a stable version
        mariadb_dict["spec"]["image"] = "docker-registry1.mariadb.com/library/mariadb:10.11.8"
        mariadb_dict["spec"]["galera"]["enabled"] = False
        mariadb_dict["spec"]["metrics"]["enabled"] = False
        mariadb_dict["spec"]["tls"] = {"enabled": True, "required": True}

        password_secret_key_ref = {"generate": False, "key": "databasePassword", "name": DB_CREDENTIALS_SECRET_NAME}

        mariadb_dict["spec"]["rootPasswordSecretKeyRef"] = password_secret_key_ref
        mariadb_dict["spec"]["passwordSecretKeyRef"] = password_secret_key_ref
        with MariaDB(kind_dict=mariadb_dict, teardown=teardown_resources) as mariadb:
            wait_for_mariadb_pods(client=admin_client, mariadb=mariadb)
            yield mariadb


@pytest.fixture(scope="class")
def trustyai_db_ca_secret(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mariadb: MariaDB,
    teardown_resources: bool,
) -> Generator[Secret, Any]:
    """Provides TLS CA certificate secret for TrustyAI to connect to MariaDB.

    In post-upgrade mode, references the existing CA secret created during pre-upgrade tests
    and cleans it up after post-upgrade tests complete.

    In pre-upgrade mode (or when no upgrade flag is set), creates a new secret by copying the
    CA certificate from the MariaDB CA secret and manages cleanup via teardown_resources.
    """
    if pytestconfig.option.post_upgrade:
        secret = Secret(
            client=admin_client,
            name=f"{TRUSTYAI_SERVICE_NAME}-db-ca",
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        yield secret
        secret.clean_up()
    else:
        mariadb_ca_secret = Secret(
            client=admin_client, name=f"{mariadb.name}-ca", namespace=model_namespace.name, ensure_exists=True
        )
        with Secret(
            client=admin_client,
            name=f"{TRUSTYAI_SERVICE_NAME}-db-ca",
            namespace=model_namespace.name,
            data_dict={"ca.crt": mariadb_ca_secret.instance.data["ca.crt"]},
            teardown=teardown_resources,
        ) as secret:
            yield secret


@pytest.fixture(scope="class")
def mlserver_runtime(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
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
        with ServingRuntimeFromTemplate(
            template_name=RuntimeTemplates.MLSERVER,
            deployment_type=KServeDeploymentType.RAW_DEPLOYMENT,
            teardown=teardown_resources,
            **mlserver_runtime_kwargs,
        ) as mlserver:
            yield mlserver


@pytest.fixture(scope="class")
def gaussian_credit_model(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_runtime: ServingRuntime,
    # kserve_raw_config: ConfigMap,
    kserve_logger_ca_bundle: ConfigMap,
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
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            model_format=XGBOOST,
            runtime=mlserver_runtime.name,
            storage_uri=GAUSSIAN_CREDIT_MODEL_STORAGE_URI,
            enable_auth=True,
            external_route=True,
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
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[ServiceAccount, Any, Any]:
    """Provides a ServiceAccount for fetching InferenceServices.

    In post-upgrade mode, references the existing ServiceAccount created during pre-upgrade tests
    and cleans it up after post-upgrade tests complete.

    In pre-upgrade mode (or when no upgrade flag is set), creates a new ServiceAccount and manages
    cleanup.
    """
    if pytestconfig.option.post_upgrade:
        service_account = ServiceAccount(
            client=admin_client,
            name=ISVC_GETTER,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        yield service_account
        service_account.clean_up()
    else:
        with create_isvc_getter_service_account(
            client=admin_client, namespace=model_namespace, name=ISVC_GETTER, teardown=teardown_resources
        ) as service_account:
            yield service_account


@pytest.fixture(scope="class")
def isvc_getter_role(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[Role, Any, Any]:
    """Provides a Role with permissions to get, list, and watch InferenceServices.

    In post-upgrade mode, references the existing Role created during pre-upgrade tests
    and cleans it up after post-upgrade tests complete.

    In pre-upgrade mode (or when no upgrade flag is set), creates a new Role with InferenceService
    access permissions and manages cleanup.
    """
    if pytestconfig.option.post_upgrade:
        role = Role(
            client=admin_client,
            name=ISVC_GETTER,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        yield role
        role.clean_up()
    else:
        with create_isvc_getter_role(
            client=admin_client, namespace=model_namespace, name=ISVC_GETTER, teardown=teardown_resources
        ) as role:
            yield role


@pytest.fixture(scope="class")
def isvc_getter_role_binding(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    isvc_getter_role: Role,
    isvc_getter_service_account: ServiceAccount,
    teardown_resources: bool,
) -> Generator[RoleBinding, Any, Any]:
    """Provides a RoleBinding to link the ServiceAccount to the InferenceService getter Role.

    In post-upgrade mode, references the existing RoleBinding created during pre-upgrade tests
    and cleans it up after post-upgrade tests complete.

    In pre-upgrade mode (or when no upgrade flag is set), creates a new RoleBinding
    and manages cleanup.
    """
    if pytestconfig.option.post_upgrade:
        role_binding = RoleBinding(
            client=admin_client,
            name=ISVC_GETTER,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        yield role_binding
        role_binding.clean_up()
    else:
        with create_isvc_getter_role_binding(
            client=admin_client,
            namespace=model_namespace,
            role=isvc_getter_role,
            service_account=isvc_getter_service_account,
            name=ISVC_GETTER,
            teardown=teardown_resources,
        ) as role_binding:
            yield role_binding


@pytest.fixture(scope="class")
def isvc_getter_token_secret(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    isvc_getter_service_account: ServiceAccount,
    isvc_getter_role_binding: RoleBinding,
    teardown_resources: bool,
) -> Generator[Secret, Any, Any]:
    """Provides a token Secret for the InferenceService getter ServiceAccount.

    In post-upgrade mode, references the existing token Secret created during pre-upgrade tests
    and cleans it up after post-upgrade tests complete.

    In pre-upgrade mode (or when no upgrade flag is set), creates a new token Secret and manages
    cleanup via teardown_resources.
    """
    if pytestconfig.option.post_upgrade:
        secret = Secret(
            client=admin_client,
            name="sa-token",
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        yield secret
        secret.clean_up()
    else:
        with create_isvc_getter_token_secret(
            client=admin_client,
            name="sa-token",
            namespace=model_namespace,
            service_account=isvc_getter_service_account,
            teardown=teardown_resources,
        ) as secret:
            yield secret


@pytest.fixture(scope="class")
def isvc_getter_token(isvc_getter_service_account: ServiceAccount, isvc_getter_token_secret: Secret) -> str:
    return RedactedString(value=create_inference_token(model_service_account=isvc_getter_service_account))
