from typing import Generator, Any

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.inference_service import InferenceService
from ocp_resources.maria_db import MariaDB
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
    TAI_METRICS_CONFIG,
    TAI_DATA_CONFIG,
    TAI_PVC_STORAGE_CONFIG,
    GAUSSIAN_CREDIT_MODEL,
    GAUSSIAN_CREDIT_MODEL_STORAGE_PATH,
    GAUSSIAN_CREDIT_MODEL_RESOURCES,
    KSERVE_MLSERVER,
    KSERVE_MLSERVER_CONTAINERS,
    KSERVE_MLSERVER_SUPPORTED_MODEL_FORMATS,
    KSERVE_MLSERVER_ANNOTATIONS,
    XGBOOST,
    TAI_DB_STORAGE_CONFIG,
    ISVC_GETTER,
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
)
from utilities.constants import KServeDeploymentType, Labels
from utilities.inference_utils import create_isvc
from utilities.infra import create_ns, create_inference_token
from utilities.minio import create_minio_data_connection_secret


INVALID_TLS_CERTIFICATE: str = "LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUJnRENDQVNlZ0F3SUJBZ0lRRGtTcXVuUWRzRmZwdi8zSm\
5TS2ZoVEFLQmdncWhrak9QUVFEQWpBVk1STXcKRVFZRFZRUURFd3B0WVhKcFlXUmlMV05oTUI0WERUSTFNRFF4TkRFME1EUXhOMW9YRFRJNE1EUXhNekUx\
TURReApOMW93RlRFVE1CRUdBMVVFQXhNS2JXRnlhV0ZrWWkxallUQlpNQk1HQnlxR1NNNDlBZ0VHQ0NxR1NNNDlBd0VICkEwSUFCQ2IxQ1IwUjV1akZ1QUR\
Gd1NsazQzUUpmdDFmTFVnOWNJNyttZ0w3bVd3MmVLUXowL04ybm9KMGpJaDYKN0NnQ2syUW1jNTdWM1podkFWQzJoU2NEbWg2aldUQlhNQTRHQTFVZER3RU\
Ivd1FFQXdJQ0JEQVBCZ05WSFJNQgpBZjhFQlRBREFRSC9NQjBHQTFVZERnUVdCQlNUa2tzSU9pL1pTbCtQRlJua2NQRlJ0QTRrMERBVkJnTlZIUkVFCkRqQ\
U1nZ3B0WVhKcFlXUmlMV05oTUFvR0NDcUdTTTQ5QkFNQ0EwY0FNRVFDSUI1Q2F6VW1WWUZQYTFkS2txUGkKbitKSEQvNVZTTGd4aHVPclgzUGcxQnlzQWlB\
RmcvTXlNWW9CZUNrUVRWdS9rUkIwK2N2Qy9RMDB4NExvVGpJaQpGdCtKMGc9PQotLS0tLUVORCBDRVJUSUZJQ0FURS0t\
LS0t"  # pragma: allowlist secret


@pytest.fixture(scope="class")
def model_namespace_2(
    request: FixtureRequest,
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    if request.param.get("modelmesh-enabled"):
        request.getfixturevalue(argname="enabled_modelmesh_in_dsc")

    if pytestconfig.option.post_upgrade:
        ns = Namespace(client=admin_client, name=request.param["name"])
        yield ns
        ns.clean_up()
    else:
        with create_ns(
            admin_client=admin_client,
            pytest_request=request,
            teardown=teardown_resources,
        ) as ns:
            yield ns


@pytest.fixture(scope="class")
def minio_data_connection_2(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace_2: Namespace,
    minio_service: Service,
) -> Generator[Secret, Any, Any]:
    with create_minio_data_connection_secret(
        minio_service=minio_service,
        model_namespace=model_namespace_2.name,
        aws_s3_bucket=request.param["bucket"],
        client=admin_client,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def trustyai_service_with_pvc_storage_2(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace_2: Namespace,
    cluster_monitoring_config: ConfigMap,
    user_workload_monitoring_config: ConfigMap,
    teardown_resources: bool,
) -> Generator[TrustyAIService, Any, Any]:
    trustyai_service_kwargs = {
        "client": admin_client,
        "namespace": model_namespace_2.name,
        "name": TRUSTYAI_SERVICE_NAME,
    }

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
def gaussian_credit_model_2(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace_2: Namespace,
    minio_pod: Pod,
    minio_service: Service,
    minio_data_connection_2: Secret,
    mlserver_runtime_2: ServingRuntime,
    trustyai_service_with_pvc_storage_2: TrustyAIService,
    teardown_resources: bool,
) -> Generator[InferenceService, Any, Any]:
    gaussian_credit_model_kwargs = {
        "client": admin_client,
        "namespace": model_namespace_2.name,
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
            runtime=mlserver_runtime_2.name,
            storage_key=minio_data_connection_2.name,
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
                runtime_name=mlserver_runtime_2.name,
            )
            yield isvc


@pytest.fixture(scope="class")
def mlserver_runtime_2(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    minio_data_connection: Secret,
    model_namespace_2: Namespace,
    teardown_resources: bool,
) -> Generator[ServingRuntime, Any, Any]:
    mlserver_runtime_kwargs = {
        "client": admin_client,
        "namespace": model_namespace_2.name,
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
def trustyai_service_with_invalid_db_cert(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    cluster_monitoring_config: ConfigMap,
    user_workload_monitoring_config: ConfigMap,
    mariadb: MariaDB,
    trustyai_invalid_db_ca_secret: None,
) -> Generator[TrustyAIService, None, None]:
    """Create a TrustyAIService deployment with an invalid database certificate set as secret.

    Yields: A TrustyAIService with invalid database certificate set.
    """
    yield from create_trustyai_service(
        client=admin_client,
        namespace=model_namespace.name,
        storage=TAI_DB_STORAGE_CONFIG,
        metrics=TAI_METRICS_CONFIG,
        wait_for_replicas=False,
    )


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
def isvc_getter_service_account_2(
    admin_client: DynamicClient, model_namespace_2: Namespace
) -> Generator[ServiceAccount, Any, Any]:
    yield from create_isvc_getter_service_account(client=admin_client, namespace=model_namespace_2, name=ISVC_GETTER)


@pytest.fixture(scope="class")
def isvc_getter_role_2(admin_client: DynamicClient, model_namespace_2: Namespace) -> Generator[Role, Any, Any]:
    yield from create_isvc_getter_role(client=admin_client, namespace=model_namespace_2, name=ISVC_GETTER)


@pytest.fixture(scope="class")
def isvc_getter_role_binding_2(
    admin_client: DynamicClient,
    model_namespace_2: Namespace,
    isvc_getter_role_2: Role,
    isvc_getter_service_account_2: ServiceAccount,
) -> Generator[RoleBinding, Any, Any]:
    yield from create_isvc_getter_role_binding(
        client=admin_client,
        namespace=model_namespace_2,
        role=isvc_getter_role_2,
        service_account=isvc_getter_service_account_2,
        name=ISVC_GETTER,
    )


@pytest.fixture(scope="class")
def isvc_getter_token_secret_2(
    admin_client: DynamicClient,
    model_namespace_2: Namespace,
    isvc_getter_service_account_2: ServiceAccount,
    isvc_getter_role_binding_2: RoleBinding,
) -> Generator[Secret, Any, Any]:
    yield from create_isvc_getter_token_secret(
        client=admin_client,
        name="sa-token",
        namespace=model_namespace_2,
        service_account=isvc_getter_service_account_2,
    )


@pytest.fixture(scope="class")
def isvc_getter_token_2(isvc_getter_service_account_2: ServiceAccount, isvc_getter_token_secret_2: Secret) -> str:
    return create_inference_token(model_service_account=isvc_getter_service_account_2)
