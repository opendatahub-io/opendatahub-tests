from typing import Generator, Any, List

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.inference_service import InferenceService
from ocp_resources.trustyai_service import TrustyAIService
from ocp_resources.service_account import ServiceAccount
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding

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
from utilities.constants import KServeDeploymentType
from utilities.inference_utils import create_isvc
from utilities.infra import create_inference_token
from utilities.minio import create_minio_data_connection_secret


@pytest.fixture(scope="class")
def model_namespaces(
    request: FixtureRequest,
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[List[Namespace], Any, None]:
    if any(param.get("modelmesh-enabled") for param in request.param):
        request.getfixturevalue("enabled_modelmesh_in_dsc")

    namespaces = []
    for param in request.param:
        ns = Namespace(
            client=admin_client,
            name=param["name"],
            teardown=teardown_resources,
        )
        ns.deploy()
        namespaces.append(ns)

    yield namespaces

    for ns in namespaces:
        ns.clean_up()


@pytest.fixture(scope="class")
def minio_data_connection_multi_ns(
    request,
    admin_client,
    model_namespaces: List[Namespace],
    minio_service,
) -> Generator[List[Secret], Any, None]:
    params = request.param
    secrets = []

    for ns, param in zip(model_namespaces, params):
        with create_minio_data_connection_secret(
            minio_service=minio_service,
            model_namespace=ns.name,
            aws_s3_bucket=param["bucket"],
            client=admin_client,
        ) as secret:
            secrets.append(secret)

    yield secrets


@pytest.fixture(scope="class")
def trustyai_service_with_pvc_storage_multi_ns(
    admin_client,
    model_namespaces: List[Namespace],
    cluster_monitoring_config,
    user_workload_monitoring_config,
    teardown_resources: bool,
) -> Generator[List[TrustyAIService], Any, None]:
    services = []

    for ns in model_namespaces:
        service = create_trustyai_service(
            client=admin_client,
            namespace=ns.name,
            name=TRUSTYAI_SERVICE_NAME,
            storage=TAI_PVC_STORAGE_CONFIG,
            metrics=TAI_METRICS_CONFIG,
            data=TAI_DATA_CONFIG,
            wait_for_replicas=True,
            teardown=teardown_resources,
        )
        services.append(service)

    yield services


@pytest.fixture(scope="class")
def mlserver_runtime_multi_ns(
    admin_client,
    minio_data_connection_multi_ns: List[Secret],
    model_namespaces: List[Namespace],
    teardown_resources: bool,
) -> Generator[List[ServingRuntime], Any, None]:
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
            teardown=teardown_resources,
        )
        runtime.deploy()
        runtimes.append(runtime)

    yield runtimes

    for runtime in runtimes:
        runtime.clean_up()


@pytest.fixture(scope="class")
def gaussian_credit_model_multi_ns(
    admin_client,
    model_namespaces: List[Namespace],
    minio_pod,
    minio_service,
    minio_data_connection_multi_ns: List[Secret],
    mlserver_runtime_multi_ns: List[ServingRuntime],
    trustyai_service_with_pvc_storage_multi_ns: List[TrustyAIService],
    teardown_resources: bool,
) -> Generator[List[InferenceService], Any, None]:
    inference_services = []

    for i, (ns, minio_secret, runtime) in enumerate(
        zip(model_namespaces, minio_data_connection_multi_ns, mlserver_runtime_multi_ns)
    ):
        with create_isvc(
            client=admin_client,
            name=f"gaussian-credit-isvc-{i}",
            namespace=ns.name,
            deployment_mode=KServeDeploymentType.SERVERLESS,
            model_format=XGBOOST,
            runtime=runtime.name,
            storage_key=minio_secret.name,
            storage_path=GAUSSIAN_CREDIT_MODEL_STORAGE_PATH,
            enable_auth=True,
            wait_for_predictor_pods=False,
            resources=GAUSSIAN_CREDIT_MODEL_RESOURCES,
            teardown=teardown_resources,
        ) as isvc:
            wait_for_isvc_deployment_registered_by_trustyai_service(
                client=admin_client,
                isvc=isvc,
                runtime_name=runtime.name,
            )
            inference_services.append(isvc)

    yield inference_services


@pytest.fixture(scope="class")
def isvc_getter_service_account_multi_ns(
    admin_client,
    model_namespaces: List[Namespace],
) -> Generator[List[ServiceAccount], Any, None]:
    sas = []
    for ns in model_namespaces:
        sa = create_isvc_getter_service_account(
            client=admin_client,
            namespace=ns,
            name=ISVC_GETTER,
        )
        sas.append(sa)
    yield sas


@pytest.fixture(scope="class")
def isvc_getter_role_multi_ns(
    admin_client,
    model_namespaces: List[Namespace],
) -> Generator[List[Role], Any, None]:
    roles = []
    for ns in model_namespaces:
        role = create_isvc_getter_role(
            client=admin_client,
            namespace=ns,
            name=ISVC_GETTER,
        )
        roles.append(role)
    yield roles


@pytest.fixture(scope="class")
def isvc_getter_role_binding_multi_ns(
    admin_client,
    model_namespaces: List[Namespace],
    isvc_getter_role_multi_ns: List[Role],
    isvc_getter_service_account_multi_ns: List[ServiceAccount],
) -> Generator[List[RoleBinding], Any, None]:
    role_bindings = []
    for ns, role, sa in zip(model_namespaces, isvc_getter_role_multi_ns, isvc_getter_service_account_multi_ns):
        rb = create_isvc_getter_role_binding(
            client=admin_client,
            namespace=ns,
            role=role,
            service_account=sa,
            name=ISVC_GETTER,
        )
        role_bindings.append(rb)
    yield role_bindings


@pytest.fixture(scope="class")
def isvc_getter_token_secret_multi_ns(
    admin_client,
    model_namespaces: List[Namespace],
    isvc_getter_service_account_multi_ns: List[ServiceAccount],
    isvc_getter_role_binding_multi_ns: List[RoleBinding],
) -> Generator[List[Secret], Any, None]:
    secrets = []
    for ns, sa in zip(model_namespaces, isvc_getter_service_account_multi_ns):
        secret = create_isvc_getter_token_secret(
            client=admin_client,
            name="sa-token",
            namespace=ns,
            service_account=sa,
        )
        secrets.append(secret)
    yield secrets


@pytest.fixture(scope="class")
def isvc_getter_token_multi_ns(
    isvc_getter_service_account_multi_ns: List[ServiceAccount],
    isvc_getter_token_secret_multi_ns: List[Secret],
) -> List[str]:
    return [create_inference_token(model_service_account=sa) for sa in isvc_getter_service_account_multi_ns]
