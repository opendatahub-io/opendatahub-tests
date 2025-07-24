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
    GAUSSIAN_CREDIT_MODEL,
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
from utilities.infra import create_inference_token, create_ns
from utilities.minio import create_minio_data_connection_secret


@pytest.fixture(scope="class")
def model_namespaces(request, admin_client) -> Generator[List[Namespace], Any, None]:
    with ExitStack() as stack:
        namespaces = [
            stack.enter_context(create_ns(admin_client=admin_client, name=param["name"])) for param in request.param
        ]
        yield namespaces


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
    admin_client,
    model_namespaces,
    minio_pod,
    minio_service,
    minio_data_connection_multi_ns,
    mlserver_runtime_multi_ns,
    trustyai_service_with_pvc_storage_multi_ns,
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
