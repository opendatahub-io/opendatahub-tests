import json
from collections.abc import Generator
from typing import Any

import pytest
import structlog
import yaml
from _pytest.nodes import Item
from _pytest.runner import CallInfo
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.gateway import Gateway
from ocp_resources.inference_service import InferenceService
from ocp_resources.job import Job
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_serving.model_runtime.vllm.utils import skip_if_not_deployment_mode
from tests.model_serving.model_server.upgrade.admission_check_upgrade_config import (
    AC_ADMISSION_CHECK_NAME,
    AC_BASELINE_CM,
    AC_CLUSTER_QUEUE,
    AC_CONTROLLER_NAME,
    AC_CPU_QUOTA,
    AC_JOB_CPU_LIMIT,
    AC_JOB_CPU_REQUEST,
    AC_JOB_MEMORY_LIMIT,
    AC_JOB_MEMORY_REQUEST,
    AC_JOB_NAME,
    AC_LOCAL_QUEUE,
    AC_MEMORY_QUOTA,
    AC_NAMESPACE,
    AC_RESOURCE_FLAVOR,
)
from tests.model_serving.model_server.upgrade.kserve_kueue_upgrade_config import (
    KSERVE_KUEUE_CLUSTER_QUEUE,
    KSERVE_KUEUE_CPU_QUOTA,
    KSERVE_KUEUE_ISVC_LABELS,
    KSERVE_KUEUE_ISVC_RESOURCES,
    KSERVE_KUEUE_LOCAL_QUEUE,
    KSERVE_KUEUE_MAX_REPLICAS,
    KSERVE_KUEUE_MEMORY_QUOTA,
    KSERVE_KUEUE_MIN_REPLICAS,
    KSERVE_KUEUE_RESOURCE_FLAVOR,
    KSERVE_KUEUE_UPGRADE_ISVC_NAME,
    KSERVE_KUEUE_UPGRADE_NAMESPACE,
    KSERVE_KUEUE_UPGRADE_RUNTIME_NAME,
    KSERVE_KUEUE_UPGRADE_S3_SECRET,
)
from tests.model_serving.model_server.upgrade.utils import (
    _create_kueue_upgrade_resources,
    _ensure_kueue_available_for_upgrade,
    _kserve_kueue_upgrade_runtime_template_kwargs,
    capture_and_save_isvc_kueue_baseline,
    capture_isvc_baseline,
    capture_llmisvc_baseline,
    load_auth_token_from_secret,
    load_baseline_from_configmap,
    save_auth_token_to_secret,
    save_baseline_to_configmap,
)
from utilities.constants import (
    KServeDeploymentType,
    ModelAndFormat,
    ModelCarImage,
    ModelFormat,
    ModelStoragePath,
    ModelVersion,
    Protocols,
    RuntimeTemplates,
    Timeout,
)
from utilities.inference_utils import create_isvc
from utilities.infra import (
    create_inference_token,
    create_isvc_view_role,
    create_ns,
    s3_endpoint_secret,
    update_configmap_data,
)
from utilities.kueue_utils import (
    ClusterQueue,
    LocalQueue,
    ResourceFlavor,
    Workload,
    activate_admission_check,
    create_admission_check,
    create_cluster_queue,
    create_local_queue,
    create_resource_flavor,
    get_workload_for_job,
)
from utilities.llmd_constants import ContainerImages, KServeGateway, LLMDGateway, ModelStorage
from utilities.llmd_utils import create_llmd_gateway, create_llmisvc
from utilities.logger import RedactedString
from utilities.resources.admission_check import AdmissionCheck
from utilities.serving_runtime import ServingRuntimeFromTemplate

LOGGER = structlog.get_logger(name=__name__)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: Item, call: CallInfo[None]) -> Generator[None, Any, Any]:
    """Track pre-upgrade test failures to prevent baseline capture on failure."""
    outcome = yield
    report = outcome.get_result()

    if call.when == "call" and report.failed and "pre_upgrade" in item.keywords:
        item.config._pre_upgrade_test_failed = True  # type: ignore[attr-defined]


UPGRADE_NAMESPACE = "upgrade-model-server"
AUTH_UPGRADE_NAMESPACE = "upgrade-auth-model-server"
MODEL_CAR_UPGRADE_NAMESPACE = "upgrade-model-car"
METRICS_UPGRADE_NAMESPACE = "upgrade-metrics"
PRIVATE_ENDPOINT_UPGRADE_NAMESPACE = "upgrade-pvt-ep"
LLMD_UPGRADE_NAMESPACE = "upgrade-llmd"
NEW_ISVC_UPGRADE_NAMESPACE = "upgrade-new-isvc"
S3_CONNECTION = "upgrade-connection"


@pytest.fixture(scope="session")
def upgrade_baseline_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
) -> dict[str, dict]:
    """Load pre-upgrade baseline values from the cluster ConfigMap.

    Only available during post-upgrade runs. Returns an empty dict during
    pre-upgrade so fixtures that depend on it can be unconditionally wired.
    """
    if not pytestconfig.option.post_upgrade:
        return {}

    return load_baseline_from_configmap(
        client=admin_client,
        namespace=UPGRADE_NAMESPACE,
    )


@pytest.fixture(scope="session")
def namespace_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    ns = Namespace(client=admin_client, name=UPGRADE_NAMESPACE)

    if pytestconfig.option.post_upgrade:
        yield ns
        ns.clean_up()

    else:
        with create_ns(
            admin_client=admin_client,
            name=UPGRADE_NAMESPACE,
            model_mesh_enabled=False,
            add_dashboard_label=True,
            teardown=teardown_resources,
        ) as ns:
            yield ns


@pytest.fixture(scope="session")
def s3_connection_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    namespace_fixture: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    ci_s3_bucket_name: str,
    ci_s3_bucket_region: str,
    ci_s3_bucket_endpoint: str,
    teardown_resources: bool,
) -> Generator[Secret, Any, Any]:
    secret_kwargs = {
        "client": admin_client,
        "name": S3_CONNECTION,
        "namespace": namespace_fixture.name,
    }

    secret = Secret(**secret_kwargs)

    if pytestconfig.option.post_upgrade:
        yield secret
        secret.clean_up()

    else:
        with s3_endpoint_secret(
            **secret_kwargs,
            aws_access_key=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_s3_region=ci_s3_bucket_region,
            aws_s3_bucket=ci_s3_bucket_name,
            aws_s3_endpoint=ci_s3_bucket_endpoint,
            teardown=teardown_resources,
        ) as secret:
            yield secret


@pytest.fixture(scope="session")
def serving_runtime_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    namespace_fixture: Namespace,
    teardown_resources: bool,
) -> Generator[ServingRuntime, Any, Any]:
    runtime_kwargs = {
        "client": admin_client,
        "name": "upgrade-runtime",
        "namespace": namespace_fixture.name,
    }

    model_runtime = ServingRuntime(**runtime_kwargs)

    if pytestconfig.option.post_upgrade:
        yield model_runtime
        model_runtime.clean_up()

    else:
        with ServingRuntimeFromTemplate(
            **runtime_kwargs,
            template_name=RuntimeTemplates.OVMS_KSERVE,
            multi_model=False,
            enable_http=True,
            teardown=teardown_resources,
            resources={
                ModelFormat.OVMS: {
                    "requests": {"cpu": "1", "memory": "4Gi"},
                    "limits": {"cpu": "2", "memory": "8Gi"},
                }
            },
        ) as model_runtime:
            yield model_runtime


@pytest.fixture(scope="session")
def inference_service_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    serving_runtime_fixture: ServingRuntime,
    s3_connection_fixture: Secret,
    teardown_resources: bool,
) -> Generator[InferenceService, Any, Any]:
    isvc_kwargs = {
        "client": admin_client,
        "name": "upgrade-isvc",
        "namespace": serving_runtime_fixture.namespace,
    }

    isvc = InferenceService(**isvc_kwargs)

    if pytestconfig.option.post_upgrade:
        yield isvc

        isvc.clean_up()

    else:
        with create_isvc(
            runtime=serving_runtime_fixture.name,
            model_format=ModelAndFormat.OPENVINO_IR,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            storage_key=s3_connection_fixture.name,
            storage_path=ModelStoragePath.OPENVINO_EXAMPLE_MODEL,
            model_version=ModelVersion.OPSET13,
            external_route=False,
            teardown=teardown_resources,
            **isvc_kwargs,
        ) as isvc:
            yield isvc


def _capture_and_save_baseline(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    isvc: InferenceService,
) -> None:
    """Capture ISVC baseline values and persist to the shared ConfigMap.

    No-op during post-upgrade runs.
    """
    if pytestconfig.option.post_upgrade:
        return

    baselines = {
        isvc.name: capture_isvc_baseline(
            client=admin_client,
            isvc=isvc,
        ),
    }
    save_baseline_to_configmap(
        client=admin_client,
        namespace=UPGRADE_NAMESPACE,
        baselines=baselines,
    )


@pytest.fixture(scope="session")
def capture_upgrade_baseline(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    inference_service_fixture: InferenceService,
) -> None:
    """Capture baseline values for the basic raw-deployment ISVC."""
    _capture_and_save_baseline(pytestconfig=pytestconfig, admin_client=admin_client, isvc=inference_service_fixture)


@pytest.fixture(scope="session")
def capture_auth_upgrade_baseline(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    auth_inference_service_fixture: InferenceService,
    auth_inference_token_fixture: str,
) -> None:
    """Capture baseline values and auth token for the auth ISVC."""
    if pytestconfig.option.post_upgrade:
        return

    _capture_and_save_baseline(
        pytestconfig=pytestconfig, admin_client=admin_client, isvc=auth_inference_service_fixture
    )
    save_auth_token_to_secret(
        client=admin_client,
        namespace=UPGRADE_NAMESPACE,
        token=str(auth_inference_token_fixture),
    )


@pytest.fixture(scope="session")
def capture_model_car_upgrade_baseline(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_car_inference_service_fixture: InferenceService,
) -> None:
    """Capture baseline values for the model-car ISVC."""
    _capture_and_save_baseline(
        pytestconfig=pytestconfig, admin_client=admin_client, isvc=model_car_inference_service_fixture
    )


@pytest.fixture(scope="session")
def capture_metrics_upgrade_baseline(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    metrics_inference_service_fixture: InferenceService,
) -> None:
    """Capture baseline values for the metrics ISVC."""
    _capture_and_save_baseline(
        pytestconfig=pytestconfig, admin_client=admin_client, isvc=metrics_inference_service_fixture
    )


@pytest.fixture(scope="session")
def capture_private_endpoint_upgrade_baseline(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    private_endpoint_inference_service_fixture: InferenceService,
) -> None:
    """Capture baseline values for the private-endpoint ISVC."""
    _capture_and_save_baseline(
        pytestconfig=pytestconfig, admin_client=admin_client, isvc=private_endpoint_inference_service_fixture
    )


# Authentication Upgrade Fixtures
@pytest.fixture(scope="session")
def auth_namespace_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    """Namespace for authentication upgrade tests."""
    ns = Namespace(client=admin_client, name=AUTH_UPGRADE_NAMESPACE)

    if pytestconfig.option.post_upgrade:
        yield ns
        ns.clean_up()
    else:
        with create_ns(
            admin_client=admin_client,
            name=AUTH_UPGRADE_NAMESPACE,
            model_mesh_enabled=False,
            add_dashboard_label=True,
            teardown=teardown_resources,
        ) as ns:
            yield ns


@pytest.fixture(scope="session")
def auth_s3_connection_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    auth_namespace_fixture: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    ci_s3_bucket_name: str,
    ci_s3_bucket_region: str,
    ci_s3_bucket_endpoint: str,
    teardown_resources: bool,
) -> Generator[Secret, Any, Any]:
    """S3 connection secret for authentication upgrade tests."""
    secret_kwargs = {
        "client": admin_client,
        "name": "auth-upgrade-connection",
        "namespace": auth_namespace_fixture.name,
    }

    secret = Secret(**secret_kwargs)

    if pytestconfig.option.post_upgrade:
        yield secret
        secret.clean_up()
    else:
        with s3_endpoint_secret(
            **secret_kwargs,
            aws_access_key=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_s3_region=ci_s3_bucket_region,
            aws_s3_bucket=ci_s3_bucket_name,
            aws_s3_endpoint=ci_s3_bucket_endpoint,
            teardown=teardown_resources,
        ) as secret:
            yield secret


@pytest.fixture(scope="session")
def auth_service_account_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    auth_namespace_fixture: Namespace,
    auth_s3_connection_fixture: Secret,
    teardown_resources: bool,
) -> Generator[ServiceAccount, Any, Any]:
    """ServiceAccount for token-based authentication during upgrade tests."""
    sa_kwargs = {
        "client": admin_client,
        "namespace": auth_namespace_fixture.name,
        "name": "auth-upgrade-sa",
    }

    sa = ServiceAccount(**sa_kwargs)

    if pytestconfig.option.post_upgrade:
        yield sa
        sa.clean_up()
    else:
        with ServiceAccount(
            **sa_kwargs,
            secrets=[{"name": auth_s3_connection_fixture.name}],
            teardown=teardown_resources,
        ) as sa:
            yield sa


@pytest.fixture(scope="session")
def auth_serving_runtime_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    auth_namespace_fixture: Namespace,
    teardown_resources: bool,
) -> Generator[ServingRuntime, Any, Any]:
    """ServingRuntime for authentication upgrade tests."""
    runtime_kwargs = {
        "client": admin_client,
        "name": "auth-upgrade-runtime",
        "namespace": auth_namespace_fixture.name,
    }

    model_runtime = ServingRuntime(**runtime_kwargs)

    if pytestconfig.option.post_upgrade:
        yield model_runtime
        model_runtime.clean_up()
    else:
        with ServingRuntimeFromTemplate(
            **runtime_kwargs,
            template_name=RuntimeTemplates.OVMS_KSERVE,
            multi_model=False,
            enable_http=True,
            enable_grpc=False,
            teardown=teardown_resources,
            resources={
                ModelFormat.OVMS: {
                    "requests": {"cpu": "1", "memory": "4Gi"},
                    "limits": {"cpu": "2", "memory": "8Gi"},
                }
            },
        ) as model_runtime:
            yield model_runtime


@pytest.fixture(scope="session")
def auth_inference_service_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    auth_serving_runtime_fixture: ServingRuntime,
    auth_s3_connection_fixture: Secret,
    auth_service_account_fixture: ServiceAccount,
    teardown_resources: bool,
) -> Generator[InferenceService, Any, Any]:
    """InferenceService with authentication enabled for upgrade tests."""
    isvc_kwargs = {
        "client": admin_client,
        "name": "auth-upgrade-isvc",
        "namespace": auth_serving_runtime_fixture.namespace,
    }

    isvc = InferenceService(**isvc_kwargs)

    if pytestconfig.option.post_upgrade:
        yield isvc
        isvc.clean_up()
    else:
        with create_isvc(
            runtime=auth_serving_runtime_fixture.name,
            model_format=ModelAndFormat.OPENVINO_IR,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            storage_key=auth_s3_connection_fixture.name,
            storage_path=ModelStoragePath.OPENVINO_EXAMPLE_MODEL,
            model_version=ModelVersion.OPSET13,
            external_route=True,
            enable_auth=True,
            model_service_account=auth_service_account_fixture.name,
            teardown=teardown_resources,
            **isvc_kwargs,
        ) as isvc:
            yield isvc


@pytest.fixture(scope="session")
def auth_view_role_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    auth_inference_service_fixture: InferenceService,
    teardown_resources: bool,
) -> Generator[Role, Any, Any]:
    """Role for viewing InferenceService during authentication upgrade tests."""
    role_kwargs = {
        "client": admin_client,
        "name": f"{auth_inference_service_fixture.name}-view",
        "namespace": auth_inference_service_fixture.namespace,
    }

    role = Role(**role_kwargs)

    if pytestconfig.option.post_upgrade:
        yield role
        role.clean_up()
    else:
        with create_isvc_view_role(
            client=admin_client,
            isvc=auth_inference_service_fixture,
            name=role_kwargs["name"],
            resource_names=[auth_inference_service_fixture.name],
            teardown=teardown_resources,
        ) as role:
            yield role


@pytest.fixture(scope="session")
def auth_role_binding_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    auth_view_role_fixture: Role,
    auth_service_account_fixture: ServiceAccount,
    teardown_resources: bool,
) -> Generator[RoleBinding, Any, Any]:
    """RoleBinding for authentication upgrade tests."""
    rb_kwargs = {
        "client": admin_client,
        "namespace": auth_service_account_fixture.namespace,
        "name": f"{Protocols.HTTP}-{auth_service_account_fixture.name}-view",
    }

    rb = RoleBinding(**rb_kwargs)

    if pytestconfig.option.post_upgrade:
        yield rb
        rb.clean_up()
    else:
        with RoleBinding(
            **rb_kwargs,
            role_ref_name=auth_view_role_fixture.name,
            role_ref_kind=auth_view_role_fixture.kind,
            subjects_kind=auth_service_account_fixture.kind,
            subjects_name=auth_service_account_fixture.name,
            teardown=teardown_resources,
        ) as rb:
            yield rb


@pytest.fixture(scope="session")
def auth_inference_token_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    auth_service_account_fixture: ServiceAccount,
    auth_role_binding_fixture: RoleBinding,
) -> str:
    """Authentication token for upgrade tests.

    Pre-upgrade: creates a fresh token and returns it (also persisted to
    the baseline ConfigMap by capture_auth_upgrade_baseline).
    Post-upgrade: loads the pre-upgrade token from the ConfigMap so
    inference tests prove the old token still works after the upgrade.
    """
    if pytestconfig.option.post_upgrade:
        return RedactedString(
            value=load_auth_token_from_secret(
                client=admin_client,
                namespace=UPGRADE_NAMESPACE,
            )
        )

    return RedactedString(value=create_inference_token(model_service_account=auth_service_account_fixture))


@pytest.fixture(scope="session")
def auth_fresh_token_fixture(
    pytestconfig: pytest.Config,
    auth_service_account_fixture: ServiceAccount,
    auth_role_binding_fixture: RoleBinding,
) -> str | None:
    """Fresh authentication token created post-upgrade.

    Only available during post-upgrade runs. Used to verify that new
    token creation works on the upgraded control plane.
    """
    if not pytestconfig.option.post_upgrade:
        return None

    return RedactedString(value=create_inference_token(model_service_account=auth_service_account_fixture))


# Model Car Upgrade Fixtures
@pytest.fixture(scope="session")
def model_car_namespace_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    """Namespace for Model Car upgrade tests."""
    ns = Namespace(client=admin_client, name=MODEL_CAR_UPGRADE_NAMESPACE)

    if pytestconfig.option.post_upgrade:
        yield ns
        ns.clean_up()
    else:
        with create_ns(
            admin_client=admin_client,
            name=MODEL_CAR_UPGRADE_NAMESPACE,
            model_mesh_enabled=False,
            add_dashboard_label=True,
            teardown=teardown_resources,
        ) as ns:
            yield ns


@pytest.fixture(scope="session")
def model_car_serving_runtime_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_car_namespace_fixture: Namespace,
    teardown_resources: bool,
) -> Generator[ServingRuntime, Any, Any]:
    """ServingRuntime for Model Car upgrade tests."""
    runtime_kwargs = {
        "client": admin_client,
        "name": "model-car-upgrade-runtime",
        "namespace": model_car_namespace_fixture.name,
    }

    model_runtime = ServingRuntime(**runtime_kwargs)

    if pytestconfig.option.post_upgrade:
        yield model_runtime
        model_runtime.clean_up()
    else:
        with ServingRuntimeFromTemplate(
            **runtime_kwargs,
            template_name=RuntimeTemplates.OVMS_KSERVE,
            multi_model=False,
            enable_http=True,
            teardown=teardown_resources,
            resources={
                ModelFormat.OVMS: {
                    "requests": {"cpu": "1", "memory": "4Gi"},
                    "limits": {"cpu": "2", "memory": "8Gi"},
                }
            },
        ) as model_runtime:
            yield model_runtime


@pytest.fixture(scope="session")
def model_car_inference_service_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_car_serving_runtime_fixture: ServingRuntime,
    teardown_resources: bool,
) -> Generator[InferenceService, Any, Any]:
    """InferenceService using OCI Model Car image for upgrade tests."""
    isvc_kwargs = {
        "client": admin_client,
        "name": "model-car-upgrade-isvc",
        "namespace": model_car_serving_runtime_fixture.namespace,
    }

    isvc = InferenceService(**isvc_kwargs)

    if pytestconfig.option.post_upgrade:
        yield isvc
        isvc.clean_up()
    else:
        with create_isvc(
            runtime=model_car_serving_runtime_fixture.name,
            model_format=model_car_serving_runtime_fixture.instance.spec.supportedModelFormats[0].name,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            storage_uri=ModelCarImage.MNIST_8_1,
            external_route=True,
            teardown=teardown_resources,
            wait_for_predictor_pods=False,
            **isvc_kwargs,
        ) as isvc:
            yield isvc


# Metrics Upgrade Fixtures
@pytest.fixture(scope="session")
def upgrade_user_workload_monitoring_config_map(
    admin_client: DynamicClient,
    cluster_monitoring_config: ConfigMap,
) -> Generator[ConfigMap]:
    """
    Session-scoped user workload monitoring ConfigMap for upgrade tests.

    Unlike the class-scoped fixture in conftest.py, this fixture does NOT
    delete PVCs on teardown, preserving Prometheus historical data across
    pre-upgrade and post-upgrade test runs.
    """
    uwm_namespace = "openshift-user-workload-monitoring"

    data = {
        "config.yaml": yaml.dump({
            "prometheus": {
                "logLevel": "debug",
                "retention": "15d",
                "volumeClaimTemplate": {"spec": {"resources": {"requests": {"storage": "40Gi"}}}},
            }
        })
    }

    with update_configmap_data(
        client=admin_client,
        name="user-workload-monitoring-config",
        namespace=uwm_namespace,
        data=data,
    ) as cm:
        yield cm

    # NOTE: Intentionally NOT deleting PVCs to preserve Prometheus data
    # for post-upgrade metrics retention verification


@pytest.fixture(scope="session")
def metrics_namespace_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    """Namespace for metrics persistence upgrade tests."""
    ns = Namespace(client=admin_client, name=METRICS_UPGRADE_NAMESPACE)

    if pytestconfig.option.post_upgrade:
        yield ns
        ns.clean_up()
    else:
        with create_ns(
            admin_client=admin_client,
            name=METRICS_UPGRADE_NAMESPACE,
            model_mesh_enabled=False,
            add_dashboard_label=True,
            teardown=teardown_resources,
        ) as ns:
            yield ns


@pytest.fixture(scope="session")
def metrics_serving_runtime_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    metrics_namespace_fixture: Namespace,
    teardown_resources: bool,
) -> Generator[ServingRuntime, Any, Any]:
    """ServingRuntime for metrics persistence upgrade tests."""
    runtime_kwargs = {
        "client": admin_client,
        "name": "metrics-upgrade-runtime",
        "namespace": metrics_namespace_fixture.name,
    }

    model_runtime = ServingRuntime(**runtime_kwargs)

    if pytestconfig.option.post_upgrade:
        yield model_runtime
        model_runtime.clean_up()
    else:
        with ServingRuntimeFromTemplate(
            **runtime_kwargs,
            template_name=RuntimeTemplates.OVMS_KSERVE,
            multi_model=False,
            enable_http=True,
            teardown=teardown_resources,
            resources={
                ModelFormat.OVMS: {
                    "requests": {"cpu": "1", "memory": "4Gi"},
                    "limits": {"cpu": "2", "memory": "8Gi"},
                }
            },
        ) as model_runtime:
            yield model_runtime


@pytest.fixture(scope="session")
def metrics_inference_service_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    metrics_serving_runtime_fixture: ServingRuntime,
    teardown_resources: bool,
) -> Generator[InferenceService, Any, Any]:
    """InferenceService for metrics persistence upgrade tests."""
    isvc_kwargs = {
        "client": admin_client,
        "name": "metrics-upgrade-isvc",
        "namespace": metrics_serving_runtime_fixture.namespace,
    }

    isvc = InferenceService(**isvc_kwargs)

    if pytestconfig.option.post_upgrade:
        yield isvc
        isvc.clean_up()
    else:
        with create_isvc(
            runtime=metrics_serving_runtime_fixture.name,
            model_format=metrics_serving_runtime_fixture.instance.spec.supportedModelFormats[0].name,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            storage_uri=ModelCarImage.MNIST_8_1,
            external_route=True,
            teardown=teardown_resources,
            wait_for_predictor_pods=False,
            **isvc_kwargs,
        ) as isvc:
            yield isvc


# Private Endpoint Upgrade Fixtures
@pytest.fixture(scope="session")
def private_endpoint_namespace_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    """Namespace for private endpoint upgrade tests."""
    ns = Namespace(client=admin_client, name=PRIVATE_ENDPOINT_UPGRADE_NAMESPACE)

    if pytestconfig.option.post_upgrade:
        yield ns
        ns.clean_up()
    else:
        with create_ns(
            admin_client=admin_client,
            name=PRIVATE_ENDPOINT_UPGRADE_NAMESPACE,
            model_mesh_enabled=False,
            add_dashboard_label=True,
            teardown=teardown_resources,
        ) as ns:
            yield ns


@pytest.fixture(scope="session")
def private_endpoint_s3_connection_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    private_endpoint_namespace_fixture: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    ci_s3_bucket_name: str,
    ci_s3_bucket_region: str,
    ci_s3_bucket_endpoint: str,
    teardown_resources: bool,
) -> Generator[Secret, Any, Any]:
    """S3 connection secret for private endpoint upgrade tests."""
    secret_kwargs = {
        "client": admin_client,
        "name": "pvt-ep-upgrade-connection",
        "namespace": private_endpoint_namespace_fixture.name,
    }

    secret = Secret(**secret_kwargs)

    if pytestconfig.option.post_upgrade:
        yield secret
        secret.clean_up()
    else:
        with s3_endpoint_secret(
            **secret_kwargs,
            aws_access_key=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_s3_region=ci_s3_bucket_region,
            aws_s3_bucket=ci_s3_bucket_name,
            aws_s3_endpoint=ci_s3_bucket_endpoint,
            teardown=teardown_resources,
        ) as secret:
            yield secret


@pytest.fixture(scope="session")
def private_endpoint_serving_runtime_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    private_endpoint_namespace_fixture: Namespace,
    teardown_resources: bool,
) -> Generator[ServingRuntime, Any, Any]:
    """ServingRuntime for private endpoint upgrade tests."""
    runtime_kwargs = {
        "client": admin_client,
        "name": "pvt-ep-upgrade-runtime",
        "namespace": private_endpoint_namespace_fixture.name,
    }

    model_runtime = ServingRuntime(**runtime_kwargs)

    if pytestconfig.option.post_upgrade:
        yield model_runtime
        model_runtime.clean_up()
    else:
        with ServingRuntimeFromTemplate(
            **runtime_kwargs,
            template_name=RuntimeTemplates.OVMS_KSERVE,
            multi_model=False,
            enable_http=True,
            teardown=teardown_resources,
            resources={
                ModelFormat.OVMS: {
                    "requests": {"cpu": "1", "memory": "4Gi"},
                    "limits": {"cpu": "2", "memory": "8Gi"},
                }
            },
        ) as model_runtime:
            yield model_runtime


@pytest.fixture(scope="session")
def private_endpoint_inference_service_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    private_endpoint_serving_runtime_fixture: ServingRuntime,
    private_endpoint_s3_connection_fixture: Secret,
    teardown_resources: bool,
) -> Generator[InferenceService, Any, Any]:
    """InferenceService with private endpoint (no external route) for upgrade tests."""
    isvc_kwargs = {
        "client": admin_client,
        "name": "pvt-ep-upgrade-isvc",
        "namespace": private_endpoint_serving_runtime_fixture.namespace,
    }

    isvc = InferenceService(**isvc_kwargs)

    if pytestconfig.option.post_upgrade:
        yield isvc
        isvc.clean_up()
    else:
        with create_isvc(
            runtime=private_endpoint_serving_runtime_fixture.name,
            model_format=ModelAndFormat.OPENVINO_IR,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            storage_key=private_endpoint_s3_connection_fixture.name,
            storage_path=ModelStoragePath.OPENVINO_EXAMPLE_MODEL,
            model_version=ModelVersion.OPSET13,
            external_route=False,
            teardown=teardown_resources,
            **isvc_kwargs,
        ) as isvc:
            yield isvc


# LLMD Upgrade Fixtures
@pytest.fixture(scope="session")
def llmd_namespace_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    """Namespace for LLMD upgrade tests."""
    ns = Namespace(client=admin_client, name=LLMD_UPGRADE_NAMESPACE)

    if pytestconfig.option.post_upgrade:
        yield ns
        ns.clean_up()
    else:
        with create_ns(
            admin_client=admin_client,
            name=LLMD_UPGRADE_NAMESPACE,
            model_mesh_enabled=False,
            add_dashboard_label=True,
            teardown=teardown_resources,
        ) as ns:
            yield ns


@pytest.fixture(scope="session")
def llmd_gateway_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Gateway, Any, Any]:
    """Shared LLMD Gateway for upgrade tests."""
    gateway = Gateway(
        client=admin_client,
        name=LLMDGateway.DEFAULT_NAME,
        namespace=LLMDGateway.DEFAULT_NAMESPACE,
        api_group=KServeGateway.API_GROUP,
    )

    if pytestconfig.option.post_upgrade:
        yield gateway
        gateway.clean_up()
    else:
        with create_llmd_gateway(
            client=admin_client,
            timeout=Timeout.TIMEOUT_1MIN,
            teardown=teardown_resources,
        ) as gateway:
            yield gateway


@pytest.fixture(scope="session")
def llmd_inference_service_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    llmd_namespace_fixture: Namespace,
    llmd_gateway_fixture: Gateway,
    teardown_resources: bool,
) -> Generator[LLMInferenceService, Any, Any]:
    """LLMInferenceService using TinyLlama OCI for upgrade tests."""
    from tests.model_serving.model_server.llmd.constants import LLMD_LIVENESS_PROBE

    llmisvc_name = "llmisvc-tinyllama-oci-cpu"

    if pytestconfig.option.post_upgrade:
        llmisvc = LLMInferenceService(
            client=admin_client,
            name=llmisvc_name,
            namespace=llmd_namespace_fixture.name,
        )
        yield llmisvc
        llmisvc.clean_up()
    else:
        with create_llmisvc(
            client=admin_client,
            name=llmisvc_name,
            namespace=llmd_namespace_fixture.name,
            storage_uri=ModelStorage.TINYLLAMA_OCI,
            container_image=ContainerImages.VLLM_CPU,
            container_resources={
                "limits": {"cpu": "1", "memory": "10Gi"},
                "requests": {"cpu": "100m", "memory": "8Gi"},
            },
            container_env=[
                {"name": "VLLM_LOGGING_LEVEL", "value": "DEBUG"},
                {
                    "name": "VLLM_ADDITIONAL_ARGS",
                    "value": (
                        "--max-num-seqs 20 --max-model-len 128 --enforce-eager --ssl-ciphers ECDHE+AESGCM:DHE+AESGCM"
                    ),
                },
                {"name": "VLLM_CPU_KVCACHE_SPACE", "value": "4"},
            ],
            liveness_probe=LLMD_LIVENESS_PROBE,
            teardown=teardown_resources,
            timeout=Timeout.TIMEOUT_15MIN,
        ) as llmisvc:
            yield llmisvc


@pytest.fixture(scope="session")
def capture_llmd_upgrade_baseline(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    llmd_inference_service_fixture: LLMInferenceService,
) -> None:
    """Capture LLMISVC baseline values and persist to a ConfigMap.

    No-op during post-upgrade runs.
    """
    if pytestconfig.option.post_upgrade:
        return

    baselines = {
        llmd_inference_service_fixture.name: capture_llmisvc_baseline(
            client=admin_client,
            llmisvc=llmd_inference_service_fixture,
        ),
    }
    save_baseline_to_configmap(
        client=admin_client,
        namespace=LLMD_UPGRADE_NAMESPACE,
        baselines=baselines,
    )


@pytest.fixture(scope="session")
def llmd_upgrade_baseline_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
) -> dict[str, dict]:
    """Load pre-upgrade LLMISVC baseline values from the cluster ConfigMap.

    Only available during post-upgrade runs. Returns an empty dict during
    pre-upgrade so fixtures that depend on it can be unconditionally wired.
    """
    if not pytestconfig.option.post_upgrade:
        return {}

    return load_baseline_from_configmap(
        client=admin_client,
        namespace=LLMD_UPGRADE_NAMESPACE,
    )


# Post-Upgrade New ISVC Creation Fixtures
@pytest.fixture(scope="session")
def new_isvc_namespace_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
) -> Generator[Namespace, Any, Any]:
    """Namespace for creating a fresh ISVC post-upgrade."""
    if not pytestconfig.option.post_upgrade:
        yield None
        return

    with create_ns(
        admin_client=admin_client,
        name=NEW_ISVC_UPGRADE_NAMESPACE,
        model_mesh_enabled=False,
        add_dashboard_label=True,
        teardown=True,
    ) as ns:
        yield ns


@pytest.fixture(scope="session")
def new_isvc_serving_runtime_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    new_isvc_namespace_fixture: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    """ServingRuntime for fresh ISVC creation on upgraded control plane."""
    if not pytestconfig.option.post_upgrade or new_isvc_namespace_fixture is None:
        yield None
        return

    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="new-isvc-upgrade-runtime",
        namespace=new_isvc_namespace_fixture.name,
        template_name=RuntimeTemplates.OVMS_KSERVE,
        multi_model=False,
        enable_http=True,
        teardown=True,
        resources={
            ModelFormat.OVMS: {
                "requests": {"cpu": "1", "memory": "4Gi"},
                "limits": {"cpu": "2", "memory": "8Gi"},
            }
        },
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="session")
def new_isvc_inference_service_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    new_isvc_serving_runtime_fixture: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    """Fresh InferenceService created on the upgraded control plane using Model Car (no S3)."""
    if not pytestconfig.option.post_upgrade or new_isvc_serving_runtime_fixture is None:
        yield None
        return

    with create_isvc(
        client=admin_client,
        name="new-isvc-post-upgrade",
        namespace=new_isvc_serving_runtime_fixture.namespace,
        runtime=new_isvc_serving_runtime_fixture.name,
        model_format=new_isvc_serving_runtime_fixture.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        storage_uri=ModelCarImage.MNIST_8_1,
        external_route=True,
        teardown=True,
        wait_for_predictor_pods=False,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="session")
def kserve_kueue_upgrade_s3_secret(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    kserve_kueue_upgrade_namespace: Namespace,
    valid_aws_config: tuple[str, str],
    teardown_resources: bool,
) -> Generator[Secret, Any, Any]:
    """S3 connection secret for the KServe Kueue upgrade ISVC."""
    _ = valid_aws_config
    secret_kwargs = {
        "client": admin_client,
        "name": KSERVE_KUEUE_UPGRADE_S3_SECRET,
        "namespace": kserve_kueue_upgrade_namespace.name,
    }
    secret = Secret(**secret_kwargs)

    if pytestconfig.option.post_upgrade:
        if not secret.exists:
            pytest.fail(
                f"[POST-UPGRADE] S3 secret '{secret.name}' not found in namespace "
                f"'{secret.namespace}'. Ensure pre-upgrade tests completed successfully."
            )
        yield secret
        if teardown_resources:
            secret.clean_up()
    elif secret.exists:
        pytest.fail(
            f"Unexpected existing S3 secret '{secret.name}' in namespace '{secret.namespace}'. "
            "Clear stale upgrade resources before pre-upgrade."
        )
    else:
        with s3_endpoint_secret(
            **secret_kwargs,
            aws_access_key=pytestconfig.option.aws_access_key_id,
            aws_secret_access_key=pytestconfig.option.aws_secret_access_key,
            aws_s3_region=pytestconfig.option.ci_s3_bucket_region,
            aws_s3_bucket=pytestconfig.option.ci_s3_bucket_name,
            aws_s3_endpoint=pytestconfig.option.ci_s3_bucket_endpoint,
            teardown=teardown_resources,
        ) as configured_secret:
            yield configured_secret


@pytest.fixture(scope="session")
def kserve_kueue_upgrade_model_storage() -> dict[str, str]:
    """Return model storage path and version for external S3."""
    return {
        "storage_path": ModelStoragePath.OPENVINO_EXAMPLE_MODEL,
        "model_version": ModelVersion.OPSET13,
    }


@pytest.fixture(scope="session")
def kserve_kueue_upgrade_serving_runtime(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    kserve_kueue_upgrade_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[ServingRuntime, Any, Any]:
    """OVMS ServingRuntime for KServe Kueue upgrade tests."""
    runtime_kwargs = {
        "client": admin_client,
        "name": KSERVE_KUEUE_UPGRADE_RUNTIME_NAME,
        "namespace": kserve_kueue_upgrade_namespace.name,
    }
    model_runtime = ServingRuntime(**runtime_kwargs)

    if pytestconfig.option.post_upgrade:
        if not model_runtime.exists:
            pytest.fail(
                f"[POST-UPGRADE] ServingRuntime '{model_runtime.name}' not found in namespace "
                f"'{model_runtime.namespace}'. Ensure pre-upgrade KServe+Kueue tests completed successfully."
            )
        yield model_runtime
        if teardown_resources:
            model_runtime.clean_up()
    elif model_runtime.exists:
        pytest.fail(
            f"Unexpected existing ServingRuntime '{model_runtime.name}' in namespace "
            f"'{model_runtime.namespace}'. Clear stale upgrade resources before pre-upgrade."
        )
    else:
        with ServingRuntimeFromTemplate(
            **_kserve_kueue_upgrade_runtime_template_kwargs(
                runtime_kwargs=runtime_kwargs,
                teardown_resources=teardown_resources,
            ),
        ) as model_runtime:
            yield model_runtime


@pytest.fixture(scope="session")
def kserve_kueue_upgrade_namespace(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    """Namespace for KServe raw ISVC + Kueue upgrade tests."""
    ns = Namespace(client=admin_client, name=KSERVE_KUEUE_UPGRADE_NAMESPACE)

    if pytestconfig.option.post_upgrade:
        if not ns.exists:
            pytest.fail(
                f"[POST-UPGRADE] Namespace '{ns.name}' not found. "
                "Ensure pre-upgrade KServe+Kueue tests completed successfully."
            )
        yield ns
        if teardown_resources:
            ns.clean_up()
    else:
        if ns.exists:
            pytest.fail(f"Unexpected existing namespace '{ns.name}'. Clear stale upgrade resources before pre-upgrade.")
        else:
            with create_ns(
                admin_client=admin_client,
                name=KSERVE_KUEUE_UPGRADE_NAMESPACE,
                model_mesh_enabled=False,
                add_dashboard_label=True,
                add_kueue_label=True,
                teardown=teardown_resources,
            ) as ns:
                yield ns


@pytest.fixture(scope="session")
def ensure_kueue_for_kserve_upgrade(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    dsc_resource: DataScienceCluster,
    kserve_kueue_upgrade_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[None, Any, Any]:
    """Ensure Kueue is available for KServe raw ISVC upgrade tests."""
    yield from _ensure_kueue_available_for_upgrade(
        pytestconfig=pytestconfig,
        admin_client=admin_client,
        dsc_resource=dsc_resource,
        namespace=kserve_kueue_upgrade_namespace.name,
        teardown_resources=teardown_resources,
    )


@pytest.fixture(scope="session")
def kserve_upgrade_kueue_resources(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    ensure_kueue_for_kserve_upgrade: None,
    kserve_kueue_upgrade_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[LocalQueue, Any, Any]:
    """Kueue ResourceFlavor, ClusterQueue, and LocalQueue for KServe upgrade tests."""
    yield from _create_kueue_upgrade_resources(
        pytestconfig=pytestconfig,
        admin_client=admin_client,
        namespace=kserve_kueue_upgrade_namespace.name,
        local_queue_name=KSERVE_KUEUE_LOCAL_QUEUE,
        cluster_queue_name=KSERVE_KUEUE_CLUSTER_QUEUE,
        resource_flavor_name=KSERVE_KUEUE_RESOURCE_FLAVOR,
        cpu_quota=KSERVE_KUEUE_CPU_QUOTA,
        memory_quota=KSERVE_KUEUE_MEMORY_QUOTA,
        teardown_resources=teardown_resources,
    )


@pytest.fixture(scope="session")
def kserve_kueue_upgrade_inference_service(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    kserve_kueue_upgrade_namespace: Namespace,
    kserve_kueue_upgrade_serving_runtime: ServingRuntime,
    kserve_kueue_upgrade_s3_secret: Secret,
    kserve_kueue_upgrade_model_storage: dict[str, str],
    kserve_upgrade_kueue_resources: LocalQueue,
    teardown_resources: bool,
) -> Generator[InferenceService, Any, Any]:
    """Raw-deployment InferenceService with Kueue queue label for upgrade tests."""
    isvc_kwargs = {
        "client": admin_client,
        "name": KSERVE_KUEUE_UPGRADE_ISVC_NAME,
        "namespace": kserve_kueue_upgrade_namespace.name,
    }
    isvc = InferenceService(**isvc_kwargs)

    if pytestconfig.option.post_upgrade:
        if not isvc.exists:
            pytest.fail(
                f"[POST-UPGRADE] InferenceService '{isvc.name}' not found in namespace "
                f"'{isvc.namespace}'. Ensure pre-upgrade KServe+Kueue tests completed successfully."
            )
        yield isvc
        if teardown_resources:
            isvc.clean_up()
    elif isvc.exists:
        pytest.fail(
            f"Unexpected existing InferenceService '{isvc.name}' in namespace '{isvc.namespace}'. "
            "Clear stale upgrade resources before pre-upgrade."
        )
    else:
        with create_isvc(
            runtime=kserve_kueue_upgrade_serving_runtime.name,
            model_format=ModelAndFormat.OPENVINO_IR,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            storage_key=kserve_kueue_upgrade_s3_secret.name,
            storage_path=kserve_kueue_upgrade_model_storage["storage_path"],
            model_version=kserve_kueue_upgrade_model_storage["model_version"],
            external_route=True,
            min_replicas=KSERVE_KUEUE_MIN_REPLICAS,
            max_replicas=KSERVE_KUEUE_MAX_REPLICAS,
            resources=KSERVE_KUEUE_ISVC_RESOURCES,
            labels=KSERVE_KUEUE_ISVC_LABELS,
            teardown=teardown_resources,
            **isvc_kwargs,
        ) as isvc:
            yield isvc
            if not getattr(pytestconfig, "_pre_upgrade_test_failed", False):
                capture_and_save_isvc_kueue_baseline(
                    pytestconfig=pytestconfig,
                    admin_client=admin_client,
                    isvc=isvc,
                )
            else:
                LOGGER.warning(
                    "Skipping baseline capture: pre-upgrade test(s) failed. "
                    "Post-upgrade tests will not have a valid baseline for comparison."
                )


@pytest.fixture
def skip_if_not_raw_deployment(
    kserve_kueue_upgrade_inference_service: InferenceService,
) -> None:
    """Skip tests when the Kueue upgrade ISVC is not deployed in RawDeployment mode."""
    skip_if_not_deployment_mode(
        isvc=kserve_kueue_upgrade_inference_service,
        deployment_types=KServeDeploymentType.RAW_DEPLOYMENT,
    )


# AdmissionCheck upgrade test fixtures


@pytest.fixture(scope="session")
def admission_check_namespace(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    """Namespace for AdmissionCheck upgrade tests, with Kueue management label."""
    ns = Namespace(client=admin_client, name=AC_NAMESPACE)

    if pytestconfig.option.post_upgrade:
        if not ns.exists:
            pytest.fail(
                f"[POST-UPGRADE] Namespace '{ns.name}' not found. Ensure pre-upgrade tests completed successfully."
            )
        yield ns
        if teardown_resources:
            ns.clean_up()
    elif ns.exists:
        pytest.fail(f"Unexpected existing namespace '{ns.name}'. Clear stale upgrade resources before pre-upgrade.")
    else:
        with create_ns(
            admin_client=admin_client,
            name=AC_NAMESPACE,
            model_mesh_enabled=False,
            add_dashboard_label=True,
            add_kueue_label=True,
            teardown=teardown_resources,
        ) as ns:
            yield ns


@pytest.fixture(scope="session")
def ensure_kueue_for_ac_upgrade(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    dsc_resource: DataScienceCluster,
    admission_check_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[None, Any, Any]:
    """Ensure Kueue DSC state for AdmissionCheck upgrade tests."""
    yield from _ensure_kueue_available_for_upgrade(
        pytestconfig=pytestconfig,
        admin_client=admin_client,
        dsc_resource=dsc_resource,
        namespace=admission_check_namespace.name,
        teardown_resources=teardown_resources,
    )


@pytest.fixture(scope="session")
def admission_check_kueue_resources(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    ensure_kueue_for_ac_upgrade: None,
    admission_check_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[dict, Any, Any]:
    """Kueue resources for AdmissionCheck upgrade test.

    Pre-upgrade: creates AdmissionCheck, ResourceFlavor, ClusterQueue, LocalQueue.
    Post-upgrade: looks up existing resources, cleans up on teardown.
    Yields dict with keys: local_queue, admission_check_name.
    """
    namespace = admission_check_namespace.name

    if pytestconfig.option.post_upgrade:
        local_queue = LocalQueue(
            client=admin_client,
            name=AC_LOCAL_QUEUE,
            cluster_queue=AC_CLUSTER_QUEUE,
            namespace=namespace,
        )
        cluster_queue = ClusterQueue(client=admin_client, name=AC_CLUSTER_QUEUE)
        resource_flavor = ResourceFlavor(client=admin_client, name=AC_RESOURCE_FLAVOR)
        admission_check = AdmissionCheck(
            client=admin_client, name=AC_ADMISSION_CHECK_NAME, controller_name=AC_CONTROLLER_NAME
        )
        missing_resources = [
            resource.name
            for resource in (local_queue, cluster_queue, resource_flavor, admission_check)
            if not resource.exists
        ]
        if missing_resources:
            pytest.fail(f"[POST-UPGRADE] Missing Kueue resources: {missing_resources}")
        yield {"local_queue": local_queue, "admission_check_name": AC_ADMISSION_CHECK_NAME}
        if teardown_resources:
            local_queue.clean_up()
            cluster_queue.clean_up()
            resource_flavor.clean_up()
            admission_check.clean_up()
    else:
        stale_resources = [
            name
            for name, resource_cls, kwargs in [
                (AC_ADMISSION_CHECK_NAME, AdmissionCheck, {"controller_name": AC_CONTROLLER_NAME}),
                (AC_CLUSTER_QUEUE, ClusterQueue, {}),
                (AC_RESOURCE_FLAVOR, ResourceFlavor, {}),
            ]
            if resource_cls(client=admin_client, name=name, **kwargs).exists
        ]
        if stale_resources:
            pytest.fail(
                f"Stale cluster-scoped Kueue resources found: {stale_resources}. "
                "Clear stale upgrade resources before pre-upgrade."
            )

        with (
            create_admission_check(
                client=admin_client,
                name=AC_ADMISSION_CHECK_NAME,
                controller_name=AC_CONTROLLER_NAME,
                teardown=teardown_resources,
            ),
            create_resource_flavor(
                client=admin_client,
                name=AC_RESOURCE_FLAVOR,
                teardown=teardown_resources,
            ),
            create_cluster_queue(
                client=admin_client,
                name=AC_CLUSTER_QUEUE,
                resource_groups=kueue_resource_groups(
                    flavor_name=AC_RESOURCE_FLAVOR,
                    cpu_quota=AC_CPU_QUOTA,
                    memory_quota=AC_MEMORY_QUOTA,
                ),
                admission_checks=[AC_ADMISSION_CHECK_NAME],
                teardown=teardown_resources,
            ),
            create_local_queue(
                client=admin_client,
                name=AC_LOCAL_QUEUE,
                cluster_queue=AC_CLUSTER_QUEUE,
                namespace=namespace,
                teardown=teardown_resources,
            ) as local_queue,
        ):
            activate_admission_check(client=admin_client, admission_check_name=AC_ADMISSION_CHECK_NAME)
            yield {"local_queue": local_queue, "admission_check_name": AC_ADMISSION_CHECK_NAME}


@pytest.fixture(scope="session")
def admission_check_job(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    admission_check_namespace: Namespace,
    admission_check_kueue_resources: dict,
    teardown_resources: bool,
) -> Generator[Job, Any, Any]:
    """Batch Job submitted to Kueue with AdmissionCheck gating.

    Pre-upgrade: creates a suspended Job with queue-name label.
    Post-upgrade: looks up the existing Job.
    """
    namespace = admission_check_namespace.name
    local_queue = admission_check_kueue_resources["local_queue"]

    if pytestconfig.option.post_upgrade:
        job = Job(client=admin_client, name=AC_JOB_NAME, namespace=namespace)
        if not job.exists:
            pytest.fail(
                f"[POST-UPGRADE] Job '{AC_JOB_NAME}' not found in namespace '{namespace}'. "
                "Ensure pre-upgrade tests completed successfully."
            )
        yield job
        if teardown_resources:
            job.clean_up()
    elif Job(client=admin_client, name=AC_JOB_NAME, namespace=namespace).exists:
        pytest.fail(
            f"Unexpected existing Job '{AC_JOB_NAME}' in namespace '{namespace}'. "
            "Clear stale upgrade resources before pre-upgrade."
        )
    else:
        job = Job(
            client=admin_client,
            kind_dict={
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {
                    "name": AC_JOB_NAME,
                    "namespace": namespace,
                    "labels": {"kueue.x-k8s.io/queue-name": local_queue.name},
                },
                "spec": {
                    "suspend": True,
                    "backoffLimit": 0,
                    "template": {
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [
                                {
                                    "name": "test",
                                    "image": "registry.access.redhat.com/ubi9/ubi-minimal:latest",
                                    "command": ["echo", "admission-check-test"],
                                    "resources": {
                                        "requests": {"cpu": AC_JOB_CPU_REQUEST, "memory": AC_JOB_MEMORY_REQUEST},
                                        "limits": {"cpu": AC_JOB_CPU_LIMIT, "memory": AC_JOB_MEMORY_LIMIT},
                                    },
                                }
                            ],
                        },
                    },
                },
            },
            teardown=teardown_resources,
        )
        job.deploy()
        yield job
        if teardown_resources:
            job.clean_up()


@pytest.fixture(scope="session")
def admission_check_workload(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    admission_check_namespace: Namespace,
    admission_check_job: Job,
    teardown_resources: bool,
) -> Generator[Workload, Any, Any]:
    """The Kueue Workload auto-created for the AdmissionCheck test Job.

    Pre-upgrade: polls until Kueue creates the Workload, saves name to ConfigMap.
    Post-upgrade: loads workload name from ConfigMap, returns the Workload.
    """
    namespace = admission_check_namespace.name

    if pytestconfig.option.post_upgrade:
        cm = ConfigMap(client=admin_client, name=AC_BASELINE_CM, namespace=namespace)
        assert cm.exists, f"Baseline ConfigMap '{AC_BASELINE_CM}' not found"
        baseline = json.loads(cm.instance.data["baseline"])
        workload = Workload(client=admin_client, name=baseline["workload_name"], namespace=namespace)
        yield workload
        if teardown_resources:
            workload.clean_up()
            cm.clean_up()
    else:
        job_uid = admission_check_job.instance.metadata.uid
        try:
            for workload in TimeoutSampler(
                wait_timeout=Timeout.TIMEOUT_4MIN,
                sleep=5,
                func=get_workload_for_job,
                client=admin_client,
                job_uid=job_uid,
                namespace=namespace,
            ):
                if workload:
                    break
        except TimeoutExpiredError:
            pytest.fail(f"Kueue did not create a Workload for Job '{admission_check_job.name}'")

        LOGGER.info(f"Kueue created Workload '{workload.name}' for Job '{admission_check_job.name}'")
        baseline = json.dumps({"workload_name": workload.name})
        cm = ConfigMap(
            client=admin_client,
            name=AC_BASELINE_CM,
            namespace=namespace,
            data={"baseline": baseline},
        )
        cm.deploy()
        yield workload
        if teardown_resources:
            cm.clean_up()
