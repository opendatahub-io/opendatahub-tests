from collections.abc import Generator
from typing import Any

import pytest
import structlog
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.gateway import Gateway
from ocp_resources.inference_service import InferenceService
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_serving.model_server.llmd.llmd_configs.config_upgrade import (
    LLMD_KUEUE_CLUSTER_QUEUE,
    LLMD_KUEUE_CPU_QUOTA,
    LLMD_KUEUE_LOCAL_QUEUE,
    LLMD_KUEUE_MEMORY_QUOTA,
    LLMD_KUEUE_RESOURCE_FLAVOR,
    UpgradeAuthKueueConfig,
)
from tests.model_serving.model_server.upgrade.utils import (
    UPGRADE_AUTH_TOKEN_SECRET_NAME,
    capture_isvc_baseline,
    capture_llmisvc_baseline,
    load_auth_token_from_secret,
    load_baseline_from_configmap,
    save_auth_token_to_secret,
    save_baseline_to_configmap,
)
from utilities.constants import (
    DscComponents,
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
from utilities.data_science_cluster_utils import get_dsc_ready_condition, wait_for_dsc_reconciliation
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
    create_cluster_queue,
    create_local_queue,
    create_resource_flavor,
    wait_for_kueue_crds_available,
)
from utilities.llmd_constants import KServeGateway, LLMDGateway
from utilities.llmd_utils import create_llmd_gateway
from utilities.logger import RedactedString
from utilities.serving_runtime import ServingRuntimeFromTemplate

LOGGER = structlog.get_logger(name=__name__)

UPGRADE_NAMESPACE = "upgrade-model-server"
AUTH_UPGRADE_NAMESPACE = "upgrade-auth-model-server"
MODEL_CAR_UPGRADE_NAMESPACE = "upgrade-model-car"
METRICS_UPGRADE_NAMESPACE = "upgrade-metrics"
PRIVATE_ENDPOINT_UPGRADE_NAMESPACE = "upgrade-pvt-ep"
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


# Post-Upgrade New ISVC Creation Fixtures
@pytest.fixture(scope="session")
def new_isvc_namespace_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
) -> Generator[Namespace, Any, Any]:
    """Namespace for creating a fresh ISVC post-upgrade."""
    if pytestconfig.option.post_upgrade:
        with create_ns(
            admin_client=admin_client,
            name=NEW_ISVC_UPGRADE_NAMESPACE,
            model_mesh_enabled=False,
            add_dashboard_label=True,
            teardown=True,
        ) as ns:
            yield ns
    else:
        yield None


@pytest.fixture(scope="session")
def new_isvc_serving_runtime_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    new_isvc_namespace_fixture: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    """ServingRuntime for fresh ISVC creation on upgraded control plane."""
    if pytestconfig.option.post_upgrade and new_isvc_namespace_fixture is not None:
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
    else:
        yield None


@pytest.fixture(scope="session")
def new_isvc_inference_service_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    new_isvc_serving_runtime_fixture: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    """Fresh InferenceService created on the upgraded control plane using Model Car (no S3)."""
    if pytestconfig.option.post_upgrade and new_isvc_serving_runtime_fixture is not None:
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
    else:
        yield None


# ---------------------------------------------------------------------------
# Fixtures used by LLMInferenceService upgrade tests
# ---------------------------------------------------------------------------


# llm-d gateway
@pytest.fixture(scope="session")
def llmisvc_upgrade_gateway(
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
        # No cleanup: the gateway is created by CI and shared across test runs.
        # Pre-upgrade creates it only if missing; post-upgrade reuses the existing one.
        yield gateway
    else:
        with create_llmd_gateway(
            client=admin_client,
            timeout=Timeout.TIMEOUT_1MIN,
            teardown=teardown_resources,
        ) as gateway:
            yield gateway


# llm-d namespaces
@pytest.fixture(scope="session")
def llmisvc_no_auth_namespace(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    """Namespace for LLMD upgrade tests."""
    ns = Namespace(client=admin_client, name="upgrade-llmd")

    if pytestconfig.option.post_upgrade:
        yield ns
        ns.clean_up()
    else:
        with create_ns(
            admin_client=admin_client,
            name="upgrade-llmd",
            model_mesh_enabled=False,
            add_dashboard_label=True,
            teardown=teardown_resources,
        ) as ns:
            yield ns


@pytest.fixture(scope="session")
def llmisvc_auth_and_kueue_namespace(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    """Namespace for auth+Kueue LLMISVC upgrade tests, with Kueue management label."""
    ns = Namespace(client=admin_client, name="upgrade-llmd-auth-and-kueue")

    if pytestconfig.option.post_upgrade:
        if not ns.exists:
            pytest.skip(
                f"[POST-UPGRADE] Namespace '{ns.name}' not found. "
                "These LLMInferenceService upgrade tests support pre-upgrade from RHOAI 3.4+. "
                "Upgrade paths from 3.3 or earlier do not create this namespace — skipping. "
                "If this is unexpected, verify that pre-upgrade tests completed successfully "
                "and that the correct upgrade path is being tested."
            )
        yield ns
        ns.clean_up()
    else:
        with create_ns(
            admin_client=admin_client,
            name="upgrade-llmd-auth-and-kueue",
            model_mesh_enabled=False,
            add_dashboard_label=True,
            add_kueue_label=True,
            teardown=teardown_resources,
        ) as ns:
            yield ns


# LLMInferenceService fixtures
@pytest.fixture(scope="session")
def llmisvc_upgrade_no_auth(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    llmisvc_no_auth_namespace: Namespace,
    llmisvc_upgrade_gateway: Gateway,
    teardown_resources: bool,
) -> Generator[LLMInferenceService, Any, Any]:
    """LLMInferenceService using TinyLlama OCI for upgrade tests."""
    from tests.model_serving.model_server.llmd.conftest import _create_llmisvc_from_config
    from tests.model_serving.model_server.llmd.llmd_configs import TinyLlamaOciConfig

    config_cls = TinyLlamaOciConfig
    llmisvc = LLMInferenceService(
        client=admin_client,
        name=config_cls.name,
        namespace=llmisvc_no_auth_namespace.name,
    )

    if pytestconfig.option.post_upgrade:
        yield llmisvc
        llmisvc.clean_up()
    else:
        with _create_llmisvc_from_config(
            config_cls=config_cls,
            namespace=llmisvc_no_auth_namespace.name,
            client=admin_client,
            teardown=teardown_resources,
        ) as llmisvc:
            yield llmisvc
            _capture_and_save_llmd_baseline(pytestconfig=pytestconfig, admin_client=admin_client, llmisvc=llmisvc)


@pytest.fixture(scope="session")
def llmisvc_upgrade_auth_and_kueue(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    llmisvc_auth_and_kueue_namespace: Namespace,
    llmisvc_upgrade_gateway: Gateway,
    llmisvc_upgrade_kueue_resources: LocalQueue,
    teardown_resources: bool,
) -> Generator[LLMInferenceService, Any, Any]:
    """Auth-enabled LLMInferenceService with Kueue integration for upgrade tests."""
    from tests.model_serving.model_server.llmd.conftest import _create_llmisvc_from_config

    config_cls = UpgradeAuthKueueConfig
    llmisvc = LLMInferenceService(
        client=admin_client,
        name=config_cls.name,
        namespace=llmisvc_auth_and_kueue_namespace.name,
    )

    if pytestconfig.option.post_upgrade:
        yield llmisvc
        if llmisvc.exists:
            llmisvc.clean_up()
    else:
        with _create_llmisvc_from_config(
            config_cls=config_cls,
            namespace=llmisvc_auth_and_kueue_namespace.name,
            client=admin_client,
            teardown=teardown_resources,
        ) as llmisvc:
            yield llmisvc
            _capture_and_save_llmd_baseline(pytestconfig=pytestconfig, admin_client=admin_client, llmisvc=llmisvc)


# Kueue for upgrade tests
def _restore_kueue_dsc_state(
    admin_client: DynamicClient,
    dsc_resource,
    namespace: str,
    kueue_dsc_state_cm_name: str,
) -> None:
    """Restore original Kueue managementState from saved ConfigMap."""
    state_cm = ConfigMap(client=admin_client, name=kueue_dsc_state_cm_name, namespace=namespace)
    if state_cm.exists:
        original_state = state_cm.instance.data.get("original_management_state")
        if original_state:
            LOGGER.info(f"Restoring Kueue managementState to '{original_state}' in DSC")
            dsc_resource.update(
                resource_dict={
                    "metadata": {"name": dsc_resource.name},
                    "spec": {"components": {DscComponents.KUEUE: {"managementState": original_state}}},
                }
            )
        state_cm.clean_up()


@pytest.fixture(scope="session")
def ensure_kueue_for_upgrade(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    dsc_resource,
    llmisvc_auth_and_kueue_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[None, Any, Any]:
    """Ensure Kueue is available for upgrade tests.

    Pre-upgrade:
      1. Skip if kueue operator is not installed.
      2. Save the original DSC kueue managementState to a ConfigMap.
      3. Patch DSC kueue to Unmanaged if needed (direct update, no ResourceEditor restore).
      4. Wait for CRDs and controller pods.
      5. Teardown: if --delete-pre-upgrade-resources, restore original state.

    Post-upgrade:
      1. Tests run without mutating DSC — verify the real post-upgrade state.
      2. Teardown: read the original state from ConfigMap and restore it.
    """
    namespace = llmisvc_auth_and_kueue_namespace.name
    kueue_dsc_state_cm_name = "upgrade-kueue-dsc-state"

    if not pytestconfig.option.post_upgrade:
        from tests.model_serving.model_server.conftest import _is_kueue_operator_installed

        # Step 1: check kueue operator is installed
        if not _is_kueue_operator_installed(admin_client):
            pytest.skip("Kueue operator is not installed, skipping Kueue upgrade tests")

        # Step 2: save original state to ConfigMap (for restore in post-upgrade or pre-upgrade cleanup)
        kueue_management_state = dsc_resource.instance.spec.components[DscComponents.KUEUE].managementState
        LOGGER.info(f"Saving original Kueue managementState '{kueue_management_state}' to ConfigMap")
        ConfigMap(
            client=admin_client,
            name=kueue_dsc_state_cm_name,
            namespace=namespace,
            data={"original_management_state": kueue_management_state},
        ).deploy()

        # Step 3: patch to Unmanaged if needed (state must persist through upgrade)
        if kueue_management_state != DscComponents.ManagementState.UNMANAGED:
            LOGGER.info(f"Patching Kueue from {kueue_management_state} to Unmanaged")
            ready_condition = get_dsc_ready_condition(dsc=dsc_resource)
            pre_patch_time = ready_condition.get("lastTransitionTime") if ready_condition else None
            dsc_resource.update(
                resource_dict={
                    "metadata": {"name": dsc_resource.name},
                    "spec": {
                        "components": {
                            DscComponents.KUEUE: {"managementState": DscComponents.ManagementState.UNMANAGED}
                        }
                    },
                }
            )
            wait_for_dsc_reconciliation(dsc=dsc_resource, baseline_time=pre_patch_time)
        else:
            LOGGER.info("Kueue already Unmanaged, no patch needed")

        # Step 4: wait for kueue CRDs and controller
        wait_for_kueue_crds_available(client=admin_client)
        yield

        # Step 5: restore if --delete-pre-upgrade-resources (debugging mode)
        if teardown_resources:
            _restore_kueue_dsc_state(
                admin_client=admin_client,
                dsc_resource=dsc_resource,
                namespace=namespace,
                kueue_dsc_state_cm_name=kueue_dsc_state_cm_name,
            )
    else:
        # Post-upgrade: tests run without mutating DSC state
        yield

        # Teardown: restore original kueue managementState from saved ConfigMap
        _restore_kueue_dsc_state(
            admin_client=admin_client,
            dsc_resource=dsc_resource,
            namespace=namespace,
            kueue_dsc_state_cm_name=kueue_dsc_state_cm_name,
        )


@pytest.fixture(scope="session")
def llmisvc_upgrade_kueue_resources(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    ensure_kueue_for_upgrade,
    llmisvc_auth_and_kueue_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[LocalQueue, Any, Any]:
    """Create Kueue resources (ResourceFlavor, ClusterQueue, LocalQueue) for upgrade tests.

    Pre-upgrade: creates resources (ensure_kueue_for_upgrade handles DSC setup).
    Post-upgrade: looks up the LocalQueue without mutating DSC state.
    """
    from tests.model_serving.model_server.conftest import kueue_resource_groups

    namespace = llmisvc_auth_and_kueue_namespace.name

    if pytestconfig.option.post_upgrade:
        local_queue = LocalQueue(
            client=admin_client,
            name=LLMD_KUEUE_LOCAL_QUEUE,
            cluster_queue=LLMD_KUEUE_CLUSTER_QUEUE,
            namespace=namespace,
        )
        yield local_queue
        local_queue.clean_up()
        ClusterQueue(client=admin_client, name=LLMD_KUEUE_CLUSTER_QUEUE).clean_up()
        ResourceFlavor(client=admin_client, name=LLMD_KUEUE_RESOURCE_FLAVOR).clean_up()
    else:
        with (
            create_resource_flavor(
                client=admin_client,
                name=LLMD_KUEUE_RESOURCE_FLAVOR,
                teardown=teardown_resources,
            ),
            create_cluster_queue(
                client=admin_client,
                name=LLMD_KUEUE_CLUSTER_QUEUE,
                resource_groups=kueue_resource_groups(
                    flavor_name=LLMD_KUEUE_RESOURCE_FLAVOR,
                    cpu_quota=LLMD_KUEUE_CPU_QUOTA,
                    memory_quota=LLMD_KUEUE_MEMORY_QUOTA,
                ),
                teardown=teardown_resources,
            ),
            create_local_queue(
                client=admin_client,
                name=LLMD_KUEUE_LOCAL_QUEUE,
                cluster_queue=LLMD_KUEUE_CLUSTER_QUEUE,
                namespace=namespace,
                teardown=teardown_resources,
            ) as local_queue,
        ):
            yield local_queue


# Auth for upgrade tests
@pytest.fixture(scope="session")
def llmisvc_upgrade_token(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    llmisvc_upgrade_auth_and_kueue: LLMInferenceService,
) -> Generator[str, Any, Any]:
    """Auth token for the auth-enabled LLMISVC, persisted across upgrade via a Secret.

    Pre-upgrade: creates a ServiceAccount, Role (get on the LLMISVC), and RoleBinding,
    generates a token from the SA, and saves it to a Secret on the cluster.
    Post-upgrade: loads the same token from the Secret. Tests use it to verify that
    auth RBAC survived the upgrade and the token is still accepted.
    Teardown (post-upgrade only): cleans up SA, Role, RoleBinding, and the token Secret.

    Args:
        pytestconfig: Pytest config to check pre/post upgrade mode.
        admin_client: Kubernetes dynamic client.
        llmisvc_upgrade_auth_and_kueue: The auth-enabled LLMISVC to create RBAC for.

    Yields:
        RedactedString with the auth token.
    """
    svc = llmisvc_upgrade_auth_and_kueue
    namespace = svc.namespace

    if pytestconfig.option.post_upgrade:
        token = load_auth_token_from_secret(
            client=admin_client,
            namespace=namespace,
        )
        yield RedactedString(value=token)
        ServiceAccount(client=admin_client, namespace=namespace, name=f"{svc.name}-auth-sa").clean_up()
        Role(client=admin_client, name=f"{svc.name}-view", namespace=namespace).clean_up()
        RoleBinding(client=admin_client, name=f"{svc.name}-auth-sa-view", namespace=namespace).clean_up()
        Secret(client=admin_client, name=UPGRADE_AUTH_TOKEN_SECRET_NAME, namespace=namespace).clean_up()
    else:
        sa = ServiceAccount(client=admin_client, namespace=namespace, name=f"{svc.name}-auth-sa")
        sa.deploy()

        role = Role(
            client=admin_client,
            name=f"{svc.name}-view",
            namespace=namespace,
            rules=[
                {
                    "apiGroups": [svc.api_group],
                    "resources": ["llminferenceservices"],
                    "verbs": ["get"],
                    "resourceNames": [svc.name],
                }
            ],
        )
        role.deploy()

        RoleBinding(
            client=admin_client,
            namespace=namespace,
            name=f"{svc.name}-auth-sa-view",
            role_ref_name=role.name,
            role_ref_kind=role.kind,
            subjects_kind="ServiceAccount",
            subjects_name=sa.name,
        ).deploy()

        token = create_inference_token(model_service_account=sa)
        save_auth_token_to_secret(
            client=admin_client,
            namespace=namespace,
            token=token,
        )
        yield RedactedString(value=token)


# Baseline for llm-d upgrade tests
def _capture_and_save_llmd_baseline(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    llmisvc: LLMInferenceService,
) -> None:
    """Capture LLMISVC baseline and save ConfigMap in the LLMISVC's own namespace. No-op during post-upgrade."""
    if pytestconfig.option.post_upgrade:
        return

    baselines = {
        llmisvc.name: capture_llmisvc_baseline(client=admin_client, llmisvc=llmisvc),
    }
    save_baseline_to_configmap(
        client=admin_client,
        namespace=llmisvc.namespace,
        baselines=baselines,
    )
