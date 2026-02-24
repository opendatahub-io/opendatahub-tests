from collections.abc import Generator
from typing import Any

import pytest
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from simple_logger.logger import get_logger

from utilities.constants import (
    KServeDeploymentType,
    ModelAndFormat,
    ModelCarImage,
    ModelFormat,
    ModelStoragePath,
    ModelVersion,
    Protocols,
    RuntimeTemplates,
)
from utilities.inference_utils import create_isvc
from utilities.infra import (
    create_inference_token,
    create_isvc_view_role,
    create_ns,
    s3_endpoint_secret,
    update_configmap_data,
)
from utilities.logger import RedactedString
from utilities.serving_runtime import ServingRuntimeFromTemplate

LOGGER = get_logger(name=__name__)

UPGRADE_NAMESPACE = "upgrade-model-server"
AUTH_UPGRADE_NAMESPACE = "upgrade-auth-model-server"
MODEL_CAR_UPGRADE_NAMESPACE = "upgrade-model-car"
METRICS_UPGRADE_NAMESPACE = "upgrade-metrics"
PRIVATE_ENDPOINT_UPGRADE_NAMESPACE = "upgrade-private-endpoint"
S3_CONNECTION = "upgrade-connection"


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
    auth_service_account_fixture: ServiceAccount,
    auth_role_binding_fixture: RoleBinding,
) -> str:
    """Authentication token for upgrade tests."""
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
        "name": "private-endpoint-upgrade-connection",
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
        "name": "private-endpoint-upgrade-runtime",
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
        "name": "private-endpoint-upgrade-isvc",
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
