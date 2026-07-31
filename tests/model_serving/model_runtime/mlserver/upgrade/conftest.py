"""Session-scoped fixtures for MLServer upgrade tests.

Provides fixtures that persist through the RHOAI upgrade cycle:
- Namespace, S3 secret, ServiceAccount, ServingRuntime, InferenceService
- Pre-upgrade baseline ConfigMap for post-upgrade comparison
"""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_serving.model_runtime.mlserver.constant import MODEL_CONFIGS
from tests.model_serving.model_runtime.mlserver.upgrade.constant import (
    UPGRADE_BASELINE_CM,
    UPGRADE_ISVC_NAME,
    UPGRADE_NAMESPACE,
    UPGRADE_RESTART_KEY,
    UPGRADE_SA_NAME,
    UPGRADE_SECRET_NAME,
)
from tests.model_serving.model_runtime.mlserver.utils import (
    run_mlserver_inference,
    validate_deterministic_snapshot,
)
from tests.model_serving.model_runtime.utils import get_restart_counts
from utilities.constants import (
    KServeDeploymentType,
    ModelFormat,
    ModelInferenceRuntime,
    Protocols,
    RuntimeTemplates,
)
from utilities.inference_utils import create_isvc
from utilities.infra import create_ns, get_pods_by_isvc_label, s3_endpoint_secret
from utilities.serving_runtime import ServingRuntimeFromTemplate

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="session")
def mlserver_upgrade_namespace(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace]:
    """Namespace for MLServer upgrade tests — persists through upgrade."""
    ns = Namespace(client=admin_client, name=UPGRADE_NAMESPACE)

    if pytestconfig.option.post_upgrade:
        yield ns
        if teardown_resources:
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
def mlserver_upgrade_s3_secret(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    mlserver_upgrade_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
    teardown_resources: bool,
) -> Generator[Secret]:
    """S3 credentials secret for MLServer upgrade tests."""
    secret = Secret(client=admin_client, name=UPGRADE_SECRET_NAME, namespace=mlserver_upgrade_namespace.name)

    if pytestconfig.option.post_upgrade:
        yield secret
        secret.clean_up()
    else:
        with s3_endpoint_secret(
            client=admin_client,
            name=UPGRADE_SECRET_NAME,
            namespace=mlserver_upgrade_namespace.name,
            aws_access_key=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_s3_bucket=models_s3_bucket_name,
            aws_s3_endpoint=models_s3_bucket_endpoint,
            aws_s3_region=models_s3_bucket_region,
            teardown=teardown_resources,
        ) as secret:
            yield secret


@pytest.fixture(scope="session")
def mlserver_upgrade_service_account(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    mlserver_upgrade_namespace: Namespace,
    mlserver_upgrade_s3_secret: Secret,
    teardown_resources: bool,
) -> Generator[ServiceAccount]:
    """Service account for MLServer upgrade tests."""
    sa = ServiceAccount(
        client=admin_client,
        name=UPGRADE_SA_NAME,
        namespace=mlserver_upgrade_namespace.name,
    )

    if pytestconfig.option.post_upgrade:
        yield sa
        sa.clean_up()
    else:
        with ServiceAccount(
            client=admin_client,
            namespace=mlserver_upgrade_namespace.name,
            name=UPGRADE_SA_NAME,
            secrets=[{"name": mlserver_upgrade_s3_secret.name}],
            teardown=teardown_resources,
        ) as sa:
            yield sa


@pytest.fixture(scope="session")
def mlserver_upgrade_serving_runtime(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    mlserver_upgrade_namespace: Namespace,
    mlserver_runtime_image: str | None,
    teardown_resources: bool,
) -> Generator[ServingRuntime]:
    """CPU MLServer ServingRuntime for upgrade tests."""
    runtime = ServingRuntime(
        client=admin_client,
        name=ModelInferenceRuntime.MLSERVER_RUNTIME,
        namespace=mlserver_upgrade_namespace.name,
    )

    if pytestconfig.option.post_upgrade:
        yield runtime
        runtime.clean_up()
    else:
        with ServingRuntimeFromTemplate(
            client=admin_client,
            name=ModelInferenceRuntime.MLSERVER_RUNTIME,
            namespace=mlserver_upgrade_namespace.name,
            template_name=RuntimeTemplates.MLSERVER,
            deployment_type=KServeDeploymentType.STANDARD,
            runtime_image=mlserver_runtime_image,
            teardown=teardown_resources,
        ) as runtime:
            yield runtime


@pytest.fixture(scope="session")
def mlserver_upgrade_inference_service(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    mlserver_upgrade_namespace: Namespace,
    mlserver_upgrade_serving_runtime: ServingRuntime,
    mlserver_upgrade_service_account: ServiceAccount,
    models_s3_bucket_name: str,
    teardown_resources: bool,
) -> Generator[InferenceService]:
    """CPU MLServer InferenceService for upgrade tests."""
    isvc_kwargs = {
        "client": admin_client,
        "name": UPGRADE_ISVC_NAME,
        "namespace": mlserver_upgrade_namespace.name,
    }

    isvc = InferenceService(**isvc_kwargs)

    if pytestconfig.option.post_upgrade:
        yield isvc
        isvc.clean_up()
    else:
        storage_uri = f"s3://{models_s3_bucket_name}/mlserver/model_repository/{ModelFormat.SKLEARN}/"
        with create_isvc(
            **isvc_kwargs,
            runtime=mlserver_upgrade_serving_runtime.name,
            model_format=ModelFormat.SKLEARN,
            storage_uri=storage_uri,
            model_service_account=mlserver_upgrade_service_account.name,
            deployment_mode=KServeDeploymentType.STANDARD,
            external_route=True,
            teardown=teardown_resources,
        ) as isvc:
            yield isvc


@pytest.fixture(scope="session")
def mlserver_upgrade_baseline(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    mlserver_upgrade_namespace: Namespace,
    mlserver_upgrade_inference_service: InferenceService,
) -> ConfigMap:
    """Captures pre-upgrade baseline restart counts to ConfigMap.

    Pre-upgrade: runs inference, records restart counts.
    Post-upgrade: returns existing ConfigMap for comparison.
    """
    sklearn_config: dict[str, Any] = MODEL_CONFIGS[ModelFormat.SKLEARN]

    if pytestconfig.option.post_upgrade:
        cm = ConfigMap(
            client=admin_client,
            name=UPGRADE_BASELINE_CM,
            namespace=mlserver_upgrade_namespace.name,
        )
        assert cm.exists, f"Baseline ConfigMap '{UPGRADE_BASELINE_CM}' not found after upgrade"
        return cm

    predictor_pods = get_pods_by_isvc_label(client=admin_client, isvc=mlserver_upgrade_inference_service)
    assert predictor_pods, f"No predictor pods for ISVC '{mlserver_upgrade_inference_service.name}'"

    baseline_response = run_mlserver_inference(
        isvc=mlserver_upgrade_inference_service,
        input_data=sklearn_config["rest_query"],
        model_version=sklearn_config["model_version"],
        protocol=Protocols.REST,
    )
    validate_deterministic_snapshot(response=baseline_response)

    restart_counts = get_restart_counts(pod=predictor_pods[0])
    cm = ConfigMap(
        client=admin_client,
        name=UPGRADE_BASELINE_CM,
        namespace=mlserver_upgrade_namespace.name,
        data={UPGRADE_RESTART_KEY: json.dumps(restart_counts)},
        teardown=False,
    )
    cm.deploy()
    LOGGER.info(
        "Pre-upgrade baseline recorded",
        isvc_name=mlserver_upgrade_inference_service.name,
        restart_counts=restart_counts,
    )
    return cm
