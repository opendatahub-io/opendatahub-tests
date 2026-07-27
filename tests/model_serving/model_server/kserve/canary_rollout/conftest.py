"""Fixtures for KServe canary rollout (RawDeployment) tests."""

from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_serving.model_runtime.mlserver.probes.utils import (
    MLSERVER_LIVENESS_PROBE,
    MLSERVER_READINESS_PROBE,
)
from tests.model_serving.model_server.kserve.canary_rollout.constants import (
    CANARY_FEATURE_NAME,
    CANARY_MODEL_FORMAT,
    CANARY_NAMESPACE_PREFIX,
    CANARY_STORAGE_URI,
    DEFAULT_CANARY_TRAFFIC_PERCENT,
    DEFAULT_DEPLOYMENT_MODE,
    STABLE_MODEL_FORMAT,
    STABLE_STORAGE_URI,
)
from tests.model_serving.model_server.kserve.canary_rollout.utils import create_canary_inference_service
from utilities.constants import Containers, RuntimeTemplates
from utilities.infra import create_ns
from utilities.serving_runtime import ServingRuntimeFromTemplate

pytestmark = [pytest.mark.rawdeployment, pytest.mark.tier1]


@pytest.fixture(scope="package")
def canary_rollout_namespace(
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
) -> Generator[Namespace, Any, Any]:
    """Shared namespace for canary rollout tests."""
    with create_ns(
        admin_client=admin_client,
        unprivileged_client=unprivileged_client,
        name=f"{CANARY_NAMESPACE_PREFIX}-ns",
    ) as namespace:
        yield namespace


@pytest.fixture(scope="package")
def canary_mlserver_runtime(
    admin_client: DynamicClient,
    canary_rollout_namespace: Namespace,
    mlserver_runtime_image: str,
) -> Generator[ServingRuntime, Any, Any]:
    """MLServer ServingRuntime for status-code traffic fingerprinting (V2 infer)."""
    # Cluster MLServer template probes hit /v2/models/{{.Name}}/ready where .Name is the
    # InferenceService name. Canary ISVCs use names like kserve-canary-10, not sklearn.
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=f"{CANARY_FEATURE_NAME}-runtime",
        namespace=canary_rollout_namespace.name,
        template_name=RuntimeTemplates.MLSERVER,
        deployment_type=DEFAULT_DEPLOYMENT_MODE,
        runtime_image=mlserver_runtime_image,
        containers={
            Containers.KSERVE_CONTAINER_NAME: {
                "readinessProbe": {
                    **MLSERVER_READINESS_PROBE,
                    # Model loads in seconds; default 120s makes Ready look stuck.
                    "initialDelaySeconds": 10,
                },
                "livenessProbe": {
                    **MLSERVER_LIVENESS_PROBE,
                    "initialDelaySeconds": 30,
                },
            }
        },
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="package")
def canary_sklearn_inference_service(
    admin_client: DynamicClient,
    canary_rollout_namespace: Namespace,
    canary_mlserver_runtime: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    """Shared 10% canary ISVC for CRD/ROUTE (must not be permanently promoted)."""
    with create_canary_inference_service(
        client=admin_client,
        name=f"{CANARY_NAMESPACE_PREFIX}-10",
        namespace=canary_rollout_namespace.name,
        runtime=canary_mlserver_runtime.name,
        stable_model_format=STABLE_MODEL_FORMAT,
        stable_storage_uri=STABLE_STORAGE_URI,
        canary_model_format=CANARY_MODEL_FORMAT,
        canary_storage_uri=CANARY_STORAGE_URI,
        canary_traffic_percent=DEFAULT_CANARY_TRAFFIC_PERCENT,
        deployment_mode=DEFAULT_DEPLOYMENT_MODE,
        external_route=True,
    ) as isvc:
        yield isvc


@pytest.fixture
def canary_e2e_inference_service(
    admin_client: DynamicClient,
    canary_rollout_namespace: Namespace,
    canary_mlserver_runtime: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    """Dedicated ISVC for E2E promotion so package-scoped fixtures stay at 10% canary."""
    with create_canary_inference_service(
        client=admin_client,
        name=f"{CANARY_NAMESPACE_PREFIX}-e2e",
        namespace=canary_rollout_namespace.name,
        runtime=canary_mlserver_runtime.name,
        stable_model_format=STABLE_MODEL_FORMAT,
        stable_storage_uri=STABLE_STORAGE_URI,
        canary_model_format=CANARY_MODEL_FORMAT,
        canary_storage_uri=CANARY_STORAGE_URI,
        canary_traffic_percent=DEFAULT_CANARY_TRAFFIC_PERCENT,
        deployment_mode=DEFAULT_DEPLOYMENT_MODE,
        external_route=True,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="package")
def canary_ctrl_inference_service(
    admin_client: DynamicClient,
    canary_rollout_namespace: Namespace,
    canary_mlserver_runtime: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    """InferenceService with 20% canary traffic for controller behavior tests."""
    with create_canary_inference_service(
        client=admin_client,
        name=f"{CANARY_NAMESPACE_PREFIX}-ctrl",
        namespace=canary_rollout_namespace.name,
        runtime=canary_mlserver_runtime.name,
        stable_model_format=STABLE_MODEL_FORMAT,
        stable_storage_uri=STABLE_STORAGE_URI,
        canary_model_format=CANARY_MODEL_FORMAT,
        canary_storage_uri=CANARY_STORAGE_URI,
        canary_traffic_percent=20,
        deployment_mode=DEFAULT_DEPLOYMENT_MODE,
        external_route=True,
    ) as isvc:
        yield isvc
