from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.template import Template
from pytest import FixtureRequest

from tests.model_serving.model_runtime.triton.constant import PREDICT_RESOURCES, TRITON_IMAGE
from tests.model_serving.model_runtime.triton.probes.utils import (
    TRITON_LIVENESS_PROBE,
    TRITON_READINESS_PROBE,
    create_triton_template,
)
from utilities.constants import Containers, KServeDeploymentType, Protocols, Timeout
from utilities.inference_utils import create_isvc
from utilities.infra import get_pods_by_isvc_label
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def triton_probes_rest_serving_runtime_template(
    admin_client: DynamicClient, triton_probes_runtime_image: str
) -> Generator[Template]:
    """Create Triton REST runtime template for probe tests."""
    with create_triton_template(
        admin_client=admin_client, protocol=Protocols.REST, triton_runtime_image=triton_probes_runtime_image
    ) as template:
        yield template


@pytest.fixture(scope="class")
def triton_probes_model_service_account(
    admin_client: DynamicClient, kserve_s3_secret: Secret, model_namespace: Namespace
) -> Generator[ServiceAccount, Any, Any]:
    """Create service account for Triton model storage access."""
    with ServiceAccount(
        client=admin_client,
        namespace=model_namespace.name,
        name="triton-models-bucket-sa",
        secrets=[{"name": kserve_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def triton_probes_serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    triton_probes_runtime_image: str,
    triton_probes_rest_serving_runtime_template: Template,
) -> Generator[ServingRuntime, Any, Any]:
    """Triton REST ServingRuntime with readiness and liveness httpGet probes on kserve-container."""
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="triton-runtime",
        namespace=model_namespace.name,
        template_name=triton_probes_rest_serving_runtime_template.name,
        deployment_type=request.param["deployment_type"],
        runtime_image=triton_probes_runtime_image,
        containers={
            Containers.KSERVE_CONTAINER_NAME: {
                "readinessProbe": TRITON_READINESS_PROBE,
                "livenessProbe": TRITON_LIVENESS_PROBE,
            }
        },
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def triton_probes_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    triton_probes_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    triton_probes_model_service_account: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    """Triton REST InferenceService with probe-enabled runtime backed by S3 model storage."""
    isvc_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": request.param["name"],
        "namespace": model_namespace.name,
        "runtime": triton_probes_serving_runtime.name,
        "storage_uri": s3_models_storage_uri,
        "model_format": triton_probes_serving_runtime.instance.spec.supportedModelFormats[0].name,
        "model_service_account": triton_probes_model_service_account.name,
        "deployment_mode": request.param.get("deployment_mode", KServeDeploymentType.STANDARD),
        "external_route": True,
        "resources": PREDICT_RESOURCES.get("resources"),
        "volumes": PREDICT_RESOURCES.get("volumes"),
        "volumes_mounts": PREDICT_RESOURCES.get("volume_mounts"),
        "timeout": request.param.get("timeout", Timeout.TIMEOUT_20MIN),  # Increased for larger models
    }

    if min_replicas := request.param.get("min-replicas"):
        isvc_kwargs["min_replicas"] = min_replicas

    # Allow override of resources for larger models
    if custom_resources := request.param.get("resources"):
        isvc_kwargs["resources"] = custom_resources

    # Allow override of volumes for larger models (e.g., more shared memory)
    if custom_volumes := request.param.get("volumes"):
        isvc_kwargs["volumes"] = custom_volumes

    # Allow override of volume mounts
    if custom_volume_mounts := request.param.get("volume_mounts"):
        isvc_kwargs["volumes_mounts"] = custom_volume_mounts

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc


@pytest.fixture
def triton_probes_pod_resource(admin_client: DynamicClient, triton_probes_inference_service: InferenceService) -> Pod:
    """Get the predictor pod for the Triton probe InferenceService.

    Raises:
        AssertionError: If no pods found for the InferenceService.
    """
    pods = get_pods_by_isvc_label(client=admin_client, isvc=triton_probes_inference_service)
    assert pods, f"No pods found for InferenceService {triton_probes_inference_service.name}"
    return pods[0]


@pytest.fixture(scope="session")
def triton_probes_runtime_image() -> str:
    """Return the Triton runtime image to use for probe tests."""
    return TRITON_IMAGE
