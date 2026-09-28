"""Shared AutoGluon runtime configuration and image-resolution helpers."""

import os
from typing import Any

import requests
import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError, ResourceNotUniqueError
from ocp_resources.inference_service import InferenceService

from utilities.constants import KServeDeploymentType, ModelFormat, ModelVersion
from utilities.inference_utils import get_exposed_isvc_url
from utilities.operator_utils import get_cluster_service_version

LOGGER = structlog.get_logger(name=__name__)

PREDICT_RESOURCES: dict[str, dict[str, dict[str, str]]] = {
    "resources": {"requests": {"cpu": "1", "memory": "2Gi"}, "limits": {"cpu": "1", "memory": "2Gi"}},
}


class ProtocolVersion:
    """KServe protocol versions used for AutoGluon inference."""

    V1: str = "v1"
    V2: str = "v2"


def build_serving_runtime_kwargs(namespace: str, image: str, name: str) -> dict[str, Any]:
    """Build keyword arguments for a namespace-scoped AutoGluon ServingRuntime."""
    return {
        "name": name,
        "namespace": namespace,
        "annotations": {
            "opendatahub.io/dashboard": "true",
            "opendatahub.io/kserve-runtime": ModelFormat.AUTOGLUON,
            "openshift.io/display-name": "AutoGluon Runtime",
        },
        "spec_annotations": {"prometheus.io/path": "/metrics", "prometheus.io/port": "8080"},
        "multi_model": False,
        "protocol_versions": [ProtocolVersion.V1, ProtocolVersion.V2],
        "supported_model_formats": [
            {"name": ModelFormat.AUTOGLUON, "version": ModelVersion.AUTOGLUON_1, "autoSelect": True},
        ],
        "containers": [
            {
                "name": "kserve-container",
                "image": image,
                "args": ["--model_name={{.Name}}", "--model_dir=/mnt/models", "--http_port=8080"],
                "ports": [{"containerPort": 8080, "protocol": "TCP"}],
                "resources": PREDICT_RESOURCES["resources"],
            },
        ],
    }


def get_autogluon_image_from_csv(admin_client: DynamicClient, applications_namespace: str) -> str | None:
    """Resolve the AutoGluon server image from RHOAI CSV related images."""
    try:
        csv = get_cluster_service_version(
            client=admin_client,
            prefix="rhods-operator",
            namespace=applications_namespace,
        )
    except (ResourceNotFoundError, ResourceNotUniqueError) as ex:
        LOGGER.warning(
            "Skipping AutoGluon CSV image lookup and using fallback chain",
            namespace=applications_namespace,
            error=str(ex),
        )
        return None

    for image_info in csv.instance.spec.get("relatedImages", []):
        image_url = image_info.get("image", "")
        if "autogluon" in image_url.lower():
            LOGGER.info("Found AutoGluon image from RHOAI CSV", image_url=image_url)
            return image_url
    return None


def get_runtime_image_override() -> str | None:
    """Return an optional AutoGluon runtime image from the environment."""
    return os.environ.get("AUTOGLUON_RUNTIME_IMAGE") or None


def run_autogluon_inference(
    isvc: InferenceService,
    input_data: dict[str, Any],
    protocol_version: str,
    model_version: str,
) -> Any:
    """Run an inference request against an externally exposed AutoGluon service."""
    _ = model_version
    deployment_mode = isvc.instance.metadata.annotations.get("serving.kserve.io/deploymentMode")
    if deployment_mode not in KServeDeploymentType.RAW_DEPLOYMENT_MODES:
        raise ValueError(f"Unsupported deployment mode: {deployment_mode}")
    model_name = isvc.instance.metadata.name
    if protocol_version == ProtocolVersion.V2:
        endpoint = f"/v2/models/{model_name}/infer"
    elif protocol_version == ProtocolVersion.V1:
        endpoint = f"/v1/models/{model_name}:predict"
    else:
        raise ValueError(f"Unsupported protocol version: {protocol_version}")
    response = requests.post(
        url=f"{get_exposed_isvc_url(isvc=isvc)}{endpoint}",
        json=input_data,
        verify=_get_inference_tls_verify(),
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _get_inference_tls_verify() -> bool | str:
    """Resolve AutoGluon inference TLS verification from the environment."""
    ca_bundle_path = os.environ.get("AUTOGLUON_INFERENCE_CA_BUNDLE")
    if ca_bundle_path:
        return ca_bundle_path
    verify_env = os.environ.get("AUTOGLUON_INFERENCE_TLS_VERIFY", "true").strip().lower()
    if verify_env in {"1", "true", "yes", "on"}:
        return True
    if verify_env in {"0", "false", "no", "off"}:
        return False
    raise ValueError("Unsupported AUTOGLUON_INFERENCE_TLS_VERIFY value")


def validate_deterministic_response(response: Any) -> None:
    """Validate deterministic AutoGluon responses using structural checks."""
    assert response, "Response is empty"
    assert isinstance(response, dict), f"Response is not a dict: {response}"

    if "outputs" in response:
        outputs = response["outputs"]
        assert isinstance(outputs, list), "Outputs is not a list"
        assert outputs, "Outputs list is empty"
        first_output = outputs[0]
        assert isinstance(first_output, dict), f"Output entry is not a dict: {first_output}"
        output_data = first_output.get("data")
        assert isinstance(output_data, list), "Output data is not a list"
        assert output_data, "Output data is empty"
        return

    if "predictions" in response:
        predictions = response["predictions"]
        assert isinstance(predictions, list), "Predictions is not a list"
        assert predictions, "Predictions list is empty"
        return

    raise AssertionError(f"Unsupported response format, expected outputs/predictions keys: {response}")
