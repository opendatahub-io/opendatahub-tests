"""
Utility functions for AutoGluon KServe model serving tests.

Provides runtime image resolution, inference request helpers, fuzzy response validation,
and pytest parameter dictionary builders.
"""

import os
import time
from typing import Any, Literal

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError, ResourceNotUniqueError
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.autogluon.constant import (
    BASE_RAW_DEPLOYMENT_CONFIG,
    MODEL_CONFIGS,
    V1_PREDICT_PATH_TEMPLATE,
    V2_INFER_PATH_TEMPLATE,
    OutputType,
)
from utilities.constants import KServeDeploymentType
from utilities.inference_utils import get_exposed_isvc_url
from utilities.operator_utils import get_cluster_service_version

LOGGER = structlog.get_logger(name=__name__)

ProtocolVersionLiteral = Literal["v1", "v2"]


def cleanup_autogluon_inference_service(isvc: InferenceService) -> None:
    """
    Best-effort cleanup for AutoGluon InferenceService resources.

    Args:
        isvc: InferenceService resource to clean up.
    """
    max_attempts = 3
    sleep_seconds = 5

    for attempt in range(1, max_attempts + 1):
        try:
            if isvc.clean_up(wait=True):
                return
        except Exception as ex:  # noqa: BLE001
            LOGGER.warning(
                "AutoGluon InferenceService cleanup attempt failed",
                isvc_name=isvc.name,
                namespace=isvc.namespace,
                attempt=attempt,
                max_attempts=max_attempts,
                error=str(ex),
            )

        if attempt < max_attempts:
            time.sleep(sleep_seconds)

    LOGGER.warning(
        "Failed to cleanup AutoGluon InferenceService after retries",
        isvc_name=isvc.name,
        namespace=isvc.namespace,
        max_attempts=max_attempts,
    )


def get_inference_tls_verify() -> bool | str:
    """
    Resolve TLS verification mode for AutoGluon inference HTTP requests.

    Returns:
        True to verify with system trust store, False to disable verification,
        or a CA bundle file path.

    Raises:
        ValueError: If AUTOGLUON_INFERENCE_TLS_VERIFY has unsupported value.
    """
    ca_bundle_path = os.environ.get("AUTOGLUON_INFERENCE_CA_BUNDLE")
    if ca_bundle_path:
        return ca_bundle_path

    verify_env = os.environ.get("AUTOGLUON_INFERENCE_TLS_VERIFY", "true").strip().lower()
    if verify_env in {"1", "true", "yes", "on"}:
        return True
    if verify_env in {"0", "false", "no", "off"}:
        return False
    raise ValueError("Unsupported AUTOGLUON_INFERENCE_TLS_VERIFY value. Use one of: true,false,1,0,yes,no,on,off.")


def send_rest_request(url: str, input_data: dict[str, Any], verify: bool | str = True) -> Any:
    """
    Send a REST POST request with a JSON body.

    Args:
        url: Target endpoint URL.
        input_data: JSON-serializable request payload.
        verify: TLS verification mode (True, False, or CA bundle file path).

    Returns:
        Parsed JSON response body.

    Raises:
        requests.HTTPError: If the server returns an error status code.
    """
    response = requests.post(url=url, json=input_data, verify=verify, timeout=60)
    if not response.ok:
        response_body: Any
        try:
            response_body = response.json()
        except ValueError:
            response_body = response.text
        raise requests.HTTPError(
            f"Inference request failed with status={response.status_code} url={url} response_body={response_body}",
            response=response,
        )
    return response.json()


def get_inference_endpoint(
    isvc: InferenceService,
    protocol_version: ProtocolVersionLiteral,
    model_version: str,
) -> str:
    """
    Build the inference path for the given KServe protocol version.

    Args:
        isvc: InferenceService under test.
        protocol_version: KServe protocol version (v1 or v2).
        model_version: Model version string for versioned V2 paths.

    Returns:
        Relative URL path for the inference request.
    """
    _ = model_version  # V2 autogluonserver uses /v2/models/{name}/infer without a version segment
    model_name = isvc.instance.metadata.name
    if protocol_version == "v2":
        return V2_INFER_PATH_TEMPLATE.format(model_name=model_name)
    if protocol_version == "v1":
        return V1_PREDICT_PATH_TEMPLATE.format(model_name=model_name)
    raise ValueError(f"Unsupported protocol version: {protocol_version}")


def run_autogluon_inference(
    isvc: InferenceService,
    input_data: dict[str, Any],
    protocol_version: ProtocolVersionLiteral,
    model_version: str,
) -> Any:
    """
    Run inference against an AutoGluon predictor via external route.

    Args:
        isvc: InferenceService resource.
        input_data: Request payload.
        protocol_version: KServe protocol version (v1 or v2).
        model_version: Model version for V2 endpoints.

    Returns:
        Parsed JSON inference response.

    Raises:
        ValueError: If deployment mode is not supported for raw-style serving.
    """
    deployment_mode = isvc.instance.metadata.annotations.get("serving.kserve.io/deploymentMode")
    if deployment_mode not in KServeDeploymentType.RAW_DEPLOYMENT_MODES:
        raise ValueError(
            f"Unsupported deployment mode: {deployment_mode}. "
            f"Supported modes: {KServeDeploymentType.RAW_DEPLOYMENT_MODES}"
        )

    endpoint = get_inference_endpoint(
        isvc=isvc,
        protocol_version=protocol_version,
        model_version=model_version,
    )
    host = get_exposed_isvc_url(isvc=isvc)
    return send_rest_request(
        url=f"{host}{endpoint}",
        input_data=input_data,
        verify=get_inference_tls_verify(),
    )


def validate_deterministic_response(response: Any) -> None:
    """
    Validate deterministic AutoGluon responses using fuzzy structural checks.

    Args:
        response: Actual inference response.

    Raises:
        AssertionError: If the response structure is invalid or empty.
    """
    assert response, "Response is empty"
    assert isinstance(response, dict), f"Response is not a dict: {response}"

    if "outputs" in response:
        outputs = response["outputs"]
        assert isinstance(outputs, list), "Outputs must be a list"
        assert outputs, "Outputs list is empty"

        first_output = outputs[0]
        assert isinstance(first_output, dict), f"Output entry must be a dict, got {type(first_output).__name__}"
        output_data = first_output.get("data")
        assert isinstance(output_data, list), "Output data must be a list"
        assert output_data, "Output data is empty"
        return

    if "predictions" in response:
        predictions = response["predictions"]
        assert isinstance(predictions, list), "Predictions must be a list"
        assert predictions, "Predictions list is empty"
        return

    raise AssertionError(f"Unsupported response format, expected outputs/predictions keys: {response}")


def validate_inference_request(
    isvc: InferenceService,
    input_payload: dict[str, Any],
    protocol_version: ProtocolVersionLiteral,
    model_version: str,
    model_output_type: str,
) -> None:
    """
    Run inference and validate deterministic responses with fuzzy checks.

    Args:
        isvc: InferenceService resource.
        input_payload: Inference request body.
        protocol_version: KServe protocol version.
        model_version: Model version string.
        model_output_type: OutputType.DETERMINISTIC or OutputType.NON_DETERMINISTIC.
    """
    response = run_autogluon_inference(
        isvc=isvc,
        input_data=input_payload,
        protocol_version=protocol_version,
        model_version=model_version,
    )

    if model_output_type == OutputType.DETERMINISTIC:
        validate_deterministic_response(response=response)
    elif model_output_type == OutputType.NON_DETERMINISTIC:
        assert response, "Expected non-empty inference response"
    else:
        raise ValueError(f"Unsupported model output type: {model_output_type}")


def get_autogluon_image_from_csv(admin_client: DynamicClient, applications_namespace: str) -> str | None:
    """
    Resolve the AutoGluon server image from RHOAI CSV relatedImages.

    Args:
        admin_client: Kubernetes dynamic client.
        applications_namespace: Namespace where the operator CSV is installed.

    Returns:
        Matching image reference, or None if not found.
    """
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

    related_images = csv.instance.spec.get("relatedImages", [])
    for image_info in related_images:
        image_url = image_info.get("image", "")
        if "autogluon" in image_url.lower():
            LOGGER.info("Found AutoGluon image from RHOAI CSV", image_url=image_url)
            return image_url
    return None


def get_autogluon_image_from_related_images(related_images_refs: set[str]) -> str | None:
    """
    Pick an AutoGluon runtime image from a set of related image refs.

    Args:
        related_images_refs: Image refs collected from the cluster CSV.

    Returns:
        First image containing 'autogluon', or None.
    """
    for image_ref in sorted(related_images_refs):
        if "autogluon" in image_ref.lower():
            return image_ref
    return None


def resolve_autogluon_runtime_image(
    admin_client: DynamicClient,
    applications_namespace: str,
    related_images_refs: set[str],
    override_image: str | None = None,
) -> str:
    """
    Resolve the AutoGluon runtime container image.

    Resolution order: CLI/env override, CSV relatedImages, then related_images_refs fixture set.

    Args:
        admin_client: Kubernetes dynamic client.
        applications_namespace: Operator applications namespace.
        related_images_refs: Related images from session fixture.
        override_image: Optional image override from AUTOGLUON_RUNTIME_IMAGE or CLI.

    Returns:
        Container image reference.

    Raises:
        pytest.skip: If no image can be resolved on the cluster.
    """
    if override_image:
        return override_image

    image = get_autogluon_image_from_csv(
        admin_client=admin_client,
        applications_namespace=applications_namespace,
    )
    if image:
        return image

    image = get_autogluon_image_from_related_images(related_images_refs=related_images_refs)
    if image:
        return image

    pytest.skip(
        "AutoGluon runtime image not found. Install kserve-autogluonserver on the cluster, "
        "or set AUTOGLUON_RUNTIME_IMAGE for an override."
    )


def get_runtime_image_override() -> str | None:
    """Return optional AutoGluon runtime image from environment."""
    return os.environ.get("AUTOGLUON_RUNTIME_IMAGE") or None


def get_model_storage_uri_dict(predictor_type: str) -> dict[str, str]:
    """
    Build s3_models_storage_uri indirect parameter for a predictor variant.

    Args:
        predictor_type: Key in MODEL_CONFIGS (e.g. tabular-v2).

    Returns:
        dict with model-dir key for the S3 prefix.
    """
    if predictor_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown predictor type: {predictor_type}")
    prefix = MODEL_CONFIGS[predictor_type]["s3_prefix"].strip("/")
    return {"model-dir": prefix}


def get_model_namespace_dict(predictor_type: str) -> dict[str, str]:
    """
    Build model_namespace indirect parameter for a predictor variant.

    Args:
        predictor_type: Predictor type identifier.

    Returns:
        dict with unique namespace name.
    """
    return {"name": f"autogluon-{predictor_type}-s3"}


def get_deployment_config_dict(
    predictor_type: str,
    deployment_mode: str = KServeDeploymentType.STANDARD,
) -> dict[str, Any]:
    """
    Build serving runtime / inference service parameter dict.

    Args:
        predictor_type: Predictor type used as ISVC and SR parameter name.
        deployment_mode: KServe deployment mode.

    Returns:
        Deployment configuration dictionary for indirect fixtures.
    """
    if deployment_mode not in KServeDeploymentType.RAW_DEPLOYMENT_MODES:
        return {}
    return {
        "name": predictor_type,
        "predictor_type": predictor_type,
        **BASE_RAW_DEPLOYMENT_CONFIG,
        "deployment_mode": deployment_mode,
    }


def get_test_case_id(
    predictor_type: str,
    deployment_mode: str = KServeDeploymentType.STANDARD,
) -> str:
    """
    Generate pytest param id for an AutoGluon S3 test case.

    Args:
        predictor_type: Predictor variant name.
        deployment_mode: KServe deployment mode.

    Returns:
        Test case id string.
    """
    return f"{predictor_type}-s3-{deployment_mode.strip()}"
