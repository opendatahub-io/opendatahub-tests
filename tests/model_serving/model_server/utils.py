import json
import re
from contextlib import contextmanager
from string import Template
from typing import Any, Dict, Generator, Optional

from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from simple_logger.logger import get_logger

from tests.model_serving.model_server.private_endpoint.utils import (
    InvalidStorageArgument,
)
from utilities.constants import KServeDeploymentType
from utilities.inference_utils import UserInference


LOGGER = get_logger(name=__name__)


class InferenceResponseError(Exception):
    pass


@contextmanager
def create_isvc(
    client: DynamicClient,
    name: str,
    namespace: str,
    deployment_mode: str,
    model_format: str,
    runtime: str,
    storage_uri: Optional[str] = None,
    storage_key: Optional[str] = None,
    storage_path: Optional[str] = None,
    wait: bool = True,
    enable_auth: bool = False,
    external_route: Optional[bool] = None,
    model_service_account: Optional[str] = "",
    min_replicas: Optional[int] = None,
    argument: Optional[list[str]] = None,
    resources: Optional[dict[str, Any]] = None,
    volumes: Optional[dict[str, Any]] = None,
    volumes_mounts: Optional[dict[str, Any]] = None,
    model_version: Optional[str] = None,
) -> Generator[InferenceService, Any, Any]:
    labels: Dict[str, str] = {}
    predictor_dict: Dict[str, Any] = {
        "minReplicas": min_replicas,
        "model": {
            "modelFormat": {"name": model_format},
            "version": "1",
            "runtime": runtime,
        },
    }

    if model_version:
        predictor_dict["model"]["modelFormat"]["version"] = model_version

    _check_storage_arguments(storage_uri, storage_key, storage_path)
    if storage_uri:
        predictor_dict["model"]["storageUri"] = storage_uri
    elif storage_key:
        predictor_dict["model"]["storage"] = {"key": storage_key, "path": storage_path}
    if model_service_account:
        predictor_dict["serviceAccountName"] = model_service_account

    if min_replicas:
        predictor_dict["minReplicas"] = min_replicas
    if argument:
        predictor_dict["model"]["args"] = argument
    if resources:
        predictor_dict["model"]["resources"] = resources
    if volumes_mounts:
        predictor_dict["model"]["volumeMounts"] = volumes_mounts
    if volumes:
        predictor_dict["volumes"] = volumes
    annotations = {"serving.kserve.io/deploymentMode": deployment_mode}

    if deployment_mode == KServeDeploymentType.SERVERLESS:
        annotations.update({
            "serving.knative.openshift.io/enablePassthrough": "true",
            "sidecar.istio.io/inject": "true",
            "sidecar.istio.io/rewriteAppHTTPProbers": "true",
        })

    if enable_auth:
        # TODO: add modelmesh support
        if deployment_mode == KServeDeploymentType.SERVERLESS:
            annotations["security.opendatahub.io/enable-auth"] = "true"
        elif deployment_mode == KServeDeploymentType.RAW_DEPLOYMENT:
            labels["security.openshift.io/enable-authentication"] = "true"

    # default to True if deployment_mode is Serverless (default behavior of Serverless) if was not provided by the user
    if external_route is None and deployment_mode == KServeDeploymentType.SERVERLESS:
        external_route = True

    if external_route and deployment_mode == KServeDeploymentType.RAW_DEPLOYMENT:
        labels["networking.kserve.io/visibility"] = "exposed"

    if deployment_mode == KServeDeploymentType.SERVERLESS and external_route is False:
        labels["networking.knative.dev/visibility"] = "cluster-local"

    with InferenceService(
        client=client,
        name=name,
        namespace=namespace,
        annotations=annotations,
        predictor=predictor_dict,
        label=labels,
    ) as inference_service:
        if wait:
            inference_service.wait_for_condition(
                condition=inference_service.Condition.READY,
                status=inference_service.Condition.Status.TRUE,
                timeout=10 * 60,
            )
        yield inference_service


def _check_storage_arguments(
    storage_uri: Optional[str],
    storage_key: Optional[str],
    storage_path: Optional[str],
) -> None:
    if (storage_uri and storage_path) or (not storage_uri and not storage_key) or (storage_key and not storage_path):
        raise InvalidStorageArgument(storage_uri, storage_key, storage_path)


def verify_inference_response(
    inference_service: InferenceService,
    runtime: str,
    inference_type: str,
    protocol: str,
    model_name: Optional[str] = None,
    text: Optional[str] = None,
    use_default_query: bool = False,
    expected_response_text: Optional[str] = None,
    insecure: bool = False,
    token: Optional[str] = None,
    authorized_user: Optional[bool] = None,
) -> None:
    model_name = model_name or inference_service.name

    inference = UserInference(
        inference_service=inference_service,
        runtime=runtime,
        inference_type=inference_type,
        protocol=protocol,
    )

    res = inference.run_inference(
        model_name=model_name,
        text=text,
        use_default_query=use_default_query,
        token=token,
        insecure=insecure,
    )

    if authorized_user is False:
        auth_header = "x-ext-auth-reason"

        if auth_reason := re.search(rf"{auth_header}: (.*)", res["output"], re.MULTILINE):
            reason = auth_reason.group(1).lower()

            if token:
                assert re.search(r"not (?:authenticated|authorized)", reason)

            else:
                assert "credential not found" in reason

        else:
            raise ValueError(f"Auth header {auth_header} not found in response. Response: {res['output']}")

    else:
        use_regex = False

        if use_default_query:
            expected_response_text_config = inference.inference_config.get("default_query_model")
            if not expected_response_text_config:
                raise ValueError(
                    f"Missing default_query_model config for inference {runtime}. "
                    f"Config: {expected_response_text_config}"
                )

            if inference.inference_config.get("support_multi_default_queries"):
                query_config = expected_response_text_config.get(inference_type)
                if not query_config:
                    raise ValueError(
                        f"Missing default_query_model config for inference {runtime}. "
                        f"Config: {expected_response_text_config}"
                    )
                expected_response_text = query_config.get("query_output", "")
                use_regex = query_config.get("use_regex", False)

            else:
                expected_response_text = expected_response_text_config.get("query_output")

            if not expected_response_text:
                raise ValueError(f"Missing response text key for inference {runtime}")

            if isinstance(expected_response_text, dict):
                expected_response_text = Template(expected_response_text.get("response_output")).safe_substitute(
                    model_name=model_name
                )

        if inference.inference_response_text_key_name:
            if inference_type == inference.STREAMING:
                if output := re.findall(
                    rf"{inference.inference_response_text_key_name}\": \"(.*)\"",
                    res[inference.inference_response_key_name],
                    re.MULTILINE,
                ):
                    assert "".join(output) == expected_response_text

            elif inference_type == inference.INFER:
                assert json.dumps(res[inference.inference_response_key_name]).replace(" ", "") == expected_response_text

            elif use_regex:
                assert re.search(expected_response_text, json.dumps(res[inference.inference_response_text_key_name]))  # type: ignore[arg-type]

            else:
                response = res[inference.inference_response_key_name]
                if isinstance(response, list):
                    response = response[0]

                assert response[inference.inference_response_text_key_name] == expected_response_text

        else:
            raise InferenceResponseError(f"Inference response output not found in response. Response: {res}")
