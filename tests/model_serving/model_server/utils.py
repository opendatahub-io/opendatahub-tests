import json
import re
from contextlib import contextmanager
from string import Template
from typing import Any, Dict, Generator, Optional

from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from utilities.constants import KServeDeploymentType
from utilities.exceptions import FailedPodsError, InferenceResponseError, InvalidStorageArgumentError
from utilities.inference_utils import UserInference
from utilities.infra import (
    get_pods_by_isvc_label,
    wait_for_inference_deployment_replicas,
)

LOGGER = get_logger(name=__name__)


def verify_no_failed_pods(client: DynamicClient, isvc: InferenceService) -> None:
    failed_pods: dict[str, Any] = {}

    for pods in TimeoutSampler(
        wait_timeout=5 * 60,
        sleep=10,
        func=get_pods_by_isvc_label,
        client=client,
        isvc=isvc,
    ):
        if pods:
            if all([pod.instance.status.phase == pod.Status.RUNNING for pod in pods]):
                return

            for pod in pods:
                pod_status = pod.instance.status
                if init_container_status := pod_status.initContainerStatuses:
                    if container_terminated := init_container_status[0].lastState.terminated:
                        if container_terminated.reason == "Error":
                            failed_pods[pod.name] = pod_status

                elif pod_status.phase in (
                    pod.Status.CRASH_LOOPBACK_OFF,
                    pod.Status.FAILED,
                    pod.Status.IMAGE_PULL_BACK_OFF,
                    pod.Status.ERR_IMAGE_PULL,
                ):
                    failed_pods[pod.name] = pod_status

            if failed_pods:
                raise FailedPodsError(pods=failed_pods)


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
    wait_for_predictor_pods: bool = True,
    autoscaler_mode: Optional[str] = None,
    multi_node_worker_spec: Optional[dict[str, int]] = None,
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

    _check_storage_arguments(storage_uri=storage_uri, storage_key=storage_key, storage_path=storage_path)
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

    if autoscaler_mode:
        annotations["serving.kserve.io/autoscalerClass"] = autoscaler_mode

    if multi_node_worker_spec is not None:
        predictor_dict["workerSpec"] = multi_node_worker_spec

    with InferenceService(
        client=client,
        name=name,
        namespace=namespace,
        annotations=annotations,
        predictor=predictor_dict,
        label=labels,
    ) as inference_service:
        if wait_for_predictor_pods:
            verify_no_failed_pods(client=client, isvc=inference_service)
            wait_for_inference_deployment_replicas(
                client=client, isvc=inference_service, deployment_mode=deployment_mode
            )

        if wait:
            inference_service.wait_for_condition(
                condition=inference_service.Condition.READY,
                status=inference_service.Condition.Status.TRUE,
                timeout=15 * 60,
            )

        yield inference_service


def _check_storage_arguments(
    storage_uri: Optional[str],
    storage_key: Optional[str],
    storage_path: Optional[str],
) -> None:
    if (storage_uri and storage_path) or (not storage_uri and not storage_key) or (storage_key and not storage_path):
        raise InvalidStorageArgumentError(storage_uri=storage_uri, storage_key=storage_key, storage_path=storage_path)


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
            expected_response_text_config: Dict[str, Any] = inference.inference_config.get("default_query_model", {})
            use_regex = expected_response_text_config.get("use_regex", False)

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

            if isinstance(expected_response_text, str):
                expected_response_text = Template(expected_response_text).safe_substitute(model_name=model_name)

            elif isinstance(expected_response_text, dict):
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

            elif inference_type == inference.INFER or use_regex:
                formatted_res = json.dumps(res[inference.inference_response_text_key_name]).replace(" ", "")
                if use_regex:
                    assert re.search(expected_response_text, formatted_res)  # type: ignore[arg-type]  # noqa: E501

                else:
                    assert (
                        json.dumps(res[inference.inference_response_key_name]).replace(" ", "")
                        == expected_response_text
                    )

            else:
                response = res[inference.inference_response_key_name]
                if isinstance(response, list):
                    response = response[0]

                assert response[inference.inference_response_text_key_name] == expected_response_text

        else:
            raise InferenceResponseError(f"Inference response output not found in response. Response: {res}")
