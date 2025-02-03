from __future__ import annotations

import json
import re
from contextlib import contextmanager
from string import Template
from typing import Any, Generator, Optional

from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from utilities.constants import (
    Annotations,
    KServeDeploymentType,
    Labels,
)
from utilities.exceptions import (
    FailedPodsError,
    InferenceResponseError,
    InvalidStorageArgumentError,
)
from utilities.inference_utils import UserInference
from utilities.infra import (
    get_inference_serving_runtime,
    get_pods_by_isvc_label,
    wait_for_inference_deployment_replicas,
)
from utilities.jira import is_jira_open

LOGGER = get_logger(name=__name__)


def verify_no_failed_pods(client: DynamicClient, isvc: InferenceService, runtime_name: str | None) -> None:
    """
    Verify no failed pods.

    Args:
        client (DynamicClient): DynamicClient object
        isvc (InferenceService): InferenceService object
        runtime_name (str): ServingRuntime name

    Raises:
            FailedPodsError: If any pod is in failed state

    """
    failed_pods: dict[str, Any] = {}

    LOGGER.info("Verifying no failed pods")
    for pods in TimeoutSampler(
        wait_timeout=5 * 60,
        sleep=10,
        func=get_pods_by_isvc_label,
        client=client,
        isvc=isvc,
        runtime_name=runtime_name,
    ):
        if pods:
            if all([pod.instance.status.phase == pod.Status.RUNNING for pod in pods]):
                return

            for pod in pods:
                pod_status = pod.instance.status
                if pod_status.containerStatuses:
                    for container_status in pod_status.containerStatuses:
                        if (state := container_status.state.waiting) and state.reason == pod.Status.IMAGE_PULL_BACK_OFF:
                            failed_pods[pod.name] = pod_status

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
    """
    Create InferenceService object.

    Args:
        client (DynamicClient): DynamicClient object
        name (str): InferenceService name
        namespace (str): Namespace name
        deployment_mode (str): Deployment mode
        model_format (str): Model format
        runtime (str): ServingRuntime name
        storage_uri (str): Storage URI
        storage_key (str): Storage key
        storage_path (str): Storage path
        wait (bool): Wait for InferenceService to be ready
        enable_auth (bool): Enable authentication
        external_route (bool): External route
        model_service_account (str): Model service account
        min_replicas (int): Minimum replicas
        argument (list[str]): Argument
        resources (dict[str, Any]): Resources
        volumes (dict[str, Any]): Volumes
        volumes_mounts (dict[str, Any]): Volumes mounts
        model_version (str): Model version
        wait_for_predictor_pods (bool): Wait for predictor pods
        autoscaler_mode (str): Autoscaler mode
        multi_node_worker_spec (dict[str, int]): Multi node worker spec
        wait_for_predictor_pods (bool): Wait for predictor pods

    Yields:
        InferenceService: InferenceService object

    """
    labels: dict[str, str] = {}
    predictor_dict: dict[str, Any] = {
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

    annotations = {Annotations.KserveIo.DEPLOYMENT_MODE: deployment_mode}

    if deployment_mode == KServeDeploymentType.SERVERLESS:
        annotations.update({
            "serving.knative.openshift.io/enablePassthrough": "true",
            "sidecar.istio.io/inject": "true",
            "sidecar.istio.io/rewriteAppHTTPProbers": "true",
        })

    if enable_auth:
        # model mesh auth is set in servingruntime
        if deployment_mode == KServeDeploymentType.SERVERLESS:
            annotations[Annotations.KserveAuth.SECURITY] = "true"
        elif deployment_mode == KServeDeploymentType.RAW_DEPLOYMENT:
            labels[Labels.KserveAuth.SECURITY] = "true"

    # default to True if deployment_mode is Serverless (default behavior of Serverless) if was not provided by the user
    # model mesh external route is set in servingruntime
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
            verify_no_failed_pods(client=client, isvc=inference_service, runtime_name=runtime)
            wait_for_inference_deployment_replicas(client=client, isvc=inference_service, runtime_name=runtime)

        if wait:
            # Modelmesh 2nd server in the ns will fail to be Ready; isvc needs to be re-applied
            if is_jira_open(jira_id="RHOAIENG-13636") and deployment_mode == KServeDeploymentType.MODEL_MESH:
                for isvc in InferenceService.get(dyn_client=client, namespace=namespace):
                    _runtime = get_inference_serving_runtime(isvc=isvc)
                    isvc_annotations = isvc.instance.metadata.annotations
                    if (
                        _runtime.name != runtime
                        and isvc_annotations
                        and isvc_annotations.get(Annotations.KserveIo.DEPLOYMENT_MODE)
                        == KServeDeploymentType.MODEL_MESH
                    ):
                        LOGGER.warning(
                            "Bug RHOAIENG-13636 - re-creating isvc if there's already a modelmesh isvc in the namespace"
                        )
                        inference_service.clean_up()
                        inference_service.deploy()

                        break

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
    """
    Check if storage_uri, storage_key and storage_path are valid.

    Args:
        storage_uri (str): URI of the storage.
        storage_key (str): Key of the storage.
        storage_path (str): Path of the storage.

    Raises:
        InvalidStorageArgumentError: If storage_uri, storage_key and storage_path are not valid.
    """
    if (storage_uri and storage_path) or (not storage_uri and not storage_key) or (storage_key and not storage_path):
        raise InvalidStorageArgumentError(storage_uri=storage_uri, storage_key=storage_key, storage_path=storage_path)


def verify_inference_response(
    inference_service: InferenceService,
    inference_config: dict[str, Any],
    inference_type: str,
    protocol: str,
    model_name: Optional[str] = None,
    inference_input: Optional[Any] = None,
    use_default_query: bool = False,
    expected_response_text: Optional[str] = None,
    insecure: bool = False,
    token: Optional[str] = None,
    authorized_user: Optional[bool] = None,
) -> None:
    """
    Verify the inference response.

    Args:
        inference_service (InferenceService): Inference service.
        inference_config (dict[str, Any]): Inference config.
        inference_type (str): Inference type.
        protocol (str): Protocol.
        model_name (str): Model name.
        inference_input (Any): Inference input.
        use_default_query (bool): Use default query or not.
        expected_response_text (str): Expected response text.
        insecure (bool): Insecure mode.
        token (str): Token.
        authorized_user (bool): Authorized user.

    Raises:
        InvalidInferenceResponseError: If inference response is invalid.
        ValidationError: If inference response is invalid.

    """
    model_name = model_name or inference_service.name

    inference = UserInference(
        inference_service=inference_service,
        inference_config=inference_config,
        inference_type=inference_type,
        protocol=protocol,
    )

    res = inference.run_inference_flow(
        model_name=model_name,
        inference_input=inference_input,
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
            expected_response_text_config: dict[str, Any] = inference.inference_config.get("default_query_model", {})
            use_regex = expected_response_text_config.get("use_regex", False)

            if not expected_response_text_config:
                raise ValueError(
                    f"Missing default_query_model config for inference {inference_config}. "
                    f"Config: {expected_response_text_config}"
                )

            if inference.inference_config.get("support_multi_default_queries"):
                query_config = expected_response_text_config.get(inference_type)
                if not query_config:
                    raise ValueError(
                        f"Missing default_query_model config for inference {inference_config}. "
                        f"Config: {expected_response_text_config}"
                    )
                expected_response_text = query_config.get("query_output", "")
                use_regex = query_config.get("use_regex", False)

            else:
                expected_response_text = expected_response_text_config.get("query_output")

            if not expected_response_text:
                raise ValueError(f"Missing response text key for inference {inference_config}")

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
