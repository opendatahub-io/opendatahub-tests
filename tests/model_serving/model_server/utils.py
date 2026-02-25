import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from string import Template
from typing import Any

from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.inference_graph import InferenceGraph
from ocp_resources.inference_service import InferenceService
from ocp_resources.utils.constants import DEFAULT_CLUSTER_RETRY_EXCEPTIONS
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutExpiredError, TimeoutSampler, TimeoutWatch

from tests.model_serving.model_server.kserve.keda.utils import get_isvc_keda_scaledobject
from utilities.constants import KServeDeploymentType, Protocols, Timeout
from utilities.exceptions import (
    InferenceResponseError,
)
from utilities.inference_utils import UserInference
from utilities.infra import get_pods_by_isvc_label

LOGGER = get_logger(name=__name__)


def verify_inference_response(
    inference_service: InferenceService | InferenceGraph,
    inference_config: dict[str, Any],
    inference_type: str,
    protocol: str,
    model_name: str | None = None,
    inference_input: Any | None = None,
    use_default_query: bool = False,
    expected_response_text: str | None = None,
    insecure: bool = False,
    token: str | None = None,
    authorized_user: bool | None = None,
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

        elif (
            isinstance(inference_service, InferenceGraph)
            and inference.deployment_mode == KServeDeploymentType.RAW_DEPLOYMENT
        ):
            assert "x-forbidden-reason: Access to the InferenceGraph is not allowed" in res["output"]

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
                    assert "".join(output) == expected_response_text, (
                        f"Expected: {expected_response_text} does not match response: {output}"
                    )

            elif inference_type == inference.INFER or use_regex:
                formatted_res = json.dumps(res[inference.inference_response_text_key_name]).replace(" ", "")
                if use_regex:
                    assert re.search(expected_response_text, formatted_res), (  # type: ignore[arg-type]
                        f"Expected: {expected_response_text} not found in: {formatted_res}"
                    )

                else:
                    formatted_res = json.dumps(res[inference.inference_response_key_name]).replace(" ", "")
                    assert formatted_res == expected_response_text, (
                        f"Expected: {expected_response_text} does not match output: {formatted_res}"
                    )

            else:
                response = res[inference.inference_response_key_name]
                if isinstance(response, list):
                    response = response[0]

                if isinstance(response, dict):
                    response_text = response[inference.inference_response_text_key_name]
                    assert response_text == expected_response_text, (
                        f"Expected: {expected_response_text} does not match response: {response_text}"
                    )

                else:
                    raise InferenceResponseError(
                        "Inference response output does not match expected output format."
                        f"Expected: {expected_response_text}.\nResponse: {res}"
                    )

        else:
            raise InferenceResponseError(f"Inference response output not found in response. Response: {res}")


def run_inference_multiple_times(
    isvc: InferenceService,
    inference_config: dict[str, Any],
    inference_type: str,
    protocol: str,
    iterations: int,
    model_name: str | None = None,
    run_in_parallel: bool = False,
) -> None:
    """
    Run inference multiple times.

    Args:
        isvc (InferenceService): Inference service.
        inference_config (dict[str, Any]): Inference config.
        inference_type (str): Inference type.
        protocol (str): Protocol.
        model_name (str): Model name.
        iterations (int): Number of iterations.
        run_in_parallel (bool, optional): Run inference in parallel.

    """
    futures = []

    with ThreadPoolExecutor() as executor:
        for iteration in range(iterations):
            infer_kwargs = {
                "inference_service": isvc,
                "inference_config": inference_config,
                "inference_type": inference_type,
                "protocol": protocol,
                "model_name": model_name,
                "use_default_query": True,
            }

            if run_in_parallel:
                futures.append(executor.submit(verify_inference_response, **infer_kwargs))
            else:
                verify_inference_response(**infer_kwargs)

        if futures:
            exceptions = [_exception for result in as_completed(futures) if (_exception := result.exception())]

            if exceptions:
                raise InferenceResponseError(f"Failed to run inference. Error: {exceptions}")


def verify_keda_scaledobject(
    client: DynamicClient,
    isvc: InferenceService,
    expected_trigger_type: str | None = None,
    expected_query: str | None = None,
    expected_threshold: str | None = None,
) -> None:
    """
    Verify the KEDA ScaledObject.

    Args:
        client: DynamicClient instance
        isvc: InferenceService instance
        expected_trigger_type: Expected trigger type
        expected_query: Expected query string
        expected_threshold: Expected threshold as string (e.g. "50.000000")
    """
    scaled_object = get_isvc_keda_scaledobject(client=client, isvc=isvc)
    trigger_meta = scaled_object.instance.spec.triggers[0].metadata
    trigger_type = scaled_object.instance.spec.triggers[0].type
    query = trigger_meta.get("query")
    threshold = trigger_meta.get("threshold")

    assert trigger_type == expected_trigger_type, (
        f"Trigger type {trigger_type} does not match expected {expected_trigger_type}"
    )
    assert query == expected_query, f"Query {query} does not match expected {expected_query}"
    assert int(float(threshold)) == int(float(expected_threshold)), (
        f"Threshold {threshold} does not match expected {expected_threshold}"
    )


def run_concurrent_load_for_keda_scaling(
    isvc: InferenceService,
    inference_config: dict[str, Any],
    num_concurrent: int = 5,
    duration: int = 120,
) -> None:
    """
    Run a concurrent load to test the keda scaling functionality.

    Args:
        isvc: InferenceService instance
        inference_config: Inference config
        num_concurrent: Number of concurrent requests
        duration: Duration in seconds to run the load test
    """

    def _make_request() -> None:
        verify_inference_response(
            inference_service=isvc,
            inference_config=inference_config,
            inference_type="completions",
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    timeout_watch = TimeoutWatch(timeout=duration)
    with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        while timeout_watch.remaining_time() > 0:
            futures = [executor.submit(_make_request) for _ in range(num_concurrent)]
            wait(fs=futures)


def inference_service_pods_sampler(
    client: DynamicClient, isvc: InferenceService, timeout: int, sleep: int = 1
) -> TimeoutSampler:
    """
    Returns TimeoutSampler for inference service.

    Args:
        client (DynamicClient): DynamicClient object
        isvc (InferenceService): InferenceService object
        timeout (int): Timeout in seconds
        sleep (int): Sleep time in seconds

    Returns:
        TimeoutSampler: TimeoutSampler object

    """
    return TimeoutSampler(
        wait_timeout=timeout,
        sleep=sleep,
        func=get_pods_by_isvc_label,
        client=client,
        isvc=isvc,
    )


def verify_final_pod_count(unprivileged_client: DynamicClient, isvc: InferenceService, final_pod_count: int):
    """Verify final pod count after running load tests for KEDA scaling."""

    for pods in inference_service_pods_sampler(
        client=unprivileged_client,
        isvc=isvc,
        timeout=Timeout.TIMEOUT_5MIN,
        sleep=10,
    ):
        if pods and len(pods) == final_pod_count:
            return
    raise AssertionError(f"Timed out waiting for {final_pod_count} pods. Current pod count: {len(pods) if pods else 0}")


def verify_no_inference_pods(
    client: DynamicClient, isvc: InferenceService, wait_timeout: int = Timeout.TIMEOUT_4MIN
) -> bool:
    """
    Verify that no inference pods are running for the given InferenceService.

    Args:
        client (DynamicClient): DynamicClient object
        isvc (InferenceService): InferenceService object
        wait_timeout (int): Timeout in seconds, default is 4 minutes

    Returns:
        bool: True if no pods are running, False otherwise
    Raises:
        TimeoutError: If pods exist after the timeout.

    """
    pods = []

    try:
        for pods in TimeoutSampler(
            wait_timeout=wait_timeout,
            sleep=5,
            exceptions_dict=DEFAULT_CLUSTER_RETRY_EXCEPTIONS,
            func=get_pods_by_isvc_label,
            client=client,
            isvc=isvc,
        ):
            if not pods:
                return True

    except TimeoutExpiredError as e:
        if isinstance(e.last_exp, ResourceNotFoundError):
            return True
        LOGGER.error(f"{[pod.name for pod in pods]} were not deleted")
        return False
    return True
