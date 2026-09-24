import json
import re
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from contextlib import contextmanager
from string import Template
from typing import Any

import pytest
import structlog
from kubernetes.client.exceptions import ApiException
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.inference_graph import InferenceGraph
from ocp_resources.inference_service import InferenceService
from ocp_resources.node import Node
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.utils.constants import DEFAULT_CLUSTER_RETRY_EXCEPTIONS
from timeout_sampler import TimeoutExpiredError, TimeoutSampler, TimeoutWatch
from urllib3.exceptions import HTTPError

from tests.model_serving.model_server.kserve.autoscaling.keda.utils import get_isvc_keda_scaledobject
from utilities.constants import ApiGroups, KServeDeploymentType, Protocols, Timeout
from utilities.exceptions import (
    InferenceResponseError,
)
from utilities.inference_utils import Inference, UserInference
from utilities.infra import get_pods_by_isvc_label
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG
from utilities.resources.llm_inference_service import LLMInferenceService

LOGGER = structlog.get_logger(name=__name__)

# ---------------------------------------------------------------------------
# ConnectionsAPI annotation keys (shared by the ISVC and LLMISVC ConnectionsAPI
# coverage under kserve/storage/ and llmd/, respectively)
# ---------------------------------------------------------------------------
CONNECTIONS_ANNOTATION: str = f"{ApiGroups.OPENDATAHUB_IO}/connections"
CONNECTION_PATH_ANNOTATION: str = f"{ApiGroups.OPENDATAHUB_IO}/connection-path"
CONNECTION_TYPE_PROTOCOL_ANNOTATION: str = f"{ApiGroups.OPENDATAHUB_IO}/connection-type-protocol"


def skip_test(reason: str) -> None:
    """Log a visible skip banner and call pytest.skip."""
    border = "=" * 60
    LOGGER.warning("\n".join(["", border, f"  SKIP — {reason}", border, ""]))
    pytest.skip(reason)


def add_connection_annotations(
    resource: InferenceService | LLMInferenceService, connections: str, connection_path: str | None = None
) -> None:
    """Patch a resource's metadata to add ConnectionsAPI annotations, exercising the UPDATE path.

    Args:
        resource: InferenceService or LLMInferenceService to patch.
        connections: Value for the `opendatahub.io/connections` annotation.
        connection_path: Optional value for the `opendatahub.io/connection-path` annotation
            (S3 sub-path).
    """
    annotations: dict[str, str] = {CONNECTIONS_ANNOTATION: connections}
    if connection_path:
        annotations[CONNECTION_PATH_ANNOTATION] = connection_path
    resource.update(resource_dict={"metadata": {"name": resource.name, "annotations": annotations}})


def remove_connection_annotations(resource: InferenceService | LLMInferenceService) -> None:
    """Patch a resource's metadata to null out ConnectionsAPI annotations, exercising UPDATE-remove.

    Args:
        resource: InferenceService or LLMInferenceService to patch.
    """
    resource.update(
        resource_dict={
            "metadata": {
                "name": resource.name,
                "annotations": {CONNECTIONS_ANNOTATION: None, CONNECTION_PATH_ANNOTATION: None},
            }
        }
    )


# ---------------------------------------------------------------------------
# ConnectionsAPI — connection Secret factories (shared by ISVC and LLMISVC suites)
# ---------------------------------------------------------------------------
@contextmanager
def create_uri_connection_secret(
    client: DynamicClient,
    name: str,
    namespace: str,
    uri: str,
    teardown: bool = True,
) -> Generator[Secret, Any, Any]:
    """Create a `uri`-typed ConnectionsAPI Secret pointing at a `hf://` model reference.

    The odh-model-controller ConnectionsAPI webhook reads the secret's `URI` key (via the
    `opendatahub.io/connection-type-protocol: uri` annotation) and injects it into the consuming
    InferenceService's `predictor.model.storageUri` or LLMInferenceService's `spec.model.uri` on
    CREATE/UPDATE.

    Args:
        client: Kubernetes dynamic client.
        name: Name of the Secret to create.
        namespace: Namespace to create the Secret in.
        uri: Model reference to store under the `URI` key (e.g. `hf://org/model`).
        teardown: Whether to delete the Secret on context exit.

    Yields:
        Secret: The created connection Secret.
    """
    with Secret(
        client=client,
        name=name,
        namespace=namespace,
        annotations={CONNECTION_TYPE_PROTOCOL_ANNOTATION: "uri"},
        string_data={"URI": uri},
        teardown=teardown,
    ) as secret:
        yield secret


@contextmanager
def create_oci_connection_secret(
    client: DynamicClient,
    name: str,
    namespace: str,
    docker_config_json: str = "{}",
    teardown: bool = True,
) -> Generator[Secret, Any, Any]:
    """Create an `oci`-typed ConnectionsAPI Secret (dockerconfigjson) for OCI modelcar pulls.

    Defaults to an empty docker config, which is sufficient for the public quay.io modelcar images
    reused by the ISVC/LLMISVC ConnectionsAPI suites — the webhook still injects `imagePullSecrets`
    referencing this Secret regardless of its credential content.

    Args:
        client: Kubernetes dynamic client.
        name: Name of the Secret to create.
        namespace: Namespace to create the Secret in.
        docker_config_json: Raw `.dockerconfigjson` payload. Defaults to an empty JSON object.
        teardown: Whether to delete the Secret on context exit.

    Yields:
        Secret: The created connection Secret.
    """
    with Secret(
        client=client,
        name=name,
        namespace=namespace,
        type="kubernetes.io/dockerconfigjson",
        annotations={CONNECTION_TYPE_PROTOCOL_ANNOTATION: "oci"},
        string_data={".dockerconfigjson": docker_config_json},
        teardown=teardown,
    ) as secret:
        yield secret


# ---------------------------------------------------------------------------
# ConnectionsAPI — ServiceAccount assertion (shared by ISVC and LLMISVC suites)
# ---------------------------------------------------------------------------
def assert_service_account_exists(
    client: DynamicClient, namespace: str, name: str, timeout: int = Timeout.TIMEOUT_1MIN
) -> None:
    """Assert that the `{secret}-sa` ServiceAccount created by the S3 injection path exists.

    Polls rather than checking once, since callers may invoke this immediately after a CREATE
    with no readiness wait (the smoke-test path), and the ServiceAccount is an asynchronous side
    effect of the ConnectionsAPI webhook/controller rather than part of the admission response.

    Args:
        client: Kubernetes dynamic client.
        namespace: Namespace the ServiceAccount is expected in.
        name: Expected ServiceAccount name (`{secret_name}-sa`).
        timeout: Seconds to wait for the ServiceAccount to appear.

    Raises:
        AssertionError: If the ServiceAccount does not exist within `timeout` seconds.
    """
    service_account = ServiceAccount(client=client, namespace=namespace, name=name)
    try:
        for exists in TimeoutSampler(wait_timeout=timeout, sleep=2, func=lambda: service_account.exists):
            if exists:
                return
    except TimeoutExpiredError:
        pass

    raise AssertionError(
        f"Expected ServiceAccount {name!r} to exist in namespace {namespace!r} "
        f"(S3 ConnectionsAPI injection must create it as a side effect) within {timeout}s"
    )


# ---------------------------------------------------------------------------
# ConnectionsAPI — generic cleared-predicate poller (shared by ISVC and LLMISVC suites)
# ---------------------------------------------------------------------------
def wait_for_cleared_predicate(predicate: Callable[[], bool], timeout: int, resource_label: str) -> None:
    """Poll `predicate` until connection fields are cleared, raising a resource-specific error.

    Shared by `wait_for_isvc_connection_cleared` and `wait_for_llmisvc_connection_cleared`, which
    differ only in `predicate` (which spec fields they check) and how they identify the resource
    in the error message.

    Args:
        predicate: Zero-arg callable returning `True` once the connection fields are cleared.
        timeout: Seconds to wait before giving up.
        resource_label: Human-readable resource identifier (e.g. `f"InferenceService {isvc.name}"`)
            used in the raised error message.

    Raises:
        TimeoutError: If `predicate` never returns `True` within `timeout` seconds.
    """
    try:
        for sample in TimeoutSampler(wait_timeout=timeout, sleep=5, func=predicate):
            if sample:
                return
    except TimeoutExpiredError as exc:
        raise TimeoutError(f"Connection fields on {resource_label} were not cleared within {timeout}s") from exc


def get_worker_architecture(client: DynamicClient) -> str | None:
    """Return the architecture shared by all workers, or ``None`` if unavailable or mixed."""
    architectures: set[str] = set()
    try:
        for node in Node.get(client=client, label_selector="node-role.kubernetes.io/worker"):
            architecture = node.instance.status.nodeInfo.architecture
            LOGGER.info(f"Detected worker node architecture: {architecture!r}")
            if not isinstance(architecture, str) or not architecture.strip():
                LOGGER.warning(f"Unable to read worker architecture: {architecture!r}")
                return None
            architectures.add(architecture)
    except (ApiException, HTTPError, ResourceNotFoundError, AttributeError) as error:
        LOGGER.warning(f"Unable to read worker architecture: {error}")
        return None

    if not architectures:
        LOGGER.warning("Unable to read worker architecture: no worker nodes found")
    elif len(architectures) > 1:
        LOGGER.warning(f"Unable to read worker architecture: mixed architectures {architectures!r}")
    else:
        return architectures.pop()

    return None


def is_arm64_cluster(client: DynamicClient) -> bool:
    """Return whether every worker node reports the ``arm64`` architecture."""
    architecture = get_worker_architecture(client=client)
    if architecture not in {None, "arm64", "amd64", "ppc64le", "s390x"}:
        LOGGER.warning(f"Unknown worker architecture: {architecture!r}")

    return architecture == "arm64"


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
    inference_timeout: int | None = None,
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
        inference_timeout (int | None): Retry timeout in seconds for the inference request.

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
        inference_timeout=inference_timeout,
    )

    if authorized_user is False:
        auth_header = "x-ext-auth-reason"

        if isinstance(res["output"], dict):
            # Response body was parsed to JSON (e.g. FastAPI HTTPException).
            # Reconstruct full response text from parsed headers so existing
            # status-line / header checks work unchanged.
            output = "\n".join(f"{k}: {v}" for k, v in res.items() if k != "output" and isinstance(v, str))
            output += "\n" + json.dumps(res["output"])
        else:
            output = res["output"]

        if auth_reason := re.search(rf"{auth_header}: (.*)", output, re.MULTILINE):
            reason = auth_reason.group(1).lower()

            if token:
                assert re.search(r"not (?:authenticated|authorized)", reason)

            else:
                assert "credential not found" in reason

        elif (
            isinstance(inference_service, InferenceGraph)
            and inference.deployment_mode in KServeDeploymentType.RAW_DEPLOYMENT_MODES
        ):
            assert "x-forbidden-reason: Access to the InferenceGraph is not allowed" in output

        elif "403 Forbidden" in output:
            resource = f"{inference_service.kind.lower()}s"
            assert re.search(rf"Forbidden \(user=.*verb=get.*resource={resource}", output)

        elif "401 Unauthorized" in output:
            # HTTP status line carries 401 — correctly rejected.
            pass

        else:
            raise ValueError(f"Auth header {auth_header} not found in response. Response: {output}")

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


def wait_for_raw_isvc_https_infer_ready(
    isvc: InferenceService,
    *,
    token: str | None = None,
    timeout: int = 300,
    sleep: int = 5,
) -> None:
    """Block until the same external HTTPS REST infer the suite uses succeeds.

    ``InferenceService`` Ready and Deployment replica counts do not imply the OpenShift
    router has reprogrammed backends for the predictor ``Service`` after auth or pod
    template changes. Waiting on **HTTP 200** and a **JSON** body on the real infer URL
    (same path as ``verify_inference_response``) is the correct readiness condition.

    Args:
        isvc: Exposed raw KServe InferenceService.
        token: Bearer token when auth is required; omit when auth is disabled.
        timeout: Maximum seconds to poll.
        sleep: Seconds between attempts.

    Raises:
        TimeoutExpiredError: If the infer path does not succeed in time.
    """

    def _https_infer_ok() -> bool:
        inference = UserInference(
            inference_service=isvc,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
        )
        try:
            out = inference.run_inference(
                model_name=isvc.name,
                use_default_query=True,
                token=token,
            )
        except ValueError, InferenceResponseError:
            return False

        if not out or not out.strip():
            return False

        status_line = out.splitlines()[0].lower()
        if not re.search(r"http/1\.\d\s+200\b", status_line):
            return False
        return "content-type: application/json" in out.lower()

    try:
        for ok in TimeoutSampler(wait_timeout=timeout, sleep=sleep, func=_https_infer_ok):
            if ok:
                LOGGER.info(f"Raw ISVC {isvc.name} external HTTPS infer ready (auth token={'yes' if token else 'no'})")
                return
    except TimeoutExpiredError:
        LOGGER.error(
            f"Timeout: InferenceService {isvc.name} in {isvc.namespace} external HTTPS infer not ready "
            f"within {timeout}s (token={'yes' if token else 'no'})"
        )
        raise


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
        timeout=300,
        sleep=10,
    ):
        if pods and len(pods) == final_pod_count:
            return
    raise AssertionError(f"Timed out waiting for {final_pod_count} pods. Current pod count: {len(pods) if pods else 0}")


def verify_no_inference_pods(client: DynamicClient, isvc: InferenceService, wait_timeout: int = 240) -> bool:
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
