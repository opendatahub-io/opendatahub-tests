"""Helpers for the ConnectionsAPI end-to-end test suite.

Provides:
    - Connection-Secret factories (URI, OCI) that complement
      ``utilities.infra.s3_endpoint_secret`` for the S3 case.
    - ``create_connection_llmisvc``: an LLMInferenceService factory that reuses the CPU vLLM
      template (image/env/resources/probes) from ``CpuConfig`` so LLMISVC connection tests never
      need a GPU.
    - Annotation-patch helpers (``add_connection_annotations`` / ``remove_connection_annotations``)
      used by the UPDATE-inject/remove tests, shared across InferenceService and
      LLMInferenceService.
    - Injection assertion helpers for both InferenceService and LLMInferenceService
      (``assert_isvc_s3_fully_injected`` / ``assert_llmisvc_s3_fully_injected`` also assert the
      `{secret}-sa` ServiceAccount side effect, since S3 injection always produces both), and
      small polling helpers for the UPDATE-remove cleanup path.
    - A webhook-configuration guard used to fail the suite fast if the odh-model-controller
      ConnectionsAPI webhooks are not the ones actually wired up on the cluster.
    - Thin wrappers around existing inference helpers (MLServer v2 REST, LLMISVC chat
      completions) used to prove a connection-injected model actually loaded.
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.mutating_webhook_config import MutatingWebhookConfiguration
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_serving.model_runtime.mlserver.constant import ONNX_REST_INPUT_QUERY, SKLEARN_REST_INPUT_QUERY
from tests.model_serving.model_runtime.mlserver.utils import run_mlserver_inference
from tests.model_serving.model_server.connections_api.constants import (
    CONNECTION_PATH_ANNOTATION,
    CONNECTION_TYPE_PROTOCOL_ANNOTATION,
    CONNECTIONS_ANNOTATION,
    ISVC_CONNECTIONS_WEBHOOK,
    LLMISVC_CHAT_PROMPT,
    LLMISVC_CONNECTIONS_WEBHOOK,
    LLMISVC_PLACEHOLDER_MODEL_URI,
    STALE_ISVC_CONNECTIONS_WEBHOOK,
    STALE_LLMISVC_CONNECTIONS_WEBHOOK,
)
from tests.model_serving.model_server.llmd.llmd_configs.config_base import CpuConfig
from tests.model_serving.model_server.llmd.utils import (
    parse_completion_text,
    send_chat_completions,
    wait_for_llmisvc,
    wait_for_llmisvc_pods_ready,
    workaround_503_no_healthy_upstream,
)
from utilities.constants import ModelFormat, Protocols, Timeout
from utilities.resources.llm_inference_service import LLMInferenceService

LOGGER = structlog.get_logger(name=__name__)


# ---------------------------------------------------------------------------
# Connection Secret factories
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
    ISVC/LLMISVC's storage URI field on CREATE/UPDATE.

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

    Defaults to an empty docker config, which is sufficient for the public quay.io modelcar
    images reused by this suite (`SharedImages.MLSERVER_SKLEARN`, `SharedImages.OCI_TINYLLAMA`) —
    the webhook still injects `imagePullSecrets` referencing this Secret regardless of its
    credential content.

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
# LLMInferenceService factory (CPU vLLM, no GPU)
# ---------------------------------------------------------------------------
def _cpu_vllm_template() -> dict[str, Any]:
    """Build the vLLM-CPU container template reused by every ConnectionsAPI LLMISVC test.

    Factors the image, env vars, resources, and probes out of `CpuConfig`
    (`tests/model_serving/model_server/llmd/llmd_configs/config_base.py`) instead of duplicating
    them, so this suite always tracks the same vLLM-CPU settings as the llmd CPU test suites.

    Returns:
        A `spec.template` dict with a single `main` container, suitable for `LLMInferenceService`.
    """
    main_container: dict[str, Any] = {
        "name": "main",
        "image": CpuConfig.container_image,
        "resources": CpuConfig.container_resources(),
        "env": CpuConfig.container_env(),
        "startupProbe": CpuConfig.startup_probe(),
        "livenessProbe": CpuConfig.liveness_probe(),
        "readinessProbe": CpuConfig.readiness_probe(),
    }
    return {"containers": [main_container]}


@contextmanager
def create_connection_llmisvc(
    client: DynamicClient,
    name: str,
    namespace: str,
    connections: str,
    connection_path: str | None = None,
    model_uri: str = LLMISVC_PLACEHOLDER_MODEL_URI,
    wait: bool = True,
    wait_for_pods: bool = True,
    teardown: bool = True,
    timeout: int = Timeout.TIMEOUT_10MIN,
) -> Generator[LLMInferenceService, Any, Any]:
    """Create an `LLMInferenceService` annotated for ConnectionsAPI injection, on CPU vLLM.

    Seeds `spec.model.uri` with a placeholder (or an OCI reference for the OCI case) that the
    ConnectionsAPI webhook is expected to overwrite/complete from the referenced connection
    Secret. Reuses the CPU vLLM-TinyLlama template so no GPU node is required. Disables the
    platform's inference-gateway auth policy (`security.opendatahub.io/enable-auth: "false"`) —
    this suite tests connection injection, not authentication, so unauthenticated inference keeps
    the functional-inference check simple.

    Args:
        client: Kubernetes dynamic client.
        name: LLMInferenceService name.
        namespace: Namespace to create the LLMInferenceService in.
        connections: Name of the connection Secret to reference via
            `opendatahub.io/connections`.
        connection_path: Optional value for `opendatahub.io/connection-path` (S3 sub-path).
        model_uri: Seed value for `spec.model.uri`; overwritten by the webhook for S3.
        wait: Wait for the `Ready` condition after creation.
        wait_for_pods: Also wait for the workload/router pods to be `Ready` (only used when
            `wait=True`).
        teardown: Whether to delete the LLMInferenceService on context exit.
        timeout: Seconds to wait for `Ready` when `wait=True`.

    Yields:
        LLMInferenceService: The created LLMInferenceService.
    """
    annotations = {
        CONNECTIONS_ANNOTATION: connections,
        "security.opendatahub.io/enable-auth": "false",
    }
    if connection_path:
        annotations[CONNECTION_PATH_ANNOTATION] = connection_path

    with LLMInferenceService(
        client=client,
        name=name,
        namespace=namespace,
        annotations=annotations,
        replicas=1,
        model={"uri": model_uri},
        router=CpuConfig.router_config(),
        template=_cpu_vllm_template(),
        teardown=teardown,
    ) as llmisvc:
        if wait:
            wait_for_llmisvc(llmisvc=llmisvc, timeout=timeout)
            if wait_for_pods:
                wait_for_llmisvc_pods_ready(client=client, llmisvc=llmisvc)
        yield llmisvc


# ---------------------------------------------------------------------------
# Annotation patch helpers (UPDATE inject/remove)
# ---------------------------------------------------------------------------
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
# ServiceAccount assertion
# ---------------------------------------------------------------------------
def assert_service_account_exists(client: DynamicClient, namespace: str, name: str) -> None:
    """Assert that the `{secret}-sa` ServiceAccount created by the S3 injection path exists.

    Args:
        client: Kubernetes dynamic client.
        namespace: Namespace the ServiceAccount is expected in.
        name: Expected ServiceAccount name (`{secret_name}-sa`).

    Raises:
        AssertionError: If the ServiceAccount does not exist.
    """
    service_account = ServiceAccount(client=client, namespace=namespace, name=name)
    assert service_account.exists, (
        f"Expected ServiceAccount {name!r} to exist in namespace {namespace!r} "
        "(S3 ConnectionsAPI injection must create it as a side effect)"
    )


# ---------------------------------------------------------------------------
# InferenceService injection assertions
# ---------------------------------------------------------------------------
def assert_isvc_s3_injected(isvc: InferenceService, secret_name: str, expected_path: str) -> None:
    """Assert an S3 connection was injected into an InferenceService's predictor spec.

    Args:
        isvc: InferenceService to inspect (re-read via `.instance`).
        secret_name: Name of the S3 connection Secret that should have been injected.
        expected_path: Expected `storage.path` value (from `opendatahub.io/connection-path`).

    Raises:
        AssertionError: If the SA name, storage key, or storage path do not match.
    """
    predictor = isvc.instance.spec.predictor
    sa_name = predictor.get("serviceAccountName")
    assert sa_name == f"{secret_name}-sa", f"Expected predictor.serviceAccountName={secret_name}-sa, got {sa_name!r}"

    storage = predictor.model.get("storage") or {}
    assert storage.get("key") == secret_name, f"Expected predictor.model.storage.key={secret_name!r}, got {storage!r}"
    assert storage.get("path") == expected_path, (
        f"Expected predictor.model.storage.path={expected_path!r}, got {storage!r}"
    )


def assert_isvc_s3_fully_injected(
    client: DynamicClient, isvc: InferenceService, namespace: str, secret_name: str, expected_path: str
) -> None:
    """Assert both effects of S3 injection on an InferenceService: spec fields and SA creation.

    S3 injection is always expected to produce both effects together (the webhook sets the
    predictor spec fields *and* creates the `{secret}-sa` ServiceAccount as a side effect), so
    every S3 CREATE/UPDATE-inject test needs both checks. Combines `assert_isvc_s3_injected` and
    `assert_service_account_exists` so callers only need one call.

    Args:
        client: Kubernetes dynamic client.
        isvc: InferenceService to inspect (re-read via `.instance`).
        namespace: Namespace the `{secret}-sa` ServiceAccount is expected in.
        secret_name: Name of the S3 connection Secret that should have been injected.
        expected_path: Expected `storage.path` value (from `opendatahub.io/connection-path`).

    Raises:
        AssertionError: If the spec fields don't match, or the ServiceAccount doesn't exist.
    """
    assert_isvc_s3_injected(isvc=isvc, secret_name=secret_name, expected_path=expected_path)
    assert_service_account_exists(client=client, namespace=namespace, name=f"{secret_name}-sa")


def assert_isvc_uri_injected(isvc: InferenceService, expected_uri: str) -> None:
    """Assert a `uri` connection was injected as `predictor.model.storageUri`.

    Args:
        isvc: InferenceService to inspect (re-read via `.instance`).
        expected_uri: Expected `storageUri` value (the connection Secret's `URI` key).

    Raises:
        AssertionError: If `storageUri` does not match.
    """
    storage_uri = isvc.instance.spec.predictor.model.get("storageUri")
    assert storage_uri == expected_uri, f"Expected predictor.model.storageUri={expected_uri!r}, got {storage_uri!r}"


def assert_isvc_oci_injected(isvc: InferenceService, secret_name: str) -> None:
    """Assert an `oci` connection was injected as a `predictor.imagePullSecrets` entry.

    Args:
        isvc: InferenceService to inspect (re-read via `.instance`).
        secret_name: Name of the OCI connection Secret expected in `imagePullSecrets`.

    Raises:
        AssertionError: If the secret name is not present in `imagePullSecrets`.
    """
    pull_secrets = isvc.instance.spec.predictor.get("imagePullSecrets") or []
    names = [dict(entry).get("name") for entry in pull_secrets]
    assert secret_name in names, f"Expected {secret_name!r} in predictor.imagePullSecrets, got {names}"


def assert_isvc_connection_cleared(isvc: InferenceService) -> None:
    """Assert an ISVC's S3 injection fields were cleared by an UPDATE-remove action.

    Args:
        isvc: InferenceService to inspect (re-read via `.instance`).

    Raises:
        AssertionError: If `serviceAccountName` or `model.storage` are still populated.
    """
    predictor = isvc.instance.spec.predictor
    sa_name = predictor.get("serviceAccountName")
    assert not sa_name, f"Expected predictor.serviceAccountName to be cleared, got {sa_name!r}"

    storage = predictor.model.get("storage")
    assert not storage, f"Expected predictor.model.storage to be cleared, got {storage!r}"


def wait_for_isvc_connection_cleared(isvc: InferenceService, timeout: int = Timeout.TIMEOUT_2MIN) -> None:
    """Poll until an ISVC's connection fields are cleared after an UPDATE-remove action.

    The webhook's cleanup runs asynchronously relative to the annotation patch, so this polls
    instead of asserting immediately after `.update()`.

    Args:
        isvc: InferenceService to poll (re-read via `.instance` on every sample).
        timeout: Seconds to wait before giving up.

    Raises:
        TimeoutError: If the fields are not cleared within `timeout` seconds.
    """

    def _cleared() -> bool:
        predictor = isvc.instance.spec.predictor
        return not predictor.get("serviceAccountName") and not predictor.model.get("storage")

    _wait_for_cleared(predicate=_cleared, timeout=timeout, resource_label=f"InferenceService {isvc.name}")


# ---------------------------------------------------------------------------
# LLMInferenceService injection assertions
# ---------------------------------------------------------------------------
def assert_llmisvc_s3_injected(llmisvc: LLMInferenceService, secret_name: str, expected_uri: str) -> None:
    """Assert an S3 connection was injected into an LLMInferenceService.

    Args:
        llmisvc: LLMInferenceService to inspect (re-read via `.instance`).
        secret_name: Name of the S3 connection Secret that should have been injected.
        expected_uri: Expected `spec.model.uri` value (`s3://{bucket}/{path}`).

    Raises:
        AssertionError: If the SA name or model URI do not match.
    """
    sa_name = llmisvc.instance.spec.template.get("serviceAccountName")
    assert sa_name == f"{secret_name}-sa", f"Expected template.serviceAccountName={secret_name}-sa, got {sa_name!r}"

    model_uri = llmisvc.instance.spec.model.get("uri")
    assert model_uri == expected_uri, f"Expected spec.model.uri={expected_uri!r}, got {model_uri!r}"


def assert_llmisvc_s3_fully_injected(
    client: DynamicClient,
    llmisvc: LLMInferenceService,
    namespace: str,
    secret_name: str,
    bucket: str,
    path: str,
) -> None:
    """Assert both effects of S3 injection on an LLMInferenceService: spec fields and SA creation.

    Computes the expected `s3://{bucket}/{path}` URI once, then combines `assert_llmisvc_s3_injected`
    and `assert_service_account_exists` so callers only need one call and never duplicate the URI
    computation.

    Args:
        client: Kubernetes dynamic client.
        llmisvc: LLMInferenceService to inspect (re-read via `.instance`).
        namespace: Namespace the `{secret}-sa` ServiceAccount is expected in.
        secret_name: Name of the S3 connection Secret that should have been injected.
        bucket: S3 bucket name backing the connection Secret.
        path: S3 sub-path (`opendatahub.io/connection-path`) the model is stored under.

    Raises:
        AssertionError: If the spec fields don't match, or the ServiceAccount doesn't exist.
    """
    expected_uri = f"s3://{bucket}/{path}"
    assert_llmisvc_s3_injected(llmisvc=llmisvc, secret_name=secret_name, expected_uri=expected_uri)
    assert_service_account_exists(client=client, namespace=namespace, name=f"{secret_name}-sa")


def assert_llmisvc_uri_injected(llmisvc: LLMInferenceService, expected_uri: str) -> None:
    """Assert a `uri` connection was injected as `spec.model.uri`.

    Args:
        llmisvc: LLMInferenceService to inspect (re-read via `.instance`).
        expected_uri: Expected `spec.model.uri` value (the connection Secret's `URI` key).

    Raises:
        AssertionError: If `spec.model.uri` does not match.
    """
    model_uri = llmisvc.instance.spec.model.get("uri")
    assert model_uri == expected_uri, f"Expected spec.model.uri={expected_uri!r}, got {model_uri!r}"


def assert_llmisvc_oci_injected(llmisvc: LLMInferenceService, secret_name: str) -> None:
    """Assert an `oci` connection was injected as a `template.imagePullSecrets` entry.

    Args:
        llmisvc: LLMInferenceService to inspect (re-read via `.instance`).
        secret_name: Name of the OCI connection Secret expected in `imagePullSecrets`.

    Raises:
        AssertionError: If the secret name is not present in `imagePullSecrets`.
    """
    pull_secrets = llmisvc.instance.spec.template.get("imagePullSecrets") or []
    names = [dict(entry).get("name") for entry in pull_secrets]
    assert secret_name in names, f"Expected {secret_name!r} in template.imagePullSecrets, got {names}"


def assert_llmisvc_connection_cleared(llmisvc: LLMInferenceService) -> None:
    """Assert an LLMISVC's S3 injection fields were cleared by an UPDATE-remove action.

    `spec.model.uri` is a typed field (`*apis.URL`), not an arbitrary map key, so the webhook's
    cleanup (`performLLMISVCCleanup` in odh-model-controller) can only reset it to an empty URL —
    it cannot remove `spec.model` itself. This checks for that emptiness rather than absence.

    Args:
        llmisvc: LLMInferenceService to inspect (re-read via `.instance`).

    Raises:
        AssertionError: If `template.serviceAccountName` is still set, or `spec.model.uri` is
            still populated.
    """
    sa_name = llmisvc.instance.spec.template.get("serviceAccountName")
    assert not sa_name, f"Expected template.serviceAccountName to be cleared, got {sa_name!r}"

    model_uri = llmisvc.instance.spec.model.get("uri")
    assert not model_uri, f"Expected spec.model.uri to be cleared, got {model_uri!r}"


def wait_for_llmisvc_connection_cleared(llmisvc: LLMInferenceService, timeout: int = Timeout.TIMEOUT_2MIN) -> None:
    """Poll until an LLMISVC's connection fields are cleared after an UPDATE-remove action.

    Args:
        llmisvc: LLMInferenceService to poll (re-read via `.instance` on every sample).
        timeout: Seconds to wait before giving up.

    Raises:
        TimeoutError: If the fields are not cleared within `timeout` seconds.
    """

    def _cleared() -> bool:
        spec = llmisvc.instance.spec
        return not spec.template.get("serviceAccountName") and not spec.model.get("uri")

    _wait_for_cleared(predicate=_cleared, timeout=timeout, resource_label=f"LLMInferenceService {llmisvc.name}")


# ---------------------------------------------------------------------------
# Webhook-configuration guard
# ---------------------------------------------------------------------------
def assert_connections_api_webhooks_configured(client: DynamicClient) -> None:
    """Fail fast if the odh-model-controller ConnectionsAPI webhooks are not correctly wired up.

    Scans every `MutatingWebhookConfiguration` on the cluster (the exact parent object name is
    an odh-model-controller implementation detail) and asserts that the new webhook entries are
    present and the old opendatahub-operator ones are gone. A missing/incorrect webhook
    configuration is a real platform-setup defect — the class of regression this suite exists to
    catch — so this must fail rather than skip.

    Args:
        client: Kubernetes dynamic client.

    Raises:
        AssertionError: If the new webhooks are missing, or a stale webhook is still present.
    """
    configured_webhooks: set[str] = set()
    for webhook_config in MutatingWebhookConfiguration.get(client=client):
        configured_webhooks.update(webhook.name for webhook in webhook_config.instance.webhooks or [])

    missing = {ISVC_CONNECTIONS_WEBHOOK, LLMISVC_CONNECTIONS_WEBHOOK} - configured_webhooks
    assert not missing, (
        f"odh-model-controller ConnectionsAPI webhook(s) not found on the cluster: {sorted(missing)}. "
        f"Configured webhooks: {sorted(configured_webhooks)}"
    )

    stale = {STALE_ISVC_CONNECTIONS_WEBHOOK, STALE_LLMISVC_CONNECTIONS_WEBHOOK} & configured_webhooks
    assert not stale, (
        f"Stale opendatahub-operator ConnectionsAPI webhook(s) still present: {sorted(stale)}. "
        "ConnectionsAPI injection must be owned exclusively by odh-model-controller."
    )


# ---------------------------------------------------------------------------
# Functional inference helpers
# ---------------------------------------------------------------------------
def run_isvc_inference(isvc: InferenceService, model_format: str, input_query: dict[str, Any] | None = None) -> None:
    """Send a v2 REST inference request and assert a successful, non-empty response.

    Proves the connection-injected storage source actually let the model load — `Ready` alone
    only proves the server passed its probes (see manual Test 1.5).

    Args:
        isvc: Ready InferenceService to query.
        model_format: Model format served (`ModelFormat.SKLEARN` or `ModelFormat.ONNX`);
            selects the matching sample v2 REST input query when `input_query` is not given.
        input_query: Explicit v2 REST input payload, for models whose input schema doesn't match
            the default sample query for their format (e.g. a reused onnx model with a different
            input tensor name/shape than the sibling mlserver suite's own onnx test model).

    Raises:
        AssertionError: If the response has no `outputs`.
    """
    if input_query is None:
        input_query = SKLEARN_REST_INPUT_QUERY if model_format == ModelFormat.SKLEARN else ONNX_REST_INPUT_QUERY
    response = run_mlserver_inference(isvc=isvc, input_data=input_query, model_version="", protocol=Protocols.REST)
    outputs = response.get("outputs") if isinstance(response, dict) else None
    assert outputs, f"Expected non-empty 'outputs' in inference response for {isvc.name}, got: {response}"


def run_llmisvc_inference(llmisvc: LLMInferenceService, prompt: str = LLMISVC_CHAT_PROMPT) -> None:
    """Send a chat completion request and assert a successful, non-empty response.

    Proves the connection-injected model source actually loaded into vLLM — `Ready` alone only
    proves the pod passed its probes. Applies the RHOAIENG-55154 warm-up workaround first, with a
    longer-than-default 2-minute budget: on resource-constrained clusters, CPU vLLM cold starts can
    outlast the workaround's normal 30s window, which would otherwise surface as a false-negative
    test failure rather than a real injection/webhook defect. `create_connection_llmisvc` disables
    the gateway's auth policy on the LLMISVC, so no bearer token is needed here.

    Args:
        llmisvc: Ready LLMInferenceService to query.
        prompt: Chat prompt to send.

    Raises:
        AssertionError: If the response status is not 200 or the completion text is empty.
    """
    workaround_503_no_healthy_upstream(llmisvc=llmisvc, prompt=prompt, timeout=Timeout.TIMEOUT_2MIN)
    status, body = send_chat_completions(llmisvc=llmisvc, prompt=prompt)
    assert status == 200, f"Expected chat completion to succeed for {llmisvc.name}, got status={status} body={body}"
    text = parse_completion_text(response_body=body)
    assert text.strip(), f"Expected non-empty completion text for {llmisvc.name}, got: {body}"


def _wait_until(predicate: Callable[[], bool], timeout: int, sleep: int = 5) -> None:
    """Poll `predicate` with `TimeoutSampler` until it returns `True`.

    Args:
        predicate: Zero-arg callable returning `True` once the awaited condition holds.
        timeout: Seconds to wait before giving up.
        sleep: Seconds between polls.

    Raises:
        TimeoutExpiredError: If `predicate` never returns `True` within `timeout` seconds
            (raised by `TimeoutSampler` itself once its iterator is exhausted).
    """
    for sample in TimeoutSampler(wait_timeout=timeout, sleep=sleep, func=predicate):
        if sample:
            return


def _wait_for_cleared(predicate: Callable[[], bool], timeout: int, resource_label: str) -> None:
    """Poll `predicate` until connection fields are cleared, raising a resource-specific error.

    Shared by `wait_for_isvc_connection_cleared` and `wait_for_llmisvc_connection_cleared`, which
    differ only in `predicate` and how they identify the resource in the error message.

    Args:
        predicate: Zero-arg callable returning `True` once the connection fields are cleared.
        timeout: Seconds to wait before giving up.
        resource_label: Human-readable resource identifier (e.g. `f"InferenceService {isvc.name}"`)
            used in the raised error message.

    Raises:
        TimeoutError: If `predicate` never returns `True` within `timeout` seconds.
    """
    try:
        _wait_until(predicate=predicate, timeout=timeout)
    except TimeoutExpiredError as exc:
        raise TimeoutError(f"Connection fields on {resource_label} were not cleared within {timeout}s") from exc
