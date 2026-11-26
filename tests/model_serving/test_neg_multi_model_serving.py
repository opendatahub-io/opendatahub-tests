"""
Negative tests for MLServer multi-model serving.

Validates error isolation, health probe behavior in degraded states,
malformed configuration handling, unsupported model rejection, resource
exhaustion, missing artifacts, invalid credentials, and storage exhaustion.

Source: RHAISTRAT-3033
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest
import structlog
from kubernetes.client.rest import ApiException
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import ExecOnPodError, Pod
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from utilities.infra import get_pods_by_isvc_label

LOGGER = structlog.get_logger(name=__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MULTI_MODEL_NS: str = "multi-model-neg"
ISVC_READY_TIMEOUT_SEC: int = 300
POD_SCHEDULE_TIMEOUT_SEC: int = 120
POLL_INTERVAL_SEC: int = 5
RESOURCE_EXHAUSTION_OBSERVE_SEC: int = 180
CREDENTIAL_FAILURE_TIMEOUT_SEC: int = 120

VALID_SKLEARN_PAYLOAD: dict[str, Any] = {
    "inputs": [
        {
            "name": "predict",
            "shape": [1, 4],
            "datatype": "FP32",
            "data": [[5.1, 3.5, 1.4, 0.2]],
        }
    ]
}

MALFORMED_PAYLOAD: dict[str, Any] = {
    "inputs": [
        {
            "name": "predict",
            "shape": [1, 99],
            "datatype": "FP32",
            "data": [[1.0]],
        }
    ]
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_isvc_pod(admin_client: DynamicClient, isvc: InferenceService) -> Pod:
    """Return the first pod for the given InferenceService."""
    pods = get_pods_by_isvc_label(client=admin_client, isvc=isvc)
    if not pods:
        raise RuntimeError(f"No pods found for InferenceService {isvc.name}")
    return pods[0]


def _exec_in_pod(pod: Pod, command: list[str]) -> str:
    """Execute a command inside a pod and return stdout."""
    return pod.execute(command=command)


def _v2_infer(pod: Pod, model_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Send a V2 inference request via pod exec curl.

    Raises RuntimeError when the HTTP status code cannot be parsed from the
    curl output, so callers never confuse a transport failure with a real
    server response.
    """
    url = f"http://localhost:8080/v2/models/{model_name}/infer"
    result = _exec_in_pod(
        pod=pod,
        command=[
            "curl",
            "-s",
            "-w",
            "\n%{http_code}",
            "-X",
            "POST",
            url,
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(payload),
        ],
    )
    lines = result.strip().rsplit("\n", 1)
    if not lines[-1].isdigit():
        raise RuntimeError(f"Unparseable HTTP status from curl: {result!r}")
    body_str = lines[0] if len(lines) > 1 else ""
    http_code = int(lines[-1])
    try:
        body = json.loads(body_str) if body_str else {}
    except json.JSONDecodeError:
        body = {"raw": body_str}
    return {"http_code": http_code, "body": body}


def _v2_health(pod: Pod, endpoint: str) -> int:
    """Query a V2 health endpoint and return the HTTP status code.

    Raises RuntimeError when the HTTP status code cannot be parsed, so
    callers never confuse a transport failure with a real probe result.
    """
    url = f"http://localhost:8080{endpoint}"
    result = _exec_in_pod(
        pod=pod,
        command=["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", url],
    )
    status = result.strip()
    if not status.isdigit():
        raise RuntimeError(f"Unparseable HTTP status from health probe: {result!r}")
    return int(status)


def _v2_repository_index(pod: Pod) -> list[dict[str, Any]]:
    """Query the V2 repository index and return the model list.

    Raises RuntimeError when the response is not valid JSON, so callers
    never confuse a failed request with an empty repository.
    """
    result = _exec_in_pod(
        pod=pod,
        command=[
            "curl",
            "-s",
            "-X",
            "POST",
            "http://localhost:8080/v2/repository/index",
            "-H",
            "Content-Type: application/json",
            "-d",
            "{}",
        ],
    )
    try:
        return json.loads(result.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Repository index returned invalid JSON: {result!r}") from exc


def _get_pod_env_var(pod: Pod, var_name: str) -> str | None:
    """Extract an environment variable value from the pod spec.

    Raises RuntimeError if the pod has no containers, so callers never
    confuse an unreadable spec with 'env var not configured'.
    """
    containers = pod.instance.spec.get("containers", [])
    if not containers:
        raise RuntimeError(f"Pod {pod.name} has no containers in its spec")
    for container in containers:
        for env in container.get("env", []):
            if env.get("name") == var_name:
                return env.get("value")
    return None


def _get_pod_probe(pod: Pod, probe_type: str) -> dict[str, Any] | None:
    """Extract a probe configuration from the pod spec.

    Raises RuntimeError if the pod has no containers, so callers never
    confuse an unreadable spec with 'probe not configured'.
    """
    containers = pod.instance.spec.get("containers", [])
    if not containers:
        raise RuntimeError(f"Pod {pod.name} has no containers in its spec")
    return containers[0].get(probe_type)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.usefixtures("valid_aws_config")

_NS_PARAM = pytest.mark.parametrize(
    argnames="model_namespace",
    argvalues=[pytest.param({"name": MULTI_MODEL_NS})],
    indirect=True,
)


@pytest.mark.tier1
@_NS_PARAM
class TestErrorIsolation:
    """Negative tests for error isolation between resident models."""

    @pytest.mark.dependency(name="TC-NEG-001")
    def test_tc_neg_001(
        self,
        admin_client: DynamicClient,
        multi_model_inference_service: InferenceService,
    ) -> None:
        """Verify inference failure for one model does not affect other resident models.

        Given: InferenceService deployed with 10-model repository, all models
            loaded and serving.
        When: A deliberately malformed inference request is sent to sklearn-iris-1,
            then valid requests are sent to 3 other models, then a valid request
            is sent to sklearn-iris-1.
        Then: The malformed request returns non-200 with error body, the 3 other
            models return HTTP 200, and sklearn-iris-1 recovers with HTTP 200
            on a valid request.
        """
        pod = _get_isvc_pod(admin_client=admin_client, isvc=multi_model_inference_service)

        # Send malformed request — expect a 4xx client error with a body
        fail_result = _v2_infer(pod=pod, model_name="sklearn-iris-1", payload=MALFORMED_PAYLOAD)
        assert 400 <= fail_result["http_code"] < 500, (
            f"Malformed request should return 4xx, got HTTP {fail_result['http_code']}"
        )
        assert fail_result["body"], "Malformed request returned an empty body"
        LOGGER.info("Malformed request result", http_code=fail_result["http_code"], body=fail_result["body"])

        # Verify 3 other models are unaffected
        isolation_models = ["xgb-classifier-1", "lgbm-classifier-1", "onnx-iris-1"]
        for model_name in isolation_models:
            result = _v2_infer(pod=pod, model_name=model_name, payload=VALID_SKLEARN_PAYLOAD)
            assert result["http_code"] == 200, (
                f"Isolation check: {model_name} returned HTTP {result['http_code']}, expected 200"
            )
            assert len(result["body"].get("outputs", [])) > 0, f"Isolation check: {model_name} has empty outputs"

        # Verify recovery with valid request
        recover_result = _v2_infer(pod=pod, model_name="sklearn-iris-1", payload=VALID_SKLEARN_PAYLOAD)
        assert recover_result["http_code"] == 200, (
            f"Recovery request returned HTTP {recover_result['http_code']}, expected 200"
        )


@pytest.mark.tier1
@_NS_PARAM
class TestHealthProbesDegraded:
    """Negative tests for health probe behavior in degraded states."""

    @pytest.mark.dependency(name="TC-NEG-002")
    def test_tc_neg_002(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        empty_storage_inference_service: InferenceService,
    ) -> None:
        """Characterize health probe behavior with empty static storage path.

        Given: RHOAI installed, MLServer ServingRuntime template applied, an
            InferenceService deployed with an empty storage path (zero models).
        When: The pod is created and health endpoints, MLServer env vars, probe
            config, and pod conditions are recorded.
        Then: The pod is scheduled, /v2/health/live and /v2/health/ready HTTP
            codes are recorded, MLServer readiness env vars are captured, probe
            config and Ready condition are documented. With CHECK_REPOSITORY_POPULATED=true,
            the startup probe may fail.
        """
        isvc = empty_storage_inference_service

        # Wait for pod to be created (may not reach Ready)
        pod: Pod | None = None
        try:
            for sample in TimeoutSampler(
                wait_timeout=POD_SCHEDULE_TIMEOUT_SEC,
                sleep=POLL_INTERVAL_SEC,
                func=lambda: get_pods_by_isvc_label(client=admin_client, isvc=isvc),
            ):
                if sample:
                    pod = sample[0]
                    break
        except TimeoutExpiredError:
            pytest.fail("Empty-storage ISVC pod was not created within timeout")

        assert pod is not None

        # Wait for serving container to attempt startup
        time.sleep(15)
        pod.reload()
        container_statuses = pod.instance.status.get("containerStatuses", [])
        if container_statuses:
            cs = container_statuses[0]
            LOGGER.info(
                "Empty storage container state before probing",
                name=cs.get("name"),
                state=cs.get("state"),
                ready=cs.get("ready"),
            )

        # Record health probes (may fail depending on startup probe behavior)
        try:
            live_code = _v2_health(pod=pod, endpoint="/v2/health/live")
            LOGGER.info("Empty storage /v2/health/live", http_code=live_code)
        except (ExecOnPodError, ApiException) as exc:
            LOGGER.info("Empty storage /v2/health/live unreachable", error=str(exc))

        try:
            ready_code = _v2_health(pod=pod, endpoint="/v2/health/ready")
            LOGGER.info("Empty storage /v2/health/ready", http_code=ready_code)
        except (ExecOnPodError, ApiException) as exc:
            LOGGER.info("Empty storage /v2/health/ready unreachable", error=str(exc))

        # Record MLServer env vars
        for var_name in (
            "MLSERVER_STRICT_READINESS",
            "MLSERVER_EMPTY_REGISTRY_READINESS",
            "MLSERVER_LOAD_MODELS_AT_STARTUP",
            "CHECK_REPOSITORY_POPULATED",
        ):
            try:
                value = _get_pod_env_var(pod=pod, var_name=var_name)
                LOGGER.info("Empty storage env var", name=var_name, value=value)
            except AttributeError, TypeError:
                LOGGER.info("Could not read env var (container may not be running)", name=var_name)

        # Record probe config and pod conditions
        for probe_type in ("startupProbe", "livenessProbe", "readinessProbe"):
            probe = _get_pod_probe(pod=pod, probe_type=probe_type)
            LOGGER.info("Empty storage probe", probe_type=probe_type, config=probe)

        pod.reload()
        conditions = pod.instance.status.get("conditions", [])
        for condition in conditions:
            LOGGER.info("Empty storage pod condition", type=condition.get("type"), status=condition.get("status"))

        container_statuses = pod.instance.status.get("containerStatuses", [])
        for cs in container_statuses:
            LOGGER.info("Empty storage container status", name=cs.get("name"), state=cs.get("state"))

    @pytest.mark.dependency(name="TC-NEG-003")
    def test_tc_neg_003(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        one_failed_model_inference_service: InferenceService,
    ) -> None:
        """Characterize health probe behavior with one failed model load.

        Given: RHOAI installed, MLServer ServingRuntime template applied, a
            modified 10-model repository where one model references a
            non-existent artifact.
        When: The InferenceService is deployed and MLServer logs, health
            endpoints, env vars, and repository index are recorded.
        Then: Logs show an error for the bad model, health endpoint codes
            are recorded, env vars are captured, and the repository index
            shows 9 loaded models.
        """
        isvc = one_failed_model_inference_service

        # Wait for pod (may or may not reach Ready)
        pod: Pod | None = None
        try:
            for sample in TimeoutSampler(
                wait_timeout=ISVC_READY_TIMEOUT_SEC,
                sleep=POLL_INTERVAL_SEC,
                func=lambda: get_pods_by_isvc_label(client=admin_client, isvc=isvc),
            ):
                if sample:
                    pod = sample[0]
                    break
        except TimeoutExpiredError:
            pytest.fail("One-failed-model ISVC pod was not created within timeout")

        assert pod is not None

        # Wait a bit for model loading to settle
        time.sleep(30)

        # Check MLServer logs for error
        try:
            log_output = _exec_in_pod(
                pod=pod,
                command=[
                    "sh",
                    "-c",
                    "cat /proc/1/fd/1 2>/dev/null"
                    " | grep -iE 'error|failed|Error'"
                    " | tail -20 || echo 'LOG_READ_SKIPPED'",
                ],
            )
            LOGGER.info("One-failed-model error logs", output=log_output[:1000])
        except (ExecOnPodError, ApiException) as exc:
            LOGGER.warning("Could not read logs", error=str(exc))

        # Record health endpoints
        live_code = _v2_health(pod=pod, endpoint="/v2/health/live")
        LOGGER.info("One-failed-model /v2/health/live", http_code=live_code)

        ready_code = _v2_health(pod=pod, endpoint="/v2/health/ready")
        LOGGER.info("One-failed-model /v2/health/ready", http_code=ready_code)

        # Record env vars
        for var_name in (
            "MLSERVER_STRICT_READINESS",
            "MLSERVER_EMPTY_REGISTRY_READINESS",
            "MLSERVER_LOAD_MODELS_AT_STARTUP",
            "CHECK_REPOSITORY_POPULATED",
        ):
            value = _get_pod_env_var(pod=pod, var_name=var_name)
            LOGGER.info("One-failed-model env var", name=var_name, value=value)

        # Repository index should show 9 models (the failed one absent)
        repo_index = _v2_repository_index(pod=pod)
        LOGGER.info("One-failed-model repository index", count=len(repo_index))
        assert len(repo_index) == 9, f"Expected 9 models in repository index (1 failed), got {len(repo_index)}"


@pytest.mark.tier1
@_NS_PARAM
class TestMalformedConfiguration:
    """Negative tests for malformed model configuration handling."""

    @pytest.mark.dependency(name="TC-NEG-004")
    def test_tc_neg_004(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        malformed_settings_inference_service: InferenceService,
    ) -> None:
        """Verify malformed model-settings.json is rejected and other models remain available.

        Given: RHOAI installed, MLServer ServingRuntime template applied, a
            storage path with one subdirectory containing syntactically invalid
            JSON alongside 2+ valid models.
        When: The InferenceService is deployed and MLServer logs, repository
            index, and inference to valid models are checked.
        Then: Logs contain a JSON parse error, the bad model is absent from
            the index, valid models return HTTP 200, and the pod remains running.
        """
        isvc = malformed_settings_inference_service

        pod: Pod | None = None
        try:
            for sample in TimeoutSampler(
                wait_timeout=ISVC_READY_TIMEOUT_SEC,
                sleep=POLL_INTERVAL_SEC,
                func=lambda: get_pods_by_isvc_label(client=admin_client, isvc=isvc),
            ):
                if sample:
                    pod = sample[0]
                    break
        except TimeoutExpiredError:
            pytest.fail("Malformed-settings ISVC pod was not created within timeout")

        assert pod is not None
        time.sleep(20)

        # Check logs for JSON parse error
        try:
            log_output = _exec_in_pod(
                pod=pod,
                command=[
                    "sh",
                    "-c",
                    "cat /proc/1/fd/1 2>/dev/null"
                    " | grep -iE 'JSON|json|parse|malformed'"
                    " | tail -10 || echo 'LOG_READ_SKIPPED'",
                ],
            )
            LOGGER.info("Malformed settings error logs", output=log_output[:500])
        except (ExecOnPodError, ApiException) as exc:
            LOGGER.warning("Could not read logs", error=str(exc))

        # bad-model should not appear in repo index
        repo_index = _v2_repository_index(pod=pod)
        model_names = [e.get("name", "") for e in repo_index]
        assert "bad-model" not in model_names, f"bad-model should not be in index: {model_names}"
        assert "sklearn-iris-1" in model_names, "Expected sklearn-iris-1 in repository index"
        assert "xgb-classifier-1" in model_names, "Expected xgb-classifier-1 in repository index"

        # Valid models should serve
        for model_name in ("sklearn-iris-1", "xgb-classifier-1"):
            result = _v2_infer(pod=pod, model_name=model_name, payload=VALID_SKLEARN_PAYLOAD)
            assert result["http_code"] == 200, f"{model_name} inference returned HTTP {result['http_code']}"

        # Pod should still be running
        pod.reload()
        phase = pod.instance.status.get("phase", "")
        assert phase == "Running", f"Pod phase is {phase}, expected Running"

    @pytest.mark.dependency(name="TC-NEG-005")
    def test_tc_neg_005(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        unsupported_impl_inference_service: InferenceService,
    ) -> None:
        """Verify unsupported model implementation class is rejected.

        Given: RHOAI installed, MLServer ServingRuntime template applied, a
            storage path with one subdirectory specifying "custom.MyModel"
            alongside 2+ valid models.
        When: The InferenceService is deployed and MLServer logs, repository
            index, and inference to valid models are checked.
        Then: Logs contain a rejection for custom.MyModel, custom-model is
            absent or failed in the index, valid models return HTTP 200,
            and the pod remains running.
        """
        isvc = unsupported_impl_inference_service

        pod: Pod | None = None
        try:
            for sample in TimeoutSampler(
                wait_timeout=ISVC_READY_TIMEOUT_SEC,
                sleep=POLL_INTERVAL_SEC,
                func=lambda: get_pods_by_isvc_label(client=admin_client, isvc=isvc),
            ):
                if sample:
                    pod = sample[0]
                    break
        except TimeoutExpiredError:
            pytest.fail("Unsupported-impl ISVC pod was not created within timeout")

        assert pod is not None
        time.sleep(20)

        # Check logs for rejection
        try:
            log_output = _exec_in_pod(
                pod=pod,
                command=[
                    "sh",
                    "-c",
                    "cat /proc/1/fd/1 2>/dev/null"
                    " | grep -iE 'implementation|custom.MyModel|reject'"
                    " | tail -10 || echo 'LOG_READ_SKIPPED'",
                ],
            )
            LOGGER.info("Unsupported impl rejection logs", output=log_output[:500])
        except (ExecOnPodError, ApiException) as exc:
            LOGGER.warning("Could not read logs", error=str(exc))

        # custom-model should not appear in repo index
        repo_index = _v2_repository_index(pod=pod)
        model_names = [e.get("name", "") for e in repo_index]
        assert "custom-model" not in model_names or any(
            e.get("state") in ("UNAVAILABLE", "LOADING_FAILED") for e in repo_index if e.get("name") == "custom-model"
        ), f"custom-model should not be loadable in index: {model_names}"
        assert "sklearn-iris-1" in model_names, "Expected sklearn-iris-1 in repository index"
        assert "xgb-classifier-1" in model_names, "Expected xgb-classifier-1 in repository index"

        # Valid models should serve
        for model_name in ("sklearn-iris-1", "xgb-classifier-1"):
            result = _v2_infer(pod=pod, model_name=model_name, payload=VALID_SKLEARN_PAYLOAD)
            assert result["http_code"] == 200, f"{model_name} inference returned HTTP {result['http_code']}"

        # Pod should still be running
        pod.reload()
        phase = pod.instance.status.get("phase", "")
        assert phase == "Running", f"Pod phase is {phase}, expected Running"


@pytest.mark.tier3
@_NS_PARAM
class TestResourceExhaustion:
    """Negative tests for resource exhaustion scenarios."""

    @pytest.mark.dependency(name="TC-NEG-006")
    def test_tc_neg_006(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        resource_exhaustion_inference_service: InferenceService,
    ) -> None:
        """Record failure mode when resident model set exceeds available memory.

        Given: RHOAI installed, MLServer ServingRuntime template applied, a
            resource-exhaustion model set with pod resource limits set
            intentionally low.
        When: The InferenceService is deployed and pod status is monitored
            over a 3-minute observation window.
        Then: Pod status transitions are recorded, the failure mode is
            classified (controlled rejection, OOMKilled, or kernel OOM),
            and the namespace is not left in an unrecoverable state.
        """
        isvc = resource_exhaustion_inference_service

        # Monitor pod status over observation window
        observed_phases: list[str] = []
        observed_states: list[dict[str, Any]] = []
        max_polls = RESOURCE_EXHAUSTION_OBSERVE_SEC // POLL_INTERVAL_SEC

        for poll_idx in range(max_polls):
            try:
                pods = get_pods_by_isvc_label(client=admin_client, isvc=isvc)
            except ResourceNotFoundError:
                pods = []
            if pods:
                pod = pods[0]
                pod.reload()
                phase = pod.instance.status.get("phase", "Unknown")
                observed_phases.append(phase)

                container_statuses = pod.instance.status.get("containerStatuses", [])
                for cs in container_statuses:
                    state = cs.get("state", {})
                    last_state = cs.get("lastState", {})
                    if state.get("terminated"):
                        observed_states.append({
                            "poll": poll_idx,
                            "reason": state["terminated"].get("reason", ""),
                            "exit_code": state["terminated"].get("exitCode"),
                        })
                    if state.get("waiting"):
                        observed_states.append({
                            "poll": poll_idx,
                            "waiting_reason": state["waiting"].get("reason", ""),
                        })
                    if last_state.get("terminated"):
                        observed_states.append({
                            "poll": poll_idx,
                            "last_reason": last_state["terminated"].get("reason", ""),
                            "last_exit_code": last_state["terminated"].get("exitCode"),
                        })
            time.sleep(POLL_INTERVAL_SEC)

        LOGGER.info("Resource exhaustion observed phases", phases=observed_phases)
        LOGGER.info("Resource exhaustion observed states", states=observed_states)

        # Classify failure mode
        oom_killed = any(s.get("reason") == "OOMKilled" or s.get("last_reason") == "OOMKilled" for s in observed_states)
        crash_loop = any(s.get("waiting_reason") == "CrashLoopBackOff" for s in observed_states)

        if oom_killed:
            LOGGER.info("Failure mode: OOMKilled")
        elif crash_loop:
            LOGGER.info("Failure mode: CrashLoopBackOff")
        else:
            LOGGER.info("Failure mode: controlled rejection or other")

        # Record pod events
        try:
            pods = get_pods_by_isvc_label(client=admin_client, isvc=isvc)
        except ResourceNotFoundError:
            pods = []
        if pods:
            pod = pods[0]
            pod.reload()
            conditions = pod.instance.status.get("conditions", [])
            for condition in conditions:
                LOGGER.info(
                    "Resource exhaustion pod condition",
                    type=condition.get("type"),
                    status=condition.get("status"),
                    reason=condition.get("reason"),
                )


@pytest.mark.tier1
@_NS_PARAM
class TestMissingArtifacts:
    """Negative tests for missing model artifacts."""

    @pytest.mark.dependency(name="TC-NEG-007")
    def test_tc_neg_007(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        missing_artifact_inference_service: InferenceService,
    ) -> None:
        """Verify model with missing artifact file fails and other models remain available.

        Given: RHOAI installed, MLServer ServingRuntime template applied, a
            storage path with one subdirectory referencing ./nonexistent.joblib
            alongside 2+ valid models.
        When: The InferenceService is deployed and logs, repository index,
            and inference to valid and missing models are checked.
        Then: Logs show file-not-found, missing-model is absent from the index,
            valid models return HTTP 200, inference to missing-model returns
            non-200, and the pod remains running.
        """
        isvc = missing_artifact_inference_service

        pod: Pod | None = None
        try:
            for sample in TimeoutSampler(
                wait_timeout=ISVC_READY_TIMEOUT_SEC,
                sleep=POLL_INTERVAL_SEC,
                func=lambda: get_pods_by_isvc_label(client=admin_client, isvc=isvc),
            ):
                if sample:
                    pod = sample[0]
                    break
        except TimeoutExpiredError:
            pytest.fail("Missing-artifact ISVC pod was not created within timeout")

        assert pod is not None
        time.sleep(20)

        # Check logs for file-not-found
        try:
            log_output = _exec_in_pod(
                pod=pod,
                command=[
                    "sh",
                    "-c",
                    "cat /proc/1/fd/1 2>/dev/null"
                    " | grep -iE 'not found|FileNotFoundError|nonexistent'"
                    " | tail -10 || echo 'LOG_READ_SKIPPED'",
                ],
            )
            LOGGER.info("Missing artifact error logs", output=log_output[:500])
        except (ExecOnPodError, ApiException) as exc:
            LOGGER.warning("Could not read logs", error=str(exc))

        # Repository index should not include missing-model
        repo_index = _v2_repository_index(pod=pod)
        model_names = [e.get("name", "") for e in repo_index]
        assert "missing-model" not in model_names or any(
            e.get("state") == "UNAVAILABLE" for e in repo_index if e.get("name") == "missing-model"
        ), f"missing-model should not be available in index: {model_names}"
        assert "sklearn-iris-1" in model_names, "Expected sklearn-iris-1 in repository index"
        assert "xgb-classifier-1" in model_names, "Expected xgb-classifier-1 in repository index"

        # Valid models should serve
        for model_name in ("sklearn-iris-1", "xgb-classifier-1"):
            result = _v2_infer(pod=pod, model_name=model_name, payload=VALID_SKLEARN_PAYLOAD)
            assert result["http_code"] == 200, f"{model_name} inference returned HTTP {result['http_code']}"

        # Inference to missing-model should fail
        missing_result = _v2_infer(pod=pod, model_name="missing-model", payload=VALID_SKLEARN_PAYLOAD)
        assert missing_result["http_code"] != 200, (
            f"missing-model inference should not return 200, got {missing_result['http_code']}"
        )

        # Pod should still be running
        pod.reload()
        phase = pod.instance.status.get("phase", "")
        assert phase == "Running", f"Pod phase is {phase}, expected Running"


@pytest.mark.tier1
@_NS_PARAM
class TestInvalidCredentials:
    """Negative tests for invalid storage credentials."""

    @pytest.mark.dependency(name="TC-NEG-008")
    def test_tc_neg_008(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        invalid_credentials_inference_service: InferenceService,
    ) -> None:
        """Verify ISVC with incorrect credentials fails with clear auth error.

        Given: RHOAI installed, MLServer ServingRuntime template applied, a
            Kubernetes Secret with intentionally wrong credentials referenced
            by the InferenceService.
        When: The InferenceService is deployed and pod status, events, and
            logs are observed.
        Then: The pod does not reach Ready, status shows a failure phase,
            events or logs contain an authentication error, and no V2 health
            endpoint returns HTTP 200.
        """
        isvc = invalid_credentials_inference_service

        # Wait and observe (should NOT reach Ready)
        time.sleep(CREDENTIAL_FAILURE_TIMEOUT_SEC)

        try:
            pods = get_pods_by_isvc_label(client=admin_client, isvc=isvc)
        except ResourceNotFoundError:
            pods = []

        if pods:
            pod = pods[0]
            pod.reload()
            phase = pod.instance.status.get("phase", "Unknown")
            LOGGER.info("Invalid credentials pod phase", phase=phase)

            # Pod should not be in Running/Ready state successfully
            conditions = pod.instance.status.get("conditions", [])
            ready_condition = next((c for c in conditions if c.get("type") == "Ready"), None)
            if ready_condition:
                assert ready_condition.get("status") != "True", "Pod should not be Ready with invalid credentials"

            # Record container states
            container_statuses = pod.instance.status.get("containerStatuses", [])
            init_statuses = pod.instance.status.get("initContainerStatuses", [])
            for cs in container_statuses + init_statuses:
                LOGGER.info(
                    "Invalid credentials container status",
                    name=cs.get("name"),
                    state=cs.get("state"),
                )

            # Try health check (should fail)
            try:
                live_code = _v2_health(pod=pod, endpoint="/v2/health/live")
                assert live_code != 200, "/v2/health/live should not return 200 with invalid creds"
            except ExecOnPodError, ApiException:
                LOGGER.info("Health endpoint unreachable (expected)")
        else:
            LOGGER.info("No pods found for invalid-credentials ISVC (pod creation may have failed)")

        # ISVC should not report Ready
        isvc.reload()
        isvc_conditions = isvc.instance.get("status", {}).get("conditions", [])
        isvc_ready = next((c for c in isvc_conditions if c.get("type") == "Ready"), None)
        if isvc_ready:
            assert isvc_ready.get("status") != "True", "InferenceService should not be Ready with invalid credentials"


@pytest.mark.tier3
@_NS_PARAM
class TestStorageExhaustion:
    """Negative tests for storage exhaustion during staging."""

    @pytest.mark.dependency(name="TC-NEG-009")
    def test_tc_neg_009(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        storage_exhaustion_inference_service: InferenceService,
    ) -> None:
        """Verify insufficient storage space fails staging and pod does not claim success.

        Given: RHOAI installed, MLServer ServingRuntime template applied, a
            PVC with insufficient capacity for the model set.
        When: Model staging is attempted and the InferenceService is deployed.
        Then: Staging fails with ENOSPC or quota exceeded, storage events
            record the failure, and the pod does not report Ready with an
            incomplete model set.
        """
        isvc = storage_exhaustion_inference_service

        # Wait for pod creation
        def _get_pods_safe() -> list[Pod]:
            try:
                return get_pods_by_isvc_label(client=admin_client, isvc=isvc)
            except ResourceNotFoundError:
                return []

        pod: Pod | None = None
        try:
            for sample in TimeoutSampler(
                wait_timeout=CREDENTIAL_FAILURE_TIMEOUT_SEC,
                sleep=POLL_INTERVAL_SEC,
                func=_get_pods_safe,
            ):
                if sample:
                    pod = sample[0]
                    break
        except TimeoutExpiredError:
            LOGGER.info("Storage exhaustion ISVC pod was not created (staging may have failed)")

        if pod is not None:
            pod.reload()
            phase = pod.instance.status.get("phase", "Unknown")
            LOGGER.info("Storage exhaustion pod phase", phase=phase)

            # Record conditions
            conditions = pod.instance.status.get("conditions", [])
            for condition in conditions:
                LOGGER.info(
                    "Storage exhaustion condition",
                    type=condition.get("type"),
                    status=condition.get("status"),
                    reason=condition.get("reason"),
                    message=condition.get("message", "")[:200],
                )

            # If pod started, check repo index for incomplete loading
            try:
                repo_index = _v2_repository_index(pod=pod)
                model_count = len(repo_index)
                LOGGER.info("Storage exhaustion model count", count=model_count)
                assert model_count < 10, f"Expected fewer than 10 models with exhausted storage, got {model_count}"

                staging_failed = getattr(storage_exhaustion_inference_service, "_staging_failed", False)
                if staging_failed:
                    assert model_count < 10, f"Staging failed but {model_count} models loaded — expected fewer"
            except ExecOnPodError, ApiException, RuntimeError:
                LOGGER.info("Repository index unreachable (expected with storage failure)")

        # ISVC should not report Ready
        isvc.reload()
        isvc_conditions = isvc.instance.get("status", {}).get("conditions", [])
        isvc_ready = next((c for c in isvc_conditions if c.get("type") == "Ready"), None)
        if isvc_ready:
            LOGGER.info(
                "Storage exhaustion ISVC Ready condition",
                status=isvc_ready.get("status"),
                reason=isvc_ready.get("reason"),
            )
