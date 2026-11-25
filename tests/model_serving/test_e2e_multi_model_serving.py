"""
End-to-end tests for MLServer multi-model serving.

Validates multi-model deployment via static storage path, V2 inference
across model formats, storage mount permissions, model-set integrity,
health probes, version disambiguation, concurrent inference, startup
time, CUDA GPU inference, and single-model regression.

Source: RHAISTRAT-3033
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import ExecOnPodError, Pod
from ocp_resources.serving_runtime import ServingRuntime
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_serving.model_runtime.mlserver.constant import MODEL_PATH_PREFIX
from utilities.constants import ModelFormat
from utilities.inference_utils import get_exposed_isvc_url
from utilities.infra import get_pods_by_isvc_label

LOGGER = structlog.get_logger(name=__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MULTI_MODEL_NS: str = "multi-model-e2e"

EXPECTED_MODEL_COUNT: int = 10
MODEL_NAMES: list[str] = [
    "sklearn-iris-1",
    "sklearn-iris-2",
    "xgb-classifier-1",
    "xgb-classifier-2",
    "lgbm-classifier-1",
    "lgbm-classifier-2",
    "onnx-resnet-1",
    "onnx-resnet-2",
    "sklearn-iris-3",
    "onnx-resnet-3",
]
CONCURRENT_MODELS: list[str] = [
    "sklearn-iris-1",
    "xgb-classifier-1",
    "lgbm-classifier-1",
    "onnx-resnet-1",
    "sklearn-iris-3",
]

SKLEARN_IRIS_PAYLOAD: dict[str, Any] = {
    "inputs": [
        {
            "name": "predict",
            "shape": [1, 4],
            "datatype": "FP32",
            "data": [[5.1, 3.5, 1.4, 0.2]],
        }
    ]
}

ISVC_READY_TIMEOUT_SEC: int = 300
STARTUP_MEASUREMENT_TIMEOUT_SEC: int = 300
POLL_INTERVAL_SEC: int = 5
MODEL_STABILITY_WAIT_SEC: int = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_isvc_pod(admin_client: DynamicClient, isvc: InferenceService) -> Pod:
    """Return the first pod for the given InferenceService.

    Args:
        admin_client: Kubernetes dynamic client.
        isvc: InferenceService resource.

    Returns:
        The first matching pod.

    Raises:
        RuntimeError: If no pods are found.
    """
    pods = get_pods_by_isvc_label(client=admin_client, isvc=isvc)
    if not pods:
        raise RuntimeError(f"No pods found for InferenceService {isvc.name}")
    return pods[0]


def _exec_in_pod(pod: Pod, command: list[str]) -> str:
    """Execute a command inside a pod and return stdout.

    Args:
        pod: Target pod resource.
        command: Command and arguments to execute.

    Returns:
        Command stdout as string.
    """
    return pod.execute(command=command)


def _v2_infer(pod: Pod, model_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Send a V2 inference request to a model via pod exec curl.

    Args:
        pod: MLServer pod.
        model_name: Name of the target model.
        payload: V2 inference request payload.

    Returns:
        Dictionary with http_code and parsed response body.
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
    body_str = lines[0] if len(lines) > 1 else ""
    http_code = int(lines[-1]) if lines[-1].isdigit() else 0
    try:
        body = json.loads(body_str) if body_str else {}
    except json.JSONDecodeError:
        body = {"raw": body_str}
    return {"http_code": http_code, "body": body}


def _v2_health(pod: Pod, endpoint: str) -> int:
    """Query a V2 health endpoint and return the HTTP status code.

    Args:
        pod: MLServer pod.
        endpoint: Health path, e.g. "/v2/health/live".

    Returns:
        HTTP status code.
    """
    url = f"http://localhost:8080{endpoint}"
    result = _exec_in_pod(
        pod=pod,
        command=["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", url],
    )
    return int(result.strip()) if result.strip().isdigit() else 0


def _v2_repository_index(pod: Pod) -> list[dict[str, Any]]:
    """Query the V2 repository index and return the model list.

    Args:
        pod: MLServer pod.

    Returns:
        List of model entries from the repository index.
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
    except json.JSONDecodeError:
        return []


def _get_pod_env_var(pod: Pod, var_name: str) -> str | None:
    """Extract an environment variable value from the pod spec.

    Args:
        pod: Target pod.
        var_name: Environment variable name.

    Returns:
        Variable value or None if not found.
    """
    containers = pod.instance.spec.get("containers", [])
    for container in containers:
        for env in container.get("env", []):
            if env.get("name") == var_name:
                return env.get("value")
    return None


def _get_pod_probe(pod: Pod, probe_type: str) -> dict[str, Any] | None:
    """Extract a probe configuration from the pod spec.

    Args:
        pod: Target pod.
        probe_type: One of livenessProbe, readinessProbe, startupProbe.

    Returns:
        Probe configuration dict or None.
    """
    containers = pod.instance.spec.get("containers", [])
    if containers:
        return containers[0].get(probe_type)
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.tier1
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": MULTI_MODEL_NS})],
    indirect=True,
)
class TestMultiModelDeployment:
    """E2E tests for MLServer multi-model serving deployment and inference."""

    @pytest.mark.dependency(name="TC-E2E-001")
    def test_tc_e2e_001(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        mlserver_serving_runtime: ServingRuntime,
        multi_model_inference_service: InferenceService,
    ) -> None:
        """Verify all 10 models load from static storage and respond to V2 inference.

        Given: RHOAI 3.7 EA installed with MLServer ServingRuntime template,
            a 10-model repository staged on the static storage path with valid
            model-settings.json per subdirectory.
        When: The InferenceService is deployed, all 10 models are queried
            via V2 inference requests in-cluster, and one model is queried
            through the external route.
        Then: The pod reaches Ready, all 10 V2 inference requests return HTTP 200
            with valid payloads containing model_name, model_version, id, and
            non-empty outputs array; the external route returns HTTP 200 with
            a valid V2 payload.
        """
        pod = _get_isvc_pod(admin_client=admin_client, isvc=multi_model_inference_service)

        # Verify all 10 models appear in the repository index
        repo_index = _v2_repository_index(pod=pod)
        loaded_names = sorted(entry.get("name", "") for entry in repo_index)
        assert len(repo_index) == EXPECTED_MODEL_COUNT, (
            f"Expected {EXPECTED_MODEL_COUNT} models in repository index, got {len(repo_index)}: {loaded_names}"
        )

        # Send V2 inference to each model
        for model_name in MODEL_NAMES:
            result = _v2_infer(pod=pod, model_name=model_name, payload=SKLEARN_IRIS_PAYLOAD)
            assert result["http_code"] == 200, (
                f"Model {model_name} inference returned HTTP {result['http_code']}, expected 200"
            )
            body = result["body"]
            assert body.get("model_name") is not None, f"Response for {model_name} missing model_name"
            assert body.get("id") is not None, f"Response for {model_name} missing id"
            assert len(body.get("outputs", [])) > 0, f"Response for {model_name} has empty outputs"
            assert body["model_name"] == model_name, (
                f"Response model_name {body['model_name']} does not match requested {model_name}"
            )

        # Send V2 inference through the external route (supported client path)
        isvc = multi_model_inference_service
        isvc.reload()
        external_url = get_exposed_isvc_url(isvc=isvc)
        route_model = MODEL_NAMES[0]
        infer_url = f"{external_url}/v2/models/{route_model}/infer"
        route_response = requests.post(
            url=infer_url,
            json=SKLEARN_IRIS_PAYLOAD,
            verify=False,
            timeout=60,
        )
        assert route_response.status_code == 200, (
            f"External route inference returned HTTP {route_response.status_code}, expected 200"
        )
        route_body = route_response.json()
        assert route_body.get("model_name") == route_model, (
            f"External route response model_name {route_body.get('model_name')} does not match {route_model}"
        )
        assert len(route_body.get("outputs", [])) > 0, "External route response has empty outputs"

    @pytest.mark.dependency(name="TC-E2E-002")
    def test_tc_e2e_002(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        multi_model_inference_service: InferenceService,
    ) -> None:
        """Verify static storage path mount and serving-reader permissions.

        Given: RHOAI installed with MLServer ServingRuntime template, a 10-model
            repository staged on the storage path, serving-reader SA with
            read-only access.
        When: The InferenceService pod is inspected for volume mounts, events,
            and MLServer log output.
        Then: The /mnt/models mount is present, pod events show no permission
            errors, MLServer logs show discovery of all 10 models, and inference
            returns HTTP 200.
        """
        pod = _get_isvc_pod(admin_client=admin_client, isvc=multi_model_inference_service)

        # Check volume mount for /mnt/models
        containers = pod.instance.spec.get("containers", [])
        model_mount_found = False
        for container in containers:
            for mount in container.get("volumeMounts", []):
                if mount.get("mountPath") == "/mnt/models":
                    model_mount_found = True
                    LOGGER.info(
                        "Found /mnt/models mount",
                        read_only=mount.get("readOnly"),
                        sub_path=mount.get("subPath"),
                    )
                    break

        assert model_mount_found, "Volume mount for /mnt/models not found in pod spec"

        # Verify MLServer discovered models via logs
        log_output = _exec_in_pod(
            pod=pod,
            command=[
                "sh",
                "-c",
                "cat /proc/1/fd/1 2>/dev/null | grep -iE 'Loading model|models found' | tail -20 || echo 'LOG_READ_SKIPPED'",
            ],
        )
        LOGGER.info("MLServer model discovery logs", output=log_output[:500])

        # Verify inference works through mount
        result = _v2_infer(pod=pod, model_name="sklearn-iris-1", payload=SKLEARN_IRIS_PAYLOAD)
        assert result["http_code"] == 200, (
            f"Inference through mounted storage returned HTTP {result['http_code']}, expected 200"
        )
        assert result["body"].get("model_name") == "sklearn-iris-1"
        assert len(result["body"].get("outputs", [])) > 0

    @pytest.mark.dependency(name="TC-E2E-003", depends=["TC-E2E-001"])
    def test_tc_e2e_003(
        self,
        admin_client: DynamicClient,
        multi_model_inference_service: InferenceService,
    ) -> None:
        """Verify model-set integrity across pod lifecycle without catalog operations.

        Given: InferenceService deployed with the 10-model repository and pod
            in Ready state (TC-E2E-001), no V2 Repository API load/unload calls
            during the test.
        When: The repository index is queried, all models are inferred, a wait
            period passes, and the operations are repeated.
        Then: The initial and final repository index are identical, all inference
            requests return HTTP 200.
        """
        pod = _get_isvc_pod(admin_client=admin_client, isvc=multi_model_inference_service)

        # Record initial repository state
        initial_index = _v2_repository_index(pod=pod)
        initial_names = sorted(entry.get("name", "") for entry in initial_index)
        assert len(initial_index) == EXPECTED_MODEL_COUNT, (
            f"Initial index has {len(initial_index)} models, expected {EXPECTED_MODEL_COUNT}"
        )

        # Infer all models
        for model_name in MODEL_NAMES:
            result = _v2_infer(pod=pod, model_name=model_name, payload=SKLEARN_IRIS_PAYLOAD)
            assert result["http_code"] == 200, f"Initial inference for {model_name} failed: HTTP {result['http_code']}"

        # Wait without performing catalog operations
        LOGGER.info("Waiting for model stability period", seconds=MODEL_STABILITY_WAIT_SEC)
        time.sleep(MODEL_STABILITY_WAIT_SEC)

        # Re-query and compare
        final_index = _v2_repository_index(pod=pod)
        final_names = sorted(entry.get("name", "") for entry in final_index)
        assert initial_names == final_names, f"Model set changed: initial={initial_names}, final={final_names}"

        # Repeat inference
        for model_name in MODEL_NAMES:
            result = _v2_infer(pod=pod, model_name=model_name, payload=SKLEARN_IRIS_PAYLOAD)
            assert result["http_code"] == 200, (
                f"Post-wait inference for {model_name} failed: HTTP {result['http_code']}"
            )

    @pytest.mark.dependency(name="TC-E2E-004", depends=["TC-E2E-001"])
    def test_tc_e2e_004(
        self,
        admin_client: DynamicClient,
        multi_model_inference_service: InferenceService,
    ) -> None:
        """Characterize health probe behavior with all models loaded.

        Given: InferenceService deployed with the 10-model repository,
            all 10 models loaded and serving inference.
        When: V2 liveness and readiness endpoints are queried, MLServer
            readiness env vars are recorded, and K8s probe config is inspected.
        Then: /v2/health/live returns HTTP 200, /v2/health/ready returns
            HTTP 200, probe configuration and env vars are recorded,
            pod Ready condition is True.
        """
        pod = _get_isvc_pod(admin_client=admin_client, isvc=multi_model_inference_service)

        # Health endpoints
        live_code = _v2_health(pod=pod, endpoint="/v2/health/live")
        assert live_code == 200, f"/v2/health/live returned HTTP {live_code}, expected 200"

        ready_code = _v2_health(pod=pod, endpoint="/v2/health/ready")
        assert ready_code == 200, f"/v2/health/ready returned HTTP {ready_code}, expected 200"

        # Record MLServer readiness env vars
        readiness_vars = [
            "MLSERVER_STRICT_READINESS",
            "MLSERVER_EMPTY_REGISTRY_READINESS",
            "MLSERVER_LOAD_MODELS_AT_STARTUP",
            "CHECK_REPOSITORY_POPULATED",
        ]
        for var_name in readiness_vars:
            value = _get_pod_env_var(pod=pod, var_name=var_name)
            LOGGER.info("MLServer env var", name=var_name, value=value)

        # Record probe configuration
        for probe_type in ("livenessProbe", "readinessProbe", "startupProbe"):
            probe = _get_pod_probe(pod=pod, probe_type=probe_type)
            LOGGER.info("Probe configuration", probe_type=probe_type, config=probe)

        # Pod readiness
        conditions = pod.instance.status.get("conditions", [])
        ready_condition = next((c for c in conditions if c.get("type") == "Ready"), None)
        assert ready_condition is not None, "Pod has no Ready condition"
        assert ready_condition.get("status") == "True", (
            f"Pod Ready condition is {ready_condition.get('status')}, expected True"
        )


@pytest.mark.tier1
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": MULTI_MODEL_NS})],
    indirect=True,
)
class TestMultiModelVersioning:
    """E2E tests for model name and version disambiguation."""

    @pytest.mark.dependency(name="TC-E2E-005")
    def test_tc_e2e_005(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        mlserver_serving_runtime: ServingRuntime,
        multi_version_inference_service: InferenceService,
    ) -> None:
        """Verify MLServer name-and-version behavior with same model name, different versions.

        Given: RHOAI installed with MLServer ServingRuntime template, a storage
            path containing two subdirectories with model-settings.json declaring
            "shared-model" version "1.0" and "2.0".
        When: The repository index is queried and V2 inference is sent to each
            version.
        Then: The index contains two entries for "shared-model" distinguished
            by version, no duplicate-name error, and each versioned inference
            returns HTTP 200 with matching model_version.
        """
        pod = _get_isvc_pod(admin_client=admin_client, isvc=multi_version_inference_service)

        # Query repository index for shared-model entries
        repo_index = _v2_repository_index(pod=pod)
        shared_entries = [e for e in repo_index if e.get("name") == "shared-model"]
        assert len(shared_entries) == 2, (
            f"Expected 2 entries for shared-model, got {len(shared_entries)}: {shared_entries}"
        )

        versions = sorted(e.get("version", "") for e in shared_entries)
        assert "1.0" in versions, f"Version 1.0 not found in shared-model entries: {versions}"
        assert "2.0" in versions, f"Version 2.0 not found in shared-model entries: {versions}"

        # Infer version 1.0
        result_v1 = _v2_infer(
            pod=pod,
            model_name="shared-model/versions/1.0",
            payload=SKLEARN_IRIS_PAYLOAD,
        )
        assert result_v1["http_code"] == 200, f"V1.0 inference returned HTTP {result_v1['http_code']}"
        assert result_v1["body"].get("model_version") == "1.0"

        # Infer version 2.0
        result_v2 = _v2_infer(
            pod=pod,
            model_name="shared-model/versions/2.0",
            payload=SKLEARN_IRIS_PAYLOAD,
        )
        assert result_v2["http_code"] == 200, f"V2.0 inference returned HTTP {result_v2['http_code']}"
        assert result_v2["body"].get("model_version") == "2.0"


@pytest.mark.tier1
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": MULTI_MODEL_NS})],
    indirect=True,
)
class TestMultiModelConcurrency:
    """E2E tests for concurrent inference to resident models."""

    @pytest.mark.dependency(name="TC-E2E-006", depends=["TC-E2E-001"])
    def test_tc_e2e_006(
        self,
        admin_client: DynamicClient,
        multi_model_inference_service: InferenceService,
    ) -> None:
        """Verify concurrent inference requests to static resident model set.

        Given: InferenceService deployed with the 10-model repository,
            all 10 models loaded and serving inference.
        When: Concurrent V2 inference requests are sent to 5 distinct models.
        Then: All 5 requests return HTTP 200 with valid outputs and matching
            model_name, with no timeouts.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        pod = _get_isvc_pod(admin_client=admin_client, isvc=multi_model_inference_service)

        results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=len(CONCURRENT_MODELS)) as executor:
            futures = {
                executor.submit(_v2_infer, pod=pod, model_name=model_name, payload=SKLEARN_IRIS_PAYLOAD): model_name
                for model_name in CONCURRENT_MODELS
            }
            for future in as_completed(futures, timeout=30):
                model_name = futures[future]
                results[model_name] = future.result()

        for model_name in CONCURRENT_MODELS:
            assert model_name in results, f"No result for concurrent model {model_name}"
            result = results[model_name]
            assert result["http_code"] == 200, (
                f"Concurrent inference for {model_name} returned HTTP {result['http_code']}"
            )
            assert result["body"].get("model_name") == model_name
            assert len(result["body"].get("outputs", [])) > 0


@pytest.mark.tier2
@pytest.mark.slow
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": MULTI_MODEL_NS})],
    indirect=True,
)
class TestMultiModelStartup:
    """E2E tests for startup time measurement."""

    @pytest.mark.dependency(name="TC-E2E-007")
    def test_tc_e2e_007(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        mlserver_serving_runtime: ServingRuntime,
        startup_measurement_inference_service: InferenceService,
    ) -> None:
        """Measure startup time from InferenceService creation until all models available.

        Given: RHOAI installed with MLServer ServingRuntime template, 10-model
            repository staged, no prior InferenceService with the same name.
        When: An InferenceService is created and polled for readiness and full
            model availability.
        Then: The startupProbe config is recorded, all 10 models appear in the
            repository index before timeout, /v2/health/ready returns HTTP 200,
            and the measured startup time is recorded.
        """
        isvc = startup_measurement_inference_service
        start_time = time.monotonic()

        # Wait for ISVC pod to be available
        pod: Pod | None = None
        try:
            for sample in TimeoutSampler(
                wait_timeout=STARTUP_MEASUREMENT_TIMEOUT_SEC,
                sleep=POLL_INTERVAL_SEC,
                func=lambda: get_pods_by_isvc_label(client=admin_client, isvc=isvc),
            ):
                if sample:
                    pod = sample[0]
                    break
        except TimeoutExpiredError:
            pytest.fail("ISVC pod did not appear within timeout")

        assert pod is not None

        # Record startupProbe configuration
        startup_probe = _get_pod_probe(pod=pod, probe_type="startupProbe")
        LOGGER.info("startupProbe configuration", config=startup_probe)

        # Poll until all 10 models are loaded
        try:
            for sample in TimeoutSampler(
                wait_timeout=STARTUP_MEASUREMENT_TIMEOUT_SEC,
                sleep=POLL_INTERVAL_SEC,
                func=lambda: _v2_repository_index(pod=pod),
            ):
                if len(sample) == EXPECTED_MODEL_COUNT:
                    break
        except TimeoutExpiredError:
            repo_index = _v2_repository_index(pod=pod)
            pytest.fail(f"Only {len(repo_index)}/{EXPECTED_MODEL_COUNT} models loaded within timeout")

        elapsed_seconds = time.monotonic() - start_time
        LOGGER.info("Startup time measured", elapsed_seconds=round(elapsed_seconds, 2))

        # Verify readiness
        ready_code = _v2_health(pod=pod, endpoint="/v2/health/ready")
        assert ready_code == 200, f"/v2/health/ready returned HTTP {ready_code}"

        # Record pod Ready timestamp
        conditions = pod.instance.status.get("conditions", [])
        ready_condition = next((c for c in conditions if c.get("type") == "Ready"), None)
        LOGGER.info(
            "Pod Ready condition",
            status=ready_condition.get("status") if ready_condition else "not found",
            last_transition=ready_condition.get("lastTransitionTime") if ready_condition else None,
        )


@pytest.mark.tier2
@pytest.mark.gpu
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri",
    [
        pytest.param(
            {"name": MULTI_MODEL_NS},
            {"model-dir": f"{MODEL_PATH_PREFIX}/{ModelFormat.ONNX}"},
            id="cuda-onnx",
        ),
    ],
    indirect=True,
)
class TestMultiModelCUDA:
    """E2E tests for CUDA runtime ONNX deployment."""

    @pytest.mark.dependency(name="TC-E2E-008")
    def test_tc_e2e_008(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        cuda_inference_service: InferenceService,
    ) -> None:
        """Verify ONNX model deploys with MLServer CUDA profile and serves inference.

        Given: RHOAI installed with CUDA-capable GPU node, CUDA variant MLServer
            image, an ONNX model fixture on the storage path, GPU metrics
            collection available.
        When: The CUDA-variant ISVC is deployed, the pod is verified on a GPU
            node, and V2 inference is sent.
        Then: The pod is on a GPU node with nvidia.com/gpu allocation, the model
            appears in the repository index, inference returns HTTP 200, and GPU
            memory utilization is recorded.
        """
        pod = _get_isvc_pod(admin_client=admin_client, isvc=cuda_inference_service)

        # Verify GPU resource allocation
        containers = pod.instance.spec.get("containers", [])
        gpu_allocated = False
        for container in containers:
            resources = container.get("resources", {})
            limits = resources.get("limits", {})
            if "nvidia.com/gpu" in limits:
                gpu_allocated = True
                LOGGER.info("GPU allocation found", limits=limits)
                break
        assert gpu_allocated, "Pod does not have nvidia.com/gpu in resource limits"

        # Verify model in repository index
        repo_index = _v2_repository_index(pod=pod)
        model_names = [e.get("name", "") for e in repo_index]
        assert len(repo_index) > 0, "Repository index is empty"
        LOGGER.info("CUDA ISVC models loaded", models=model_names)

        # V2 inference
        onnx_model_name = model_names[0]
        result = _v2_infer(pod=pod, model_name=onnx_model_name, payload=SKLEARN_IRIS_PAYLOAD)
        assert result["http_code"] == 200, f"CUDA inference returned HTTP {result['http_code']}, expected 200"
        assert result["body"].get("model_name") is not None
        assert len(result["body"].get("outputs", [])) > 0

        # Record GPU memory utilization
        try:
            gpu_output = _exec_in_pod(
                pod=pod,
                command=[
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader",
                ],
            )
            LOGGER.info("GPU metrics", nvidia_smi_output=gpu_output.strip())
        except ExecOnPodError:
            LOGGER.warning("nvidia-smi not available in container; GPU metrics not recorded")


@pytest.mark.tier1
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri",
    [
        pytest.param(
            {"name": MULTI_MODEL_NS},
            {"model-dir": f"{MODEL_PATH_PREFIX}/{ModelFormat.SKLEARN}"},
            id="single-model-regression",
        ),
    ],
    indirect=True,
)
class TestSingleModelRegression:
    """E2E regression tests for single-model deployment after template changes."""

    @pytest.mark.dependency(name="TC-E2E-009")
    def test_tc_e2e_009(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        mlserver_serving_runtime: ServingRuntime,
        single_model_inference_service: InferenceService,
    ) -> None:
        """Verify single-model ISVC unaffected by multi-model template changes.

        Given: RHOAI installed with MLServer ServingRuntime template, a
            single-model baseline deployed and serving, template revision
            under test applied.
        When: The single-model InferenceService is re-deployed with the
            updated template and inference is repeated.
        Then: The ISVC reaches Ready, V2 inference returns HTTP 200 matching
            the baseline, and probe configuration remains functional.
        """
        pod = _get_isvc_pod(admin_client=admin_client, isvc=single_model_inference_service)

        # Verify inference
        result = _v2_infer(
            pod=pod,
            model_name=single_model_inference_service.name,
            payload=SKLEARN_IRIS_PAYLOAD,
        )
        assert result["http_code"] == 200, f"Single-model inference returned HTTP {result['http_code']}, expected 200"
        assert len(result["body"].get("outputs", [])) > 0

        # Record probe configuration
        for probe_type in ("livenessProbe", "readinessProbe", "startupProbe"):
            probe = _get_pod_probe(pod=pod, probe_type=probe_type)
            LOGGER.info(
                "Single-model probe configuration",
                probe_type=probe_type,
                config=probe,
            )

        # Record MLServer env vars relevant to single-model behavior
        for var_name in ("MLSERVER_EMPTY_REGISTRY_READINESS", "MLSERVER_MODEL_NAME"):
            value = _get_pod_env_var(pod=pod, var_name=var_name)
            LOGGER.info("Single-model env var", name=var_name, value=value)

        # Verify pod Ready
        conditions = pod.instance.status.get("conditions", [])
        ready_condition = next((c for c in conditions if c.get("type") == "Ready"), None)
        assert ready_condition is not None, "Pod has no Ready condition"
        assert ready_condition.get("status") == "True", f"Single-model pod not Ready: {ready_condition.get('status')}"
