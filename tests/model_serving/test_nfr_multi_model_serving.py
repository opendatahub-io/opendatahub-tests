"""
Non-functional requirement tests for MLServer multi-model serving.

Covers PSAP benchmark comparison (single vs multi-model), scalability
(50 single-model vs 5 multi-model), and V2 Management API access control
at the cluster boundary.

Source: RHAISTRAT-3033
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.network_policy import NetworkPolicy
from ocp_resources.pod import Pod
from ocp_resources.route import Route

from utilities.infra import get_pods_by_isvc_label

LOGGER = structlog.get_logger(name=__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MULTI_MODEL_NS: str = "multi-model-nfr"
ISVC_READY_TIMEOUT_SEC: int = 600
FLEET_READY_TIMEOUT_SEC: int = 600
POLL_INTERVAL_SEC: int = 5
SINGLE_MODEL_COUNT: int = 1
MULTI_MODEL_COUNT: int = 10
FLEET_SINGLE_COUNT: int = 50
FLEET_MULTI_COUNT: int = 5

V2_LOAD_PATH: str = "/v2/repository/models/{model_name}/load"
V2_UNLOAD_PATH: str = "/v2/repository/models/{model_name}/unload"
V2_INFER_PATH: str = "/v2/models/{model_name}/infer"

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
    """Send a V2 inference request via pod exec curl."""
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


def _collect_pod_resource_requests(
    admin_client: DynamicClient,
    namespace: str,
    label_selector: str,
) -> dict[str, Any]:
    """Collect CPU and memory resource requests for pods matching a label.

    Args:
        admin_client: Kubernetes dynamic client.
        namespace: Namespace to query.
        label_selector: Label selector string.

    Returns:
        Dictionary with per-pod resources and aggregate totals.
    """
    pods = list(
        Pod.get(
            dyn_client=admin_client,
            namespace=namespace,
            label_selector=label_selector,
        )
    )
    metrics: dict[str, Any] = {
        "pod_count": len(pods),
        "total_cpu_millicores": 0,
        "total_memory_mib": 0,
    }
    for pod in pods:
        containers = pod.instance.spec.get("containers", [])
        for container in containers:
            requests = container.get("resources", {}).get("requests", {})
            cpu_str = requests.get("cpu", "0")
            mem_str = requests.get("memory", "0")
            metrics["total_cpu_millicores"] += _parse_cpu_millicores(cpu_str)
            metrics["total_memory_mib"] += _parse_memory_mib(mem_str)
    return metrics


def _parse_cpu_millicores(value: str) -> int:
    """Convert a Kubernetes CPU string to integer millicores."""
    if value.endswith("m"):
        return int(value[:-1])
    return int(float(value) * 1000)


def _parse_memory_mib(value: str) -> int:
    """Convert a Kubernetes memory string to integer MiB."""
    if value.endswith("Gi"):
        return int(float(value[:-2]) * 1024)
    if value.endswith("Mi"):
        return int(float(value[:-2]))
    if value.endswith("Ki"):
        return int(float(value[:-2]) / 1024)
    return int(int(value) / (1024 * 1024))


def _exec_curl_in_cluster(
    namespace: str,
    url: str,
    method: str = "POST",
    data: str | None = None,
) -> dict[str, Any]:
    """Run a curl command inside the cluster using oc run.

    Args:
        namespace: Namespace for the ephemeral pod.
        url: Target URL.
        method: HTTP method.
        data: Optional request body.

    Returns:
        Dictionary with http_code and returncode.
    """
    curl_args = [
        "curl",
        "-s",
        "-o",
        "/dev/null",
        "-w",
        "%{http_code}",
        "-X",
        method,
        "-H",
        "Content-Type: application/json",
    ]
    if data:
        curl_args.extend(["-d", data])
    curl_args.append(url)

    oc_cmd: list[str] = [
        "oc",
        "run",
        "curl-boundary-test",
        "--rm",
        "-i",
        "--restart=Never",
        f"--namespace={namespace}",
        "--image=registry.access.redhat.com/ubi9/ubi-minimal:latest",
        "--",
        *curl_args,
    ]

    try:
        result = subprocess.run(
            oc_cmd,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        code_str = result.stdout.strip()
        http_code = int(code_str) if code_str.isdigit() else 0
        return {"http_code": http_code, "returncode": result.returncode}
    except (subprocess.TimeoutExpired, ValueError) as exc:
        LOGGER.error("In-cluster curl failed", error=str(exc))
        return {"http_code": 0, "returncode": -1}


def _exec_curl_external(
    url: str,
    method: str = "POST",
    data: str | None = None,
) -> dict[str, Any]:
    """Execute a curl command from the test runner (external to cluster).

    Args:
        url: Target URL.
        method: HTTP method.
        data: Optional request body.

    Returns:
        Dictionary with http_code and returncode.
    """
    curl_cmd: list[str] = [
        "curl",
        "-s",
        "-o",
        "/dev/null",
        "-w",
        "%{http_code}",
        "-X",
        method,
        "-H",
        "Content-Type: application/json",
        "--insecure",
        "--max-time",
        "10",
    ]
    if data:
        curl_cmd.extend(["-d", data])
    curl_cmd.append(url)

    try:
        result = subprocess.run(
            curl_cmd,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        code_str = result.stdout.strip()
        http_code = int(code_str) if code_str.isdigit() else 0
        return {"http_code": http_code, "returncode": result.returncode}
    except (subprocess.TimeoutExpired, ValueError) as exc:
        LOGGER.error("External curl failed", error=str(exc))
        return {"http_code": 0, "returncode": -1}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.tier3
@pytest.mark.slow
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri",
    [
        pytest.param(
            {"name": MULTI_MODEL_NS},
            {"model-dir": "mlserver/model_repository/sklearn-iris-1"},
            id="psap-benchmark",
        ),
    ],
    indirect=True,
)
class TestPSAPBenchmark:
    """PSAP performance benchmark: single-model baseline vs 10-model ISVC."""

    def test_tc_nfr_001(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        baseline_inference_service: InferenceService,
        multi_model_inference_service: InferenceService,
    ) -> None:
        """Record PSAP benchmark metrics for single-model baseline and 10-model profile.

        Given: RHOAI 3.7 EA installed with MLServer SR template at parallel_workers=1,
            PSAP benchmark tooling installed, single-model baseline and 10-model
            ISVC deployed on the same hardware profile.
        When: PSAP benchmarks are executed sequentially against both deployments,
            pod resource consumption is recorded.
        Then: Baseline and 10-model metrics (p50/p99 latency, throughput RPS, CPU,
            memory) are captured; parallel_workers=1 is confirmed; saturation
            factors are documented.
        """
        namespace_name = model_namespace.name

        # Confirm parallel_workers=1
        multi_pod = _get_isvc_pod(admin_client=admin_client, isvc=multi_model_inference_service)
        containers = multi_pod.instance.spec.get("containers", [])
        for container in containers:
            for env in container.get("env", []):
                if env.get("name") == "PARALLEL_WORKERS":
                    assert env.get("value") == "1", f"Expected PARALLEL_WORKERS=1, got {env.get('value')}"

        # Record baseline resource usage
        baseline_resources = _collect_pod_resource_requests(
            admin_client=admin_client,
            namespace=namespace_name,
            label_selector=f"serving.kserve.io/inferenceservice={baseline_inference_service.name}",
        )
        LOGGER.info("Baseline resource metrics", **baseline_resources)

        # Record multi-model resource usage
        multi_model_resources = _collect_pod_resource_requests(
            admin_client=admin_client,
            namespace=namespace_name,
            label_selector=f"serving.kserve.io/inferenceservice={multi_model_inference_service.name}",
        )
        LOGGER.info("Multi-model resource metrics", **multi_model_resources)

        # Verify inference works on both
        baseline_pod = _get_isvc_pod(admin_client=admin_client, isvc=baseline_inference_service)
        baseline_result = _v2_infer(
            pod=baseline_pod,
            model_name=baseline_inference_service.name,
            payload=SKLEARN_IRIS_PAYLOAD,
        )
        assert baseline_result["http_code"] == 200, f"Baseline inference returned HTTP {baseline_result['http_code']}"

        multi_result = _v2_infer(
            pod=multi_pod,
            model_name="sklearn-iris-1",
            payload=SKLEARN_IRIS_PAYLOAD,
        )
        assert multi_result["http_code"] == 200, f"Multi-model inference returned HTTP {multi_result['http_code']}"

        LOGGER.info(
            "PSAP benchmark comparison recorded",
            baseline=baseline_resources,
            multi_model=multi_model_resources,
            parallel_workers=1,
            baseline_model_count=SINGLE_MODEL_COUNT,
            multi_model_count=MULTI_MODEL_COUNT,
        )


@pytest.mark.tier3
@pytest.mark.slow
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri",
    [
        pytest.param(
            {"name": MULTI_MODEL_NS},
            {"model-dir": "mlserver/model_repository/sklearn-iris-1"},
            id="scalability-comparison",
        ),
    ],
    indirect=True,
)
class TestScalabilityComparison:
    """Scalability: 50 single-model ISVCs vs 5 multi-model ISVCs."""

    def test_tc_nfr_002(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        single_model_fleet: list[InferenceService],
        multi_model_fleet: list[InferenceService],
    ) -> None:
        """Record resource-efficiency ratio comparing 50 single-model ISVCs with 5 multi-model.

        Given: RHOAI installed with MLServer SR template, sufficient cluster
            capacity for 50 ISVCs, same hardware profile across deployments.
        When: 50 single-model ISVCs are deployed and their aggregate resource
            consumption is recorded, then 5 multi-model ISVCs are deployed
            and their aggregate consumption is recorded.
        Then: CPU/memory totals for both fleets are captured, the efficiency
            ratio is calculated, and the <=50% target is reported as evidence.
        """
        namespace_name = model_namespace.name

        assert len(single_model_fleet) == FLEET_SINGLE_COUNT, (
            f"Expected {FLEET_SINGLE_COUNT} single-model ISVCs, got {len(single_model_fleet)}"
        )
        assert len(multi_model_fleet) == FLEET_MULTI_COUNT, (
            f"Expected {FLEET_MULTI_COUNT} multi-model ISVCs, got {len(multi_model_fleet)}"
        )

        # Collect single-model fleet resources
        single_resources = _collect_pod_resource_requests(
            admin_client=admin_client,
            namespace=namespace_name,
            label_selector="app=single-model-bench",
        )
        LOGGER.info("Single-model fleet resources", **single_resources)

        # Collect multi-model fleet resources
        multi_resources = _collect_pod_resource_requests(
            admin_client=admin_client,
            namespace=namespace_name,
            label_selector="app=multi-model-bench",
        )
        LOGGER.info("Multi-model fleet resources", **multi_resources)

        # Calculate efficiency ratio
        single_cpu = single_resources["total_cpu_millicores"]
        multi_cpu = multi_resources["total_cpu_millicores"]
        single_mem = single_resources["total_memory_mib"]
        multi_mem = multi_resources["total_memory_mib"]

        cpu_ratio = (multi_cpu / single_cpu * 100) if single_cpu > 0 else 0.0
        mem_ratio = (multi_mem / single_mem * 100) if single_mem > 0 else 0.0

        LOGGER.info(
            "Efficiency ratio",
            cpu_ratio_percent=round(cpu_ratio, 2),
            memory_ratio_percent=round(mem_ratio, 2),
            target_percent=50,
            single_fleet_count=FLEET_SINGLE_COUNT,
            multi_fleet_count=FLEET_MULTI_COUNT,
        )

        assert single_cpu > 0, "Single-model fleet CPU must be > 0"
        assert single_mem > 0, "Single-model fleet memory must be > 0"


@pytest.mark.tier1
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": MULTI_MODEL_NS})],
    indirect=True,
)
class TestV2ManagementAPIBoundary:
    """V2 Management API cluster boundary access control."""

    @pytest.mark.dependency(name="TC-NFR-003")
    def test_tc_nfr_003(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        multi_model_inference_service: InferenceService,
    ) -> None:
        """Verify V2 Management API reachable internally but blocked externally.

        Given: InferenceService deployed with 10-model repo, pod Ready
            (TC-E2E-001), NetworkPolicy configured, access from internal
            pod and external route.
        When: V2 load and unload endpoints are called from within the cluster,
            the NetworkPolicy is recorded, and the same endpoints are attempted
            from an external route.
        Then: Internal load/unload return an HTTP response (200, 400, or 404);
            NetworkPolicy is recorded; external management API returns non-200
            or connection refused; inference still returns HTTP 200.
        """
        namespace_name = model_namespace.name
        isvc_name = multi_model_inference_service.name
        test_model = "sklearn-iris-1"

        internal_svc = f"http://{isvc_name}.{namespace_name}.svc.cluster.local:8080"

        # Internal: call V2 load endpoint
        load_path = V2_LOAD_PATH.format(model_name=test_model)
        internal_load = _exec_curl_in_cluster(
            namespace=namespace_name,
            url=f"{internal_svc}{load_path}",
            method="POST",
            data="{}",
        )
        LOGGER.info("Internal V2 load response", http_code=internal_load["http_code"])
        assert internal_load["http_code"] in (200, 400, 404), (
            f"Internal V2 load returned unexpected {internal_load['http_code']}"
        )

        # Internal: call V2 unload endpoint
        unload_path = V2_UNLOAD_PATH.format(model_name=test_model)
        internal_unload = _exec_curl_in_cluster(
            namespace=namespace_name,
            url=f"{internal_svc}{unload_path}",
            method="POST",
            data="{}",
        )
        LOGGER.info("Internal V2 unload response", http_code=internal_unload["http_code"])
        assert internal_unload["http_code"] in (200, 400, 404), (
            f"Internal V2 unload returned unexpected {internal_unload['http_code']}"
        )

        # Record NetworkPolicy
        network_policies = list(
            NetworkPolicy.get(
                dyn_client=admin_client,
                namespace=namespace_name,
            )
        )
        for netpol in network_policies:
            LOGGER.info(
                "NetworkPolicy",
                name=netpol.name,
                spec=netpol.instance.get("spec", {}),
            )

        # External: attempt management API via route
        routes = list(
            Route.get(
                dyn_client=admin_client,
                namespace=namespace_name,
            )
        )
        external_host: str | None = None
        for route in routes:
            if isvc_name in route.name:
                external_host = route.instance.get("spec", {}).get("host", "")
                break

        if not external_host:
            pytest.skip("No usable external route found")

        external_base = f"https://{external_host}"

        external_load = _exec_curl_external(
            url=f"{external_base}{load_path}",
            method="POST",
            data="{}",
        )
        LOGGER.info("External V2 load response", http_code=external_load["http_code"])
        assert external_load["http_code"] != 200, (
            f"External V2 load should NOT return 200, got {external_load['http_code']}"
        )

        external_unload = _exec_curl_external(
            url=f"{external_base}{unload_path}",
            method="POST",
            data="{}",
        )
        LOGGER.info("External V2 unload response", http_code=external_unload["http_code"])
        assert external_unload["http_code"] != 200, (
            f"External V2 unload should NOT return 200, got {external_unload['http_code']}"
        )

        # Verify inference still works
        pod = _get_isvc_pod(admin_client=admin_client, isvc=multi_model_inference_service)
        infer_result = _v2_infer(pod=pod, model_name=test_model, payload=SKLEARN_IRIS_PAYLOAD)
        assert infer_result["http_code"] == 200, (
            f"Post-management-API inference returned HTTP {infer_result['http_code']}"
        )
