"""Utilities for KServe canary rollout (RawDeployment) tests."""

from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.resource import ResourceEditor
from timeout_sampler import TimeoutSampler

from tests.model_serving.model_runtime.mlserver.constant import PREDICT_RESOURCES
from tests.model_serving.model_server.kserve.canary_rollout.constants import (
    TRAFFIC_INFERENCE_INPUT,
    TRAFFIC_SAMPLE_SIZE,
    TRAFFIC_TOLERANCE_PERCENT,
    V2_INFER_PATH_TEMPLATE,
)
from utilities.constants import Annotations, KServeDeploymentType, Labels, Timeout
from utilities.inference_utils import Inference
from utilities.infra import get_model_route, verify_no_failed_pods, wait_for_inference_deployment_replicas

LOGGER = structlog.get_logger(name=__name__)

# HTTP fingerprint: stable sklearn 1.0 → 200, canary mixedtype → 500 for TRAFFIC_INFERENCE_INPUT.
_STABLE_STATUS = 200
_CANARY_STATUS = 500


def build_predictor_spec(
    *,
    model_format: str,
    runtime: str,
    storage_uri: str,
    min_replicas: int = 1,
    model_service_account: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Build a KServe predictor spec fragment."""
    predictor: dict[str, Any] = {
        "model": {
            "modelFormat": {"name": model_format},
            "runtime": runtime,
            "storageUri": storage_uri,
            "resources": PREDICT_RESOURCES["resources"],
        },
        "minReplicas": min_replicas,
    }
    # Canary webhook requires predictor.name; Deployments become {isvc}-{name}-predictor.
    if name:
        predictor["name"] = name
    if model_service_account:
        predictor["serviceAccountName"] = model_service_account
    return predictor


def build_canary_entry(
    *,
    model_format: str,
    runtime: str,
    storage_uri: str,
    canary_traffic_percent: int,
    min_replicas: int = 1,
    model_service_account: str | None = None,
    predictor_name: str = "canary",
) -> dict[str, Any]:
    """Build one canary array entry matching canary-spec-status API."""
    return {
        "predictor": build_predictor_spec(
            model_format=model_format,
            runtime=runtime,
            storage_uri=storage_uri,
            min_replicas=min_replicas,
            model_service_account=model_service_account,
            name=predictor_name,
        ),
        "trafficPercent": canary_traffic_percent,
    }


def deployment_contains_storage_uri(deployment: Deployment, storage_uri: str) -> bool:
    """Return True if the deployment pod spec references the storage URI."""
    deployment_json = json.dumps(deployment.instance.to_dict())
    return storage_uri in deployment_json


def get_isvc_deployments(
    client: DynamicClient,
    isvc: InferenceService,
    runtime_name: str,
    expected_count: int,
) -> list[Deployment]:
    """Return InferenceService predictor deployments after waiting for the expected count."""
    return wait_for_inference_deployment_replicas(
        client=client,
        isvc=isvc,
        runtime_name=runtime_name,
        expected_num_deployments=expected_count,
        timeout=Timeout.TIMEOUT_15MIN,
    )


def assert_route_traffic_weights(
    isvc: InferenceService,
    *,
    stable_weight: int,
    canary_weight: int,
) -> None:
    """Assert OpenShift Route primary and alternate backend weights."""
    route = get_model_route(client=isvc.client, isvc=isvc)
    route_spec = route.instance.spec
    primary_weight = route_spec["to"]["weight"]
    alternate_backends = route_spec.get("alternateBackends") or []

    assert primary_weight == stable_weight, f"Expected stable Route weight {stable_weight}, got {primary_weight}"
    assert alternate_backends, "Expected alternateBackends on Route for canary traffic split"
    assert alternate_backends[0]["weight"] == canary_weight, (
        f"Expected canary Route weight {canary_weight}, got {alternate_backends[0]['weight']}"
    )


def assert_canary_traffic_by_status_codes(
    isvc: InferenceService,
    *,
    expected_percent: int,
    sample_size: int = TRAFFIC_SAMPLE_SIZE,
    tolerance_percent: int = TRAFFIC_TOLERANCE_PERCENT,
    model_name: str | None = None,
) -> None:
    """Assert Route traffic split by HTTP status fingerprint (200=stable, 500=canary)."""
    inference = Inference(inference_service=isvc)
    host = inference.get_inference_url()
    predict_model = model_name or isvc.name
    rest_url = f"https://{host}{V2_INFER_PATH_TEMPLATE.format(model_name=predict_model)}"

    stable_hits = 0
    canary_hits = 0
    other_statuses: dict[int, int] = {}

    for _ in range(sample_size):
        response = requests.post(url=rest_url, json=TRAFFIC_INFERENCE_INPUT, verify=False, timeout=60)
        status = response.status_code
        if status == _STABLE_STATUS:
            stable_hits += 1
        elif status == _CANARY_STATUS:
            canary_hits += 1
        else:
            other_statuses[status] = other_statuses.get(status, 0) + 1
        LOGGER.debug("canary traffic sample", status=status, url=rest_url)

    total = stable_hits + canary_hits
    LOGGER.info(
        "canary traffic by status",
        stable_200=stable_hits,
        canary_500=canary_hits,
        total=total,
        other=other_statuses,
        url=rest_url,
    )

    assert not other_statuses, (
        f"Unexpected HTTP statuses while sampling canary traffic via {rest_url}: {other_statuses}"
    )
    assert total == sample_size, (
        f"Expected {sample_size} classifiable responses, got {total} (stable={stable_hits}, canary={canary_hits})"
    )
    observed_percent = (canary_hits / total) * 100
    lower_bound = expected_percent - tolerance_percent
    upper_bound = expected_percent + tolerance_percent
    assert lower_bound <= observed_percent <= upper_bound, (
        f"Canary traffic {observed_percent:.1f}% outside expected "
        f"{expected_percent}% +/- {tolerance_percent}% "
        f"(stable_200={stable_hits}, canary_500={canary_hits})"
    )


def wait_for_canary_ready_condition(isvc: InferenceService, timeout: int = Timeout.TIMEOUT_15MIN) -> None:
    """Wait until a canary-related Ready condition is True, if present."""

    def _canary_ready() -> bool:
        conditions = isvc.instance.status.get("conditions") or []
        for condition in conditions:
            condition_type = condition.get("type", "")
            if "canary" in condition_type.lower() and condition.get("status") == "True":
                return True
        return isvc.instance.status.get("ready", False) is True

    for ready in TimeoutSampler(wait_timeout=timeout, sleep=5, func=_canary_ready):
        if ready:
            return

    raise TimeoutError(f"InferenceService {isvc.name} canary readiness condition not True within {timeout}s")


@contextmanager
def create_canary_inference_service(
    *,
    client: DynamicClient,
    name: str,
    namespace: str,
    runtime: str,
    stable_model_format: str,
    stable_storage_uri: str,
    canary_model_format: str,
    canary_storage_uri: str,
    canary_traffic_percent: int,
    deployment_mode: str = KServeDeploymentType.STANDARD,
    external_route: bool = True,
    model_service_account: str | None = None,
    teardown: bool = True,
    timeout: int = Timeout.TIMEOUT_15MIN,
) -> Generator[InferenceService]:
    """Create a RawDeployment InferenceService with a canary array entry."""
    labels: dict[str, str] = {}
    if external_route and deployment_mode in KServeDeploymentType.RAW_DEPLOYMENT_MODES:
        labels[Labels.Kserve.NETWORKING_KSERVE_IO] = Labels.Kserve.EXPOSED

    annotations = {Annotations.KserveIo.DEPLOYMENT_MODE: deployment_mode}
    predictor = build_predictor_spec(
        model_format=stable_model_format,
        runtime=runtime,
        storage_uri=stable_storage_uri,
        model_service_account=model_service_account,
    )
    canary_entry = build_canary_entry(
        model_format=canary_model_format,
        runtime=runtime,
        storage_uri=canary_storage_uri,
        canary_traffic_percent=canary_traffic_percent,
        model_service_account=model_service_account,
    )

    with (
        InferenceService(
            client=client,
            name=name,
            namespace=namespace,
            annotations=annotations,
            label=labels,
            predictor=predictor,
            teardown=teardown,
        ) as isvc,
        ResourceEditor(patches={isvc: {"spec": {"canary": [canary_entry]}}}),
    ):
        verify_no_failed_pods(
            client=client,
            isvc=isvc,
            runtime_name=runtime,
            timeout=timeout,
        )
        wait_for_inference_deployment_replicas(
            client=client,
            isvc=isvc,
            runtime_name=runtime,
            expected_num_deployments=2,
            timeout=timeout,
        )
        isvc.wait_for_condition(
            condition=isvc.Condition.READY,
            status=isvc.Condition.Status.TRUE,
            timeout=timeout,
        )
        yield isvc


def promote_canary_to_stable(
    isvc: InferenceService,
    *,
    promoted_storage_uri: str,
    runtime: str,
    model_format: str,
    timeout: int = Timeout.TIMEOUT_15MIN,
) -> None:
    """Promote canary by replacing stable storage and clearing the canary array.

    Uses a permanent patch (not a ResourceEditor context manager) so promotion is
    not undone when the helper returns.
    """
    promoted_predictor = build_predictor_spec(
        model_format=model_format,
        runtime=runtime,
        storage_uri=promoted_storage_uri,
    )
    ResourceEditor(
        patches={
            isvc: {
                "spec": {
                    "predictor": promoted_predictor,
                    "canary": [],
                },
            },
        },
    ).update()
    wait_for_inference_deployment_replicas(
        client=isvc.client,
        isvc=isvc,
        runtime_name=runtime,
        expected_num_deployments=1,
        timeout=timeout,
    )
