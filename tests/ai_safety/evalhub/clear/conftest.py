"""Fixtures for IBM CLEAR adapter FVT (RHOAIENG-63558).

Overrides the shared ``evalhub_mt_cr`` fixture to inject ``MLFLOW_TRACKING_URI``
(with the ODH ``/mlflow`` static prefix) so CLEAR jobs exercise sidecar proxy
path-prefix behaviour from eval-hub#585.
"""

from __future__ import annotations

import base64
import uuid
from collections.abc import Generator
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.evalhub import EvalHub
from ocp_resources.mlflow import MLflow
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.route import Route
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.evalhub.clear.constants import (
    CLEAR_TRACE_FILENAME,
    CLEAR_TRACES_EXPERIMENT_NAME,
    CLEAR_TRACES_PVC_NAME,
    CLEAR_TRACES_PVC_SIZE,
    CLEAR_TRACES_SUB_PATH,
    IBM_CLEAR_PROVIDER_ID,
    IBM_CLEAR_PROVIDER_NAME,
    MLFLOW_SERVICE_PORT,
    MLFLOW_STATIC_PREFIX,
)
from tests.ai_safety.evalhub.clear.utils import (
    build_clear_provider_payload,
    ensure_mlflow_experiment,
    find_provider_id_by_resource_or_name,
    minimal_trace_json_bytes,
    search_mlflow_traces,
    seed_mlflow_trace_via_rest,
)
from tests.ai_safety.evalhub.constants import (
    EVALHUB_HEALTH_PATH,
    EVALHUB_PROVIDERS_PATH,
    EVALHUB_TENANT_LABEL_KEY,
    EVALHUB_TENANT_LABEL_VALUE,
    MINIO_MC_IMAGE,
    MINIO_UPLOADER_SECURITY_CONTEXT,
)
from tests.ai_safety.evalhub.utils import (
    MLflowWithWorkspaces,
    build_headers,
    is_evalhub_crd_available,
)

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="class")
def evalhub_mt_ready(
    evalhub_mt_route: Route,
    evalhub_mt_ca_bundle_file: str,
    network_policy_access_for_evalhub_api_sa: tuple[Role, RoleBinding],
) -> None:
    """Wait for the CLEAR EvalHub route to answer /health (router catch-up)."""
    _ = network_policy_access_for_evalhub_api_sa
    url = f"https://{evalhub_mt_route.host}{EVALHUB_HEALTH_PATH}"
    try:
        for sample in TimeoutSampler(
            wait_timeout=120,
            sleep=5,
            func=lambda: requests.get(url, verify=evalhub_mt_ca_bundle_file, timeout=10),
            exceptions_dict={Exception: []},
        ):
            if sample.ok:
                LOGGER.info(f"EvalHub at {evalhub_mt_route.host} is healthy")
                return
    except TimeoutExpiredError as err:
        raise RuntimeError(f"EvalHub at {evalhub_mt_route.host} did not become healthy within 120s") from err


@pytest.fixture(scope="class")
def clear_mlflow_instance(
    admin_client: DynamicClient,
) -> Generator[MLflow, Any, Any]:
    """Deploy MLflow for CLEAR experiment tracking and trace seeding."""
    with MLflowWithWorkspaces(
        client=admin_client,
        name="mlflow",
        storage={
            "accessModes": ["ReadWriteOnce"],
            "resources": {"requests": {"storage": "10Gi"}},
        },
        backend_store_uri="sqlite:////mlflow/mlflow.db",
        artifacts_destination="file:///mlflow/artifacts",
        serve_artifacts=True,
        workspace_label_selector={
            "matchLabels": {EVALHUB_TENANT_LABEL_KEY: EVALHUB_TENANT_LABEL_VALUE},
        },
        image={"imagePullPolicy": "Always"},
        wait_for_resource=True,
    ) as mlflow:
        mlflow_deployment = Deployment(
            client=admin_client,
            name="mlflow",
            namespace=py_config["applications_namespace"],
        )
        mlflow_deployment.wait_for_replicas(timeout=300)
        yield mlflow


@pytest.fixture(scope="class")
def clear_mlflow_url(clear_mlflow_instance: MLflow) -> str:
    """External MLflow URL (includes static prefix when present on status.url)."""
    url = clear_mlflow_instance.instance.status.url
    assert url, f"MLflow {clear_mlflow_instance.name} has no status.url"
    return url.rstrip("/")


@pytest.fixture(scope="class")
def evalhub_mt_cr(  # noqa: UFN001
    admin_client: DynamicClient,
    model_namespace: Namespace,
    clear_mlflow_instance: MLflow,
) -> Generator[EvalHub, Any, Any]:
    """EvalHub MT CR with MLFLOW_TRACKING_URI pointing at the /mlflow-prefixed service."""
    _ = clear_mlflow_instance
    if not is_evalhub_crd_available(admin_client):
        pytest.fail(
            "EvalHub CRD 'evalhubs.trustyai.opendatahub.io' not available on this cluster. "
            "Install the TrustyAI/EvalHub operator first."
        )

    apps_ns = py_config["applications_namespace"]
    mlflow_uri = f"https://mlflow.{apps_ns}.svc.cluster.local:{MLFLOW_SERVICE_PORT}{MLFLOW_STATIC_PREFIX}"
    with EvalHub(
        client=admin_client,
        name="evalhub-mt",
        namespace=model_namespace.name,
        database={"type": "sqlite"},
        collections=["leaderboard-v2"],
        env=[
            {"name": "MLFLOW_TRACKING_URI", "value": mlflow_uri},
        ],
        wait_for_resource=True,
    ) as evalhub:
        yield evalhub


@pytest.fixture(scope="class")
def clear_provider_id(
    evalhub_mt_ready: None,
    evalhub_mt_route: Route,
    evalhub_mt_ca_bundle_file: str,
    tenant_a_token: str,
    tenant_a_namespace: Namespace,
) -> Generator[str, Any, Any]:
    """Resolve system ``ibm-clear`` or create a user provider pointing at the CLEAR image."""
    _ = evalhub_mt_ready
    host = evalhub_mt_route.host
    headers = build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name)
    list_resp = requests.get(
        url=f"https://{host}{EVALHUB_PROVIDERS_PATH}?limit=100",
        headers=headers,
        verify=evalhub_mt_ca_bundle_file,
        timeout=30,
    )
    assert list_resp.status_code == 200, f"Failed to list providers: {list_resp.status_code} {list_resp.text}"
    existing = find_provider_id_by_resource_or_name(
        providers_response=list_resp.json(),
        resource_id=IBM_CLEAR_PROVIDER_ID,
        name=IBM_CLEAR_PROVIDER_NAME,
    )
    if existing is not None:
        LOGGER.info("Using existing CLEAR provider", provider_id=existing)
        yield existing
        return

    create_resp = requests.post(
        url=f"https://{host}{EVALHUB_PROVIDERS_PATH}",
        headers=headers,
        json=build_clear_provider_payload(),
        verify=evalhub_mt_ca_bundle_file,
        timeout=30,
    )
    assert create_resp.status_code == 201, (
        f"Failed to create CLEAR user provider: {create_resp.status_code} {create_resp.text}"
    )
    provider_id = create_resp.json()["resource"]["id"]
    LOGGER.info("Created CLEAR user provider", provider_id=provider_id)
    try:
        yield provider_id
    finally:
        requests.delete(
            url=f"https://{host}{EVALHUB_PROVIDERS_PATH}/{provider_id}",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=30,
        )


@pytest.fixture(scope="class")
def clear_traces_pvc(
    admin_client: DynamicClient,
    tenant_a_namespace: Namespace,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    """PVC used to stage CLEAR agent-trace JSON files."""
    with PersistentVolumeClaim(
        client=admin_client,
        name=CLEAR_TRACES_PVC_NAME,
        namespace=tenant_a_namespace.name,
        accessmodes=PersistentVolumeClaim.AccessMode.RWO,
        volume_mode=PersistentVolumeClaim.VolumeMode.FILE,
        size=CLEAR_TRACES_PVC_SIZE,
    ) as pvc:
        yield pvc


@pytest.fixture(scope="class")
def clear_traces_pvc_populated(
    admin_client: DynamicClient,
    tenant_a_namespace: Namespace,
    clear_traces_pvc: PersistentVolumeClaim,
) -> PersistentVolumeClaim:
    """Write the minimal agent-trace JSON onto the CLEAR traces PVC."""
    trace_b64 = base64.b64encode(minimal_trace_json_bytes()).decode("ascii")
    populate_script = (
        f"mkdir -p /data/{CLEAR_TRACES_SUB_PATH} && "
        f"echo '{trace_b64}' | base64 -d > /data/{CLEAR_TRACES_SUB_PATH}/{CLEAR_TRACE_FILENAME} && "
        "echo 'CLEAR trace population complete'"
    )
    pod_name = f"clear-trace-writer-{uuid.uuid4().hex[:8]}"
    with Pod(
        client=admin_client,
        name=pod_name,
        namespace=tenant_a_namespace.name,
        restart_policy="Never",
        security_context={"fsGroup": 1000, "seccompProfile": {"type": "RuntimeDefault"}},
        volumes=[{"name": "traces", "persistentVolumeClaim": {"claimName": clear_traces_pvc.name}}],
        containers=[
            {
                "name": "trace-writer",
                "image": MINIO_MC_IMAGE,
                "command": ["/bin/sh", "-c"],
                "args": [populate_script],
                "volumeMounts": [{"name": "traces", "mountPath": "/data"}],
                "securityContext": MINIO_UPLOADER_SECURITY_CONTEXT,
            }
        ],
    ) as writer:
        writer.wait_for_status(status=Pod.Status.SUCCEEDED, timeout=300)
    return clear_traces_pvc


@pytest.fixture(scope="class")
def clear_seeded_mlflow_traces_experiment(
    clear_mlflow_url: str,
    evalhub_mt_ca_bundle_file: str,
    tenant_a_token: str,
    tenant_a_namespace: Namespace,
) -> dict[str, str]:
    """Create the CLEAR input-traces experiment and seed one agent trace via REST.

    Returns:
        Dict with ``experiment_name``, ``experiment_id``, and ``trace_id``.
    """
    experiment_id = ensure_mlflow_experiment(
        mlflow_url=clear_mlflow_url,
        token=tenant_a_token,
        ca_bundle_file=evalhub_mt_ca_bundle_file,
        workspace=tenant_a_namespace.name,
        experiment_name=CLEAR_TRACES_EXPERIMENT_NAME,
    )
    trace_id = seed_mlflow_trace_via_rest(
        mlflow_url=clear_mlflow_url,
        token=tenant_a_token,
        ca_bundle_file=evalhub_mt_ca_bundle_file,
        workspace=tenant_a_namespace.name,
        experiment_id=experiment_id,
    )

    def _traces_visible() -> list[dict[str, Any]]:
        return search_mlflow_traces(
            mlflow_url=clear_mlflow_url,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            workspace=tenant_a_namespace.name,
            experiment_id=experiment_id,
        )

    try:
        for traces in TimeoutSampler(wait_timeout=60, sleep=5, func=_traces_visible):
            if traces:
                break
    except TimeoutExpiredError:
        LOGGER.warning(
            "Seeded MLflow trace not yet visible via search; CLEAR materialize may still succeed",
            experiment_id=experiment_id,
            trace_id=trace_id,
        )

    return {
        "experiment_name": CLEAR_TRACES_EXPERIMENT_NAME,
        "experiment_id": experiment_id,
        "trace_id": trace_id,
    }
