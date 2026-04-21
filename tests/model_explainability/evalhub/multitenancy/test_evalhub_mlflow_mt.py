"""MLflow integration tests for EvalHub (multitenancy suite).

Requires an MLflow server already deployed on the cluster (cluster-scoped CR).
The EvalHub CR is created with MLFLOW_TRACKING_URI pointing at the existing
MLflow service. Tests verify experiment creation on job submit, experiment_id
on the job response, and filtering jobs by experiment_id.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.evalhub import EvalHub
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.model_explainability.evalhub.constants import (
    EVALHUB_JOBS_PATH,
)
from tests.model_explainability.evalhub.utils import (
    build_evalhub_job_payload,
    build_headers,
    submit_evalhub_job,
)
from utilities.constants import Timeout

# ---------------------------------------------------------------------------
# Fixtures: EvalHub with MLflow enabled (expects existing MLflow deployment)
# ---------------------------------------------------------------------------

# MLflow is a cluster-scoped CR typically deployed in opendatahub namespace.
# These tests reuse the existing MLflow instance rather than creating one.
MLFLOW_NAMESPACE = "opendatahub"
MLFLOW_SERVICE_PORT = 8443


@pytest.fixture(scope="class")
def mlflow_deployment_ready(
    admin_client: DynamicClient,
) -> Deployment:
    """Verify the existing MLflow deployment is available. Skip if not deployed."""
    deployment = Deployment(
        client=admin_client,
        name="mlflow",
        namespace=MLFLOW_NAMESPACE,
    )
    if not deployment.exists:
        pytest.skip("MLflow deployment not found in opendatahub namespace — deploy MLflow first")
    deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
    return deployment


@pytest.fixture(scope="class")
def evalhub_mt_cr(  # noqa: UFN001
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlflow_deployment_ready: Deployment,
) -> Generator[EvalHub, Any, Any]:
    """Override the shared evalhub_mt_cr fixture to add MLflow tracking.

    Same name as the conftest fixture so pytest uses this one for the
    MLflow test class. Points MLFLOW_TRACKING_URI at the existing MLflow
    service in opendatahub.
    """
    mlflow_uri = f"https://mlflow.{MLFLOW_NAMESPACE}.svc.cluster.local:{MLFLOW_SERVICE_PORT}"
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


# ---------------------------------------------------------------------------
# Tests — use shared fixtures: evalhub_mt_route, evalhub_mt_ca_bundle_file,
# tenant_a_token, tenant_a_namespace, evalhub_vllm_emulator_service
# (all chain from evalhub_mt_mlflow_cr defined above)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-mlflow"},
        ),
    ],
    indirect=True,
)
@pytest.mark.model_explainability
class TestEvalHubMLflowIntegration:
    """MLflow experiment lifecycle via EvalHub job submission."""

    def test_job_with_experiment_creates_mlflow_experiment(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """POST job with experiment block → 202, response includes experiment_id."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="mlflow-experiment-test",
        )
        payload["experiment"] = {
            "name": "odh-fvt-mlflow-experiment",
            "tags": [{"key": "suite", "value": "mlflow-mt"}],
        }

        url = f"https://{evalhub_mt_route.host}{EVALHUB_JOBS_PATH}"
        headers = build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name)
        resp = requests.post(
            url=url,
            headers=headers,
            json=payload,
            verify=evalhub_mt_ca_bundle_file,
            timeout=30,
        )
        assert resp.status_code == 202, f"Expected 202 for job with experiment, got {resp.status_code}: {resp.text}"
        body = resp.json()
        exp_id = body.get("resource", {}).get("mlflow_experiment_id")
        assert exp_id, f"Expected resource.mlflow_experiment_id in response, got: {body}"

    def test_job_without_experiment_has_no_experiment_id(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """POST job without experiment block → 202, no experiment_id."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="mlflow-no-experiment-test",
        )

        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        exp_id = data.get("resource", {}).get("mlflow_experiment_id")
        assert not exp_id, f"Expected no mlflow_experiment_id when experiment block is omitted, got: {exp_id}"

    def test_job_with_duplicate_experiment_name_reuses_experiment_id(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """POST job with duplicate experiment name reuses the same experiment_id."""
        payload1 = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="mlflow-reuse-exp-1",
        )
        payload1["experiment"] = {"name": "odh-fvt-reuse-experiment"}

        payload2 = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="mlflow-reuse-exp-2",
        )
        payload2["experiment"] = {"name": "odh-fvt-reuse-experiment"}

        url = f"https://{evalhub_mt_route.host}{EVALHUB_JOBS_PATH}"
        headers = build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name)

        resp1 = requests.post(
            url=url,
            headers=headers,
            json=payload1,
            verify=evalhub_mt_ca_bundle_file,
            timeout=30,
        )
        assert resp1.status_code == 202
        exp_id_1 = resp1.json().get("resource", {}).get("mlflow_experiment_id")
        assert exp_id_1

        resp2 = requests.post(
            url=url,
            headers=headers,
            json=payload2,
            verify=evalhub_mt_ca_bundle_file,
            timeout=30,
        )
        assert resp2.status_code == 202
        exp_id_2 = resp2.json().get("resource", {}).get("mlflow_experiment_id")
        assert exp_id_2

        assert exp_id_1 == exp_id_2, f"Expected same experiment_id for same name, got {exp_id_1} vs {exp_id_2}"

    def test_list_jobs_filtered_by_experiment_id(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """Submit jobs with experiment, list filtered by experiment_id."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="mlflow-filter-test",
        )
        payload["experiment"] = {"name": "odh-fvt-filter-experiment"}

        url = f"https://{evalhub_mt_route.host}{EVALHUB_JOBS_PATH}"
        headers = build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name)

        resp = requests.post(
            url=url,
            headers=headers,
            json=payload,
            verify=evalhub_mt_ca_bundle_file,
            timeout=30,
        )
        assert resp.status_code == 202
        exp_id = resp.json().get("resource", {}).get("mlflow_experiment_id")
        assert exp_id

        # List jobs filtered by experiment_id
        list_resp = requests.get(
            url=f"{url}?experiment_id={exp_id}&limit=10",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert list_resp.status_code == 200
        body = list_resp.json()
        assert body["total_count"] >= 1
        for item in body["items"]:
            item_exp_id = item.get("resource", {}).get("mlflow_experiment_id")
            assert item_exp_id == exp_id, (
                f"Expected all filtered jobs to have mlflow_experiment_id={exp_id}, got {item_exp_id}"
            )
