from __future__ import annotations

import pytest
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route

from tests.ai_safety.evalhub.evalcard.constants import EVALCARD_OCI_REPO, EVALCARD_OCI_TAG
from tests.ai_safety.evalhub.evalcard.utils import (
    get_evalcard_http,
    validate_evalcard_collection_fields,
    validate_evalcard_complete_mode,
    validate_evalcard_schema,
)
from tests.ai_safety.evalhub.utils import (
    build_evalhub_job_payload,
    submit_evalhub_job,
    validate_evalhub_job_completed,
    wait_for_evalhub_job,
)
from utilities.constants import OCIRegistry
from utilities.registry_utils import pull_manifest_from_oci_registry


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-evalcard"},
        ),
    ],
    indirect=True,
)
@pytest.mark.ai_safety
class TestEvalHubEvalCard:
    """EvalCard generation integration tests.

    Verifies that evaluation disclosure cards (evalcard.yaml) are generated
    and stored correctly for various job types.
    """

    def test_evalcard_single_task_run(
        self,
        evalcard_single_task_job: tuple[str, dict],
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Single-task job with MLflow experiment produces a valid eval card."""
        job_id, _ = evalcard_single_task_job

        resp = get_evalcard_http(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
        )
        assert resp.status_code == 200, f"Expected 200 for evalcard retrieval, got {resp.status_code}: {resp.text}"

        card = resp.json()
        validate_evalcard_schema(card=card, job_id=job_id)

        assert len(card["context"]["benchmarks"]) == 1, (
            f"Expected 1 benchmark in single-task card, got {len(card['context']['benchmarks'])}"
        )
        assert card["context"]["benchmarks"][0]["id"] == "arc_easy"

        assert len(card["results"]["benchmarks"]) == 1
        assert card["results"]["benchmarks"][0]["status"] == "completed"
        assert card["results"]["benchmarks"][0]["metrics"], "Benchmark metrics must not be empty"

        assert "collection" not in card["results"], "Single-task card should not have collection results"

    def test_evalcard_collection_run(
        self,
        evalcard_collection_job: tuple[str, dict],
        evalcard_collection_id: str,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Collection job produces an aggregated eval card with collection-level results."""
        job_id, _ = evalcard_collection_job

        resp = get_evalcard_http(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
        )
        assert resp.status_code == 200, f"Expected 200 for collection evalcard, got {resp.status_code}: {resp.text}"

        card = resp.json()
        validate_evalcard_schema(card=card, job_id=job_id)
        validate_evalcard_collection_fields(card=card, collection_id=evalcard_collection_id)

        assert len(card["results"]["benchmarks"]) >= 1
        for bench_result in card["results"]["benchmarks"]:
            assert bench_result["status"] == "completed"

    def test_evalcard_complete_mode(
        self,
        evalcard_complete_job: tuple[str, dict],
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Production job with pass_criteria and weights produces a complete card."""
        job_id, _ = evalcard_complete_job

        resp = get_evalcard_http(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
        )
        assert resp.status_code == 200, f"Expected 200 for complete evalcard, got {resp.status_code}: {resp.text}"

        card = resp.json()
        validate_evalcard_schema(card=card, job_id=job_id)
        validate_evalcard_complete_mode(card=card)

        context_bench = card["context"]["benchmarks"][0]
        assert context_bench.get("primary_score"), "Complete card benchmark must have primary_score in context"
        assert context_bench.get("pass_criteria"), "Complete card benchmark must have pass_criteria in context"
        assert "weight" in context_bench, "Complete card benchmark must have weight in context"

    def test_evalcard_schema_validation_non_blocking(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service,
    ) -> None:
        """Card generation failure does not block the evaluation job.

        Submits a job that should complete successfully regardless of card
        generation outcome, then verifies the job is in 'completed' state.
        """
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="evalcard-non-blocking",
        )
        payload["experiment"] = {"name": "evalcard-non-blocking-experiment"}

        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        job_id = data["resource"]["id"]

        job_result = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
            timeout=600,
        )
        validate_evalhub_job_completed(job_data=job_result)

        assert job_result["status"]["state"] == "completed", (
            "Job must complete successfully regardless of card generation outcome"
        )
        assert "card_error" not in job_result.get("status", {}), "Completed job status should not contain card_error"

    def test_evalcard_mlflow_retrieval(
        self,
        evalcard_single_task_job: tuple[str, dict],
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Eval card is stored as an MLflow artifact and retrievable.

        Verifies that the job response contains mlflow_experiment_id (indicating
        MLflow integration is active) and that the eval card can be retrieved
        via the EvalHub discovery API for the same job.
        """
        job_id, job_result = evalcard_single_task_job

        mlflow_experiment_id = job_result.get("resource", {}).get("mlflow_experiment_id")
        assert mlflow_experiment_id, (
            f"Expected mlflow_experiment_id in job response, got: {job_result.get('resource', {})}"
        )

        resp = get_evalcard_http(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
        )
        assert resp.status_code == 200, (
            f"Expected 200 for evalcard MLflow retrieval, got {resp.status_code}: {resp.text}"
        )

        card = resp.json()
        validate_evalcard_schema(card=card, job_id=job_id)

        has_mlflow_run = any(bench.get("mlflow_run_id") for bench in card["results"]["benchmarks"])
        assert has_mlflow_run, "At least one benchmark result should have mlflow_run_id when MLflow is configured"

    @pytest.mark.parametrize(
        "oci_registry_pod_with_minio",
        [pytest.param(OCIRegistry.PodConfig.REGISTRY_BASE_CONFIG)],
        indirect=True,
    )
    def test_evalcard_oci_export(
        self,
        evalcard_oci_job: tuple[str, dict],
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        oci_registry_host: str,
    ) -> None:
        """Eval card is exported to an OCI registry and retrievable as an OCI artifact."""
        job_id, _ = evalcard_oci_job

        registry_url = f"http://{oci_registry_host}"
        manifest = pull_manifest_from_oci_registry(
            registry_url=registry_url,
            repo=EVALCARD_OCI_REPO,
            tag=EVALCARD_OCI_TAG,
        )
        assert manifest, "OCI manifest must not be empty"
        assert "layers" in manifest, "OCI manifest must contain layers"
        assert len(manifest["layers"]) >= 1, "OCI manifest must have at least one layer"

        resp = get_evalcard_http(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
        )
        assert resp.status_code == 200, f"Expected 200 for OCI-exported evalcard, got {resp.status_code}: {resp.text}"
        card = resp.json()
        validate_evalcard_schema(card=card, job_id=job_id)
