import pytest
from kubernetes.client.rest import ApiException
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.pipelines_components.constants import (
    AUTORAG_INPUT_DATA_KEY,
    AUTORAG_PIPELINE_START_TIMEOUT,
    PIPELINE_POLL_INTERVAL,
)
from tests.pipelines_components.utils import (
    WORKFLOW_SUCCEEDED,
    collect_pipeline_pod_logs,
    get_pipeline_run,
    get_workflow_phase,
)


@pytest.mark.smoke
class TestAutoRAGSmoke:
    """Verify managed AutoRAG startup with synthetic data and placeholder service connections."""

    def test_autorag_pipeline_starts(
        self,
        admin_client: DynamicClient,
        autorag_run_id: str,
        autorag_managed_pipeline: dict[str, str],
        pipelines_namespace: Namespace,
        dspa_api_url: str,
        dspa_auth_headers: dict[str, str],
        dspa_ca_bundle_file: str,
    ) -> None:
        """Given synthetic documents and placeholder connections, when AutoRAG is submitted, then it starts."""
        run = get_pipeline_run(
            api_url=dspa_api_url,
            headers=dspa_auth_headers,
            run_id=autorag_run_id,
            ca_bundle=dspa_ca_bundle_file,
        )
        assert run["run_id"] == autorag_run_id, f"AutoRAG run {autorag_run_id} was not registered"
        pipeline_reference = run.get("pipeline_version_reference", {})
        assert pipeline_reference.get("pipeline_id") == autorag_managed_pipeline["pipeline_id"], (
            "AutoRAG run does not reference the managed pipeline"
        )
        assert pipeline_reference.get("pipeline_version_id") == autorag_managed_pipeline["pipeline_version_id"], (
            "AutoRAG run does not reference the managed pipeline version"
        )
        assert run.get("runtime_config", {}).get("parameters", {}).get("input_data_keys") == [AUTORAG_INPUT_DATA_KEY], (
            "AutoRAG run does not reference the uploaded documents"
        )

        last_phase: str | None = None
        try:
            for phase in TimeoutSampler(
                wait_timeout=AUTORAG_PIPELINE_START_TIMEOUT,
                sleep=PIPELINE_POLL_INTERVAL,
                func=get_workflow_phase,
                exceptions_dict={ApiException: [], ConnectionError: [], TimeoutError: []},
                admin_client=admin_client,
                namespace=pipelines_namespace.name,
                run_id=autorag_run_id,
            ):
                last_phase = phase
                if phase in ("Running", WORKFLOW_SUCCEEDED):
                    return
                if phase in ("Failed", "Error"):
                    collect_pipeline_pod_logs(
                        admin_client=admin_client,
                        namespace=pipelines_namespace.name,
                        run_id=autorag_run_id,
                    )
                    pytest.fail(f"AutoRAG run {autorag_run_id} failed before progressing: {phase}")
        except TimeoutExpiredError:
            pytest.fail(
                f"AutoRAG run {autorag_run_id} did not start within "
                f"{AUTORAG_PIPELINE_START_TIMEOUT}s (last workflow phase: {last_phase})"
            )
        pytest.fail(f"AutoRAG run {autorag_run_id} stopped being polled in phase {last_phase}")
