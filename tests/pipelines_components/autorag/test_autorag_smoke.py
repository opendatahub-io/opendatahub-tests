import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace

from tests.pipelines_components.constants import (
    AUTORAG_PIPELINE_TIMEOUT,
)
from tests.pipelines_components.utils import (
    WORKFLOW_SUCCEEDED,
    collect_pipeline_pod_logs,
    wait_for_pipeline_run,
)


@pytest.mark.smoke
class TestAutoRAGSmoke:
    """AutoRAG pipeline smoke test using the Documents RAG Optimization Pipeline.

    Deploys a self-contained DSPA with built-in MinIO, vLLM models, and OGX server,
    then runs the AutoRAG pipeline and verifies it completes successfully.
    """

    def test_autorag_pipeline_completes(
        self,
        admin_client: DynamicClient,
        autorag_run_id: str,
        pipelines_namespace: Namespace,
    ) -> None:
        """Given a DSPA with documents and benchmark data, when an AutoRAG pipeline run is submitted,
        then it succeeds."""
        phase = wait_for_pipeline_run(
            admin_client=admin_client,
            namespace=pipelines_namespace.name,
            run_id=autorag_run_id,
            timeout=AUTORAG_PIPELINE_TIMEOUT,
        )

        if phase != WORKFLOW_SUCCEEDED:
            collect_pipeline_pod_logs(
                admin_client=admin_client,
                namespace=pipelines_namespace.name,
                run_id=autorag_run_id,
            )

        assert phase == WORKFLOW_SUCCEEDED, (
            f"AutoRAG pipeline run {autorag_run_id} ended with phase '{phase}', expected '{WORKFLOW_SUCCEEDED}'"
        )
