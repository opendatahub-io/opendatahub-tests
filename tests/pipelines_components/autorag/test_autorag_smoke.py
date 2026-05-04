import uuid

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.secret import Secret

from tests.pipelines_components.constants import (
    AUTORAG_DSPA_NAMESPACE,
    AUTORAG_EMBEDDINGS_MODEL,
    AUTORAG_GENERATION_MODEL,
    AUTORAG_INPUT_DATA_KEY,
    AUTORAG_MAX_RAG_PATTERNS,
    AUTORAG_OPTIMIZATION_METRIC,
    AUTORAG_PIPELINE_TIMEOUT,
    AUTORAG_PIPELINE_YAML,
    AUTORAG_S3_BUCKET,
    AUTORAG_S3_SECRET_NAME,
    AUTORAG_TEST_DATA_KEY,
    AUTORAG_VECTOR_DB_ID,
)
from tests.pipelines_components.utils import (
    WORKFLOW_SUCCEEDED,
    collect_pipeline_pod_logs,
    create_pipeline_run,
    delete_pipeline,
    delete_pipeline_run,
    upload_pipeline,
    wait_for_pipeline_run,
)


@pytest.mark.smoke
class TestAutoRAGSmoke:
    """AutoRAG pipeline smoke test using the Documents RAG Optimization Pipeline.

    Requires a pre-existing DSPA with test data in S3 and an external Llama Stack server.
    See tests/pipelines_components/README.md for required environment variables.
    """

    def test_autorag_pipeline_completes(
        self,
        admin_client: DynamicClient,
        dspa_api_url: str,
        dspa_auth_headers: dict[str, str],
        dspa_ca_bundle_file: str,
        autorag_llama_stack_secret: Secret,
    ) -> None:
        """Given a DSPA with documents and benchmark data in S3, when an AutoRAG pipeline run is submitted,
        then it succeeds."""
        run_suffix = uuid.uuid4().hex[:8]
        pipeline_id = None
        run_id = None

        try:
            pipeline_id = upload_pipeline(
                api_url=dspa_api_url,
                headers=dspa_auth_headers,
                pipeline_yaml_path=AUTORAG_PIPELINE_YAML,
                pipeline_name=f"autorag-smoke-{run_suffix}",
                ca_bundle=dspa_ca_bundle_file,
            )

            parameters: dict = {
                "input_data_secret_name": AUTORAG_S3_SECRET_NAME,
                "input_data_bucket_name": AUTORAG_S3_BUCKET,
                "input_data_key": AUTORAG_INPUT_DATA_KEY,
                "test_data_secret_name": AUTORAG_S3_SECRET_NAME,
                "test_data_bucket_name": AUTORAG_S3_BUCKET,
                "test_data_key": AUTORAG_TEST_DATA_KEY,
                "llama_stack_secret_name": autorag_llama_stack_secret.name,
                "optimization_max_rag_patterns": AUTORAG_MAX_RAG_PATTERNS,
                "optimization_metric": AUTORAG_OPTIMIZATION_METRIC,
            }

            if AUTORAG_EMBEDDINGS_MODEL:
                parameters["embeddings_models"] = [AUTORAG_EMBEDDINGS_MODEL]
            if AUTORAG_GENERATION_MODEL:
                parameters["generation_models"] = [AUTORAG_GENERATION_MODEL]
            if AUTORAG_VECTOR_DB_ID:
                parameters["llama_stack_vector_database_id"] = AUTORAG_VECTOR_DB_ID

            run_id = create_pipeline_run(
                api_url=dspa_api_url,
                headers=dspa_auth_headers,
                pipeline_id=pipeline_id,
                run_name=f"autorag-smoke-{run_suffix}",
                parameters=parameters,
                ca_bundle=dspa_ca_bundle_file,
            )

            try:
                phase = wait_for_pipeline_run(
                    admin_client=admin_client,
                    namespace=AUTORAG_DSPA_NAMESPACE,
                    run_id=run_id,
                    timeout=AUTORAG_PIPELINE_TIMEOUT,
                )
            except TimeoutError:
                collect_pipeline_pod_logs(
                    admin_client=admin_client,
                    namespace=AUTORAG_DSPA_NAMESPACE,
                    run_id=run_id,
                )
                raise

            if phase != WORKFLOW_SUCCEEDED:
                collect_pipeline_pod_logs(
                    admin_client=admin_client,
                    namespace=AUTORAG_DSPA_NAMESPACE,
                    run_id=run_id,
                )

            assert phase == WORKFLOW_SUCCEEDED, (
                f"AutoRAG pipeline run {run_id} ended with phase '{phase}', expected '{WORKFLOW_SUCCEEDED}'"
            )
        finally:
            if run_id:
                delete_pipeline_run(
                    api_url=dspa_api_url,
                    headers=dspa_auth_headers,
                    run_id=run_id,
                    ca_bundle=dspa_ca_bundle_file,
                )
            if pipeline_id:
                delete_pipeline(
                    api_url=dspa_api_url,
                    headers=dspa_auth_headers,
                    pipeline_id=pipeline_id,
                    ca_bundle=dspa_ca_bundle_file,
                )
