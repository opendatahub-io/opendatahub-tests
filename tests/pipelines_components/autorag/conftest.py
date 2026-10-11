import os
import shlex
import uuid
from collections.abc import Generator
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_pipelines_application import DataSciencePipelinesApplication
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from timeout_sampler import TimeoutExpiredError

from tests.pipelines_components.constants import (
    AUTORAG_INPUT_DATA_KEY,
    AUTORAG_MAX_RAG_PATTERNS,
    AUTORAG_OPTIMIZATION_METRIC,
    AUTORAG_S3_BUCKET,
    AUTORAG_TEST_DATA_KEY,
    DSPA_NAME,
    DSPA_READY_BUFFER_SECONDS,
    DSPA_S3_BUCKET,
    DSPA_S3_SECRET,
    EXTERNAL_S3_SECRET,
    MANAGED_PIPELINE_AUTORAG,
    MANAGED_PIPELINE_POLL_INTERVAL,
    MANAGED_PIPELINE_WAIT_TIMEOUT,
    MINIO_MC_IMAGE,
    MINIO_UPLOADER_SECURITY_CONTEXT,
)
from tests.pipelines_components.utils import create_pipeline_run_managed, wait_for_managed_pipeline
from utilities.general import collect_pod_information
from utilities.infra import create_ns

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="session", autouse=True)
def _validate_autorag_env() -> None:
    """Check the external connections needed by the managed AutoRAG pipeline."""
    required_env = {
        "AUTORAG_INFERENCE_MODEL_NAME": "Generation model ID available through MaaS",
        "AUTORAG_EMBEDDING_MODEL_NAME": "Embedding model ID available through MaaS",
        "AUTORAG_MAAS_BASE_URL": "Base URL of an existing MaaS endpoint",
        "AUTORAG_MAAS_API_KEY": "API key for the MaaS endpoint",  # pragma: allowlist secret
        "AUTORAG_MILVUS_URI": "URI of an existing Milvus instance",
    }
    missing = [f"  {name}: {description}" for name, description in required_env.items() if not os.getenv(name)]
    if missing:
        pytest.fail("AutoRAG smoke test requires environment variables:\n" + "\n".join(missing))


@pytest.fixture(scope="class")
def pipelines_namespace(admin_client: DynamicClient) -> Generator[Namespace, Any, Any]:  # noqa: UFN001
    """Namespace for the DSPA and its pipeline connection secrets."""
    with create_ns(admin_client=admin_client, name=f"autorag-aqa-{uuid.uuid4().hex[:8]}") as namespace:
        yield namespace


@pytest.fixture(scope="class")
def autorag_maas_secret(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
) -> Generator[Secret, Any, Any]:
    """MaaS connection for the externally served inference and embedding models."""
    with Secret(
        client=admin_client,
        namespace=pipelines_namespace.name,
        name="autorag-maas-connection",
        string_data={
            "MAAS_BASE_URL": os.environ["AUTORAG_MAAS_BASE_URL"],
            "MAAS_API_KEY": os.environ["AUTORAG_MAAS_API_KEY"],
        },
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def autorag_db_secret(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
) -> Generator[Secret, Any, Any]:
    """Connection to an existing Milvus database; no database is deployed."""
    string_data = {"MILVUS_URI": os.environ["AUTORAG_MILVUS_URI"]}
    if token := os.getenv("AUTORAG_MILVUS_TOKEN"):
        string_data["MILVUS_TOKEN"] = token
    with Secret(
        client=admin_client,
        namespace=pipelines_namespace.name,
        name="autorag-vector-db-connection",
        string_data=string_data,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def autorag_managed_pipeline(
    dspa: DataSciencePipelinesApplication,
    dspa_api_url: str,
    dspa_auth_headers: dict[str, str],
    dspa_ca_bundle_file: str,
) -> dict[str, str]:
    """Discover the managed Documents RAG Optimization pipeline."""
    return wait_for_managed_pipeline(
        api_url=dspa_api_url,
        headers=dspa_auth_headers,
        display_name=MANAGED_PIPELINE_AUTORAG,
        ca_bundle=dspa_ca_bundle_file,
        timeout=DSPA_READY_BUFFER_SECONDS + MANAGED_PIPELINE_WAIT_TIMEOUT,
        poll_interval=MANAGED_PIPELINE_POLL_INTERVAL,
    )


@pytest.fixture(scope="class")
def autorag_test_data(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    dspa_s3_credentials: Secret,
    external_s3_secret: Secret,
) -> None:
    """Copy documents and benchmark data from external S3 into DSPA MinIO."""
    src_bucket = shlex.quote(s=AUTORAG_S3_BUCKET)
    src_input_key = shlex.quote(s=AUTORAG_INPUT_DATA_KEY)
    src_test_key = shlex.quote(s=AUTORAG_TEST_DATA_KEY)
    dst_bucket = shlex.quote(s=DSPA_S3_BUCKET)
    minio_endpoint = f"http://minio-{DSPA_NAME}.{pipelines_namespace.name}.svc.cluster.local:9000"

    script = (
        "export MC_CONFIG_DIR=/work/.mc && "
        "mc alias set src $SRC_ENDPOINT $SRC_ACCESS_KEY $SRC_SECRET_KEY && "
        "mc alias set dst $DST_ENDPOINT $DST_ACCESS_KEY $DST_SECRET_KEY && "
        f"mc cp --recursive src/{src_bucket}/{src_input_key} /work/input_data/ && "
        f"mc cp src/{src_bucket}/{src_test_key} /work/benchmark_data.json && "
        f"mc mb --ignore-existing dst/{dst_bucket} && "
        f"mc cp --recursive /work/input_data/ dst/{dst_bucket}/{src_input_key}/ && "
        f"mc cp /work/benchmark_data.json dst/{dst_bucket}/{src_test_key}"
    )

    with Pod(
        client=admin_client,
        name=f"autorag-data-uploader-{uuid.uuid4().hex[:8]}",
        namespace=pipelines_namespace.name,
        restart_policy="Never",
        volumes=[{"name": "work", "emptyDir": {}}],
        containers=[
            {
                "name": "minio-uploader",
                "image": MINIO_MC_IMAGE,
                "command": ["/bin/sh", "-c"],
                "args": [script],
                "volumeMounts": [{"name": "work", "mountPath": "/work"}],
                "securityContext": MINIO_UPLOADER_SECURITY_CONTEXT,
                "env": [
                    {"name": "SRC_ENDPOINT", "value": os.getenv("AWS_S3_ENDPOINT", "https://s3.amazonaws.com")},
                    {
                        "name": "SRC_ACCESS_KEY",
                        "valueFrom": {"secretKeyRef": {"name": EXTERNAL_S3_SECRET, "key": "AWS_ACCESS_KEY_ID"}},
                    },
                    {
                        "name": "SRC_SECRET_KEY",
                        "valueFrom": {"secretKeyRef": {"name": EXTERNAL_S3_SECRET, "key": "AWS_SECRET_ACCESS_KEY"}},
                    },
                    {"name": "DST_ENDPOINT", "value": minio_endpoint},
                    {
                        "name": "DST_ACCESS_KEY",
                        "valueFrom": {"secretKeyRef": {"name": DSPA_S3_SECRET, "key": "accesskey"}},
                    },
                    {
                        "name": "DST_SECRET_KEY",
                        "valueFrom": {"secretKeyRef": {"name": DSPA_S3_SECRET, "key": "secretkey"}},
                    },
                ],
            }
        ],
        wait_for_resource=True,
    ) as upload_pod:
        try:
            upload_pod.wait_for_status(status="Succeeded", timeout=300)
        except TimeoutExpiredError:
            collect_pod_information(pod=upload_pod)
            raise

    LOGGER.info("AutoRAG test data uploaded to MinIO")


@pytest.fixture(scope="class")
def autorag_run_id(
    dspa_api_url: str,
    dspa_auth_headers: dict[str, str],
    dspa_ca_bundle_file: str,
    autorag_managed_pipeline: dict[str, str],
    autorag_maas_secret: Secret,
    autorag_db_secret: Secret,
    dspa_s3_credentials: Secret,
    autorag_test_data: None,
) -> str:
    """Submit a managed AutoRAG run after its inputs and connections are ready."""
    parameters: dict[str, Any] = {
        "input_data_secret_name": dspa_s3_credentials.name,
        "input_data_bucket_name": DSPA_S3_BUCKET,
        "input_data_keys": [AUTORAG_INPUT_DATA_KEY],
        "test_data_secret_name": dspa_s3_credentials.name,
        "test_data_bucket_name": DSPA_S3_BUCKET,
        "test_data_key": AUTORAG_TEST_DATA_KEY,
        "maas_secret_name": autorag_maas_secret.name,
        "db_secret_name": autorag_db_secret.name,
        "embedding_models": [os.environ["AUTORAG_EMBEDDING_MODEL_NAME"]],
        "generation_models": [os.environ["AUTORAG_INFERENCE_MODEL_NAME"]],
        "optimization_max_rag_patterns": AUTORAG_MAX_RAG_PATTERNS,
        "optimization_metric": AUTORAG_OPTIMIZATION_METRIC,
    }

    return create_pipeline_run_managed(
        api_url=dspa_api_url,
        headers=dspa_auth_headers,
        pipeline_id=autorag_managed_pipeline["pipeline_id"],
        pipeline_version_id=autorag_managed_pipeline["pipeline_version_id"],
        run_name=f"autorag-smoke-{uuid.uuid4().hex[:8]}",
        parameters=parameters,
        ca_bundle=dspa_ca_bundle_file,
    )
