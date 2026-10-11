import json
import shlex
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.data_science_pipelines_application import DataSciencePipelinesApplication
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from timeout_sampler import TimeoutExpiredError

from tests.pipelines_components.constants import (
    AUTORAG_INPUT_DATA_KEY,
    AUTORAG_MAX_RAG_PATTERNS,
    AUTORAG_OPTIMIZATION_METRIC,
    AUTORAG_TEST_DATA_KEY,
    DSPA_NAME,
    DSPA_READY_BUFFER_SECONDS,
    DSPA_S3_BUCKET,
    DSPA_S3_SECRET,
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
    """Provide the required MaaS Secret shape with a non-routable endpoint."""
    with Secret(
        client=admin_client,
        namespace=pipelines_namespace.name,
        name="autorag-maas-connection",
        string_data={
            "MAAS_BASE_URL": "http://maas.invalid/v1",
            "MAAS_API_KEY": "placeholder",  # pragma: allowlist secret
        },
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def autorag_db_secret(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
) -> Generator[Secret, Any, Any]:
    """Provide the required Milvus Secret shape without a database service."""
    with Secret(
        client=admin_client,
        namespace=pipelines_namespace.name,
        name="autorag-vector-db-connection",
        string_data={
            "MILVUS_URI": "http://milvus.invalid:19530",
            "MILVUS_TOKEN": "placeholder",  # pragma: allowlist secret
        },
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
) -> None:
    """Upload a local document and matching benchmark record to DSPA MinIO."""
    source_document = (Path(__file__).parent / "data" / "sample.md").read_text(encoding="utf-8")
    benchmark = json.dumps([
        {
            "question": "How many raised beds are in the garden?",
            "correct_answers": ["Three raised beds."],
            "correct_answer_document_keys": [f"{AUTORAG_INPUT_DATA_KEY}/sample.md"],
        }
    ])
    input_key = shlex.quote(s=f"{AUTORAG_INPUT_DATA_KEY}/sample.md")
    benchmark_key = shlex.quote(s=AUTORAG_TEST_DATA_KEY)
    bucket = shlex.quote(s=DSPA_S3_BUCKET)
    minio_endpoint = f"http://minio-{DSPA_NAME}.{pipelines_namespace.name}.svc.cluster.local:9000"

    script = (
        "export MC_CONFIG_DIR=/work/.mc && "
        "mc alias set dst $DST_ENDPOINT $DST_ACCESS_KEY $DST_SECRET_KEY && "
        f"mc mb --ignore-existing dst/{bucket} && "
        f"mc cp /data/sample.md dst/{bucket}/{input_key} && "
        f"mc cp /data/benchmark_data.json dst/{bucket}/{benchmark_key}"
    )

    with (
        ConfigMap(
            client=admin_client,
            name=f"autorag-data-{uuid.uuid4().hex[:8]}",
            namespace=pipelines_namespace.name,
            data={"sample.md": source_document, "benchmark_data.json": benchmark},
        ) as test_data,
        Pod(
            client=admin_client,
            name=f"autorag-data-uploader-{uuid.uuid4().hex[:8]}",
            namespace=pipelines_namespace.name,
            restart_policy="Never",
            volumes=[
                {"name": "work", "emptyDir": {}},
                {"name": "test-data", "configMap": {"name": test_data.name}},
            ],
            containers=[
                {
                    "name": "minio-uploader",
                    "image": MINIO_MC_IMAGE,
                    "command": ["/bin/sh", "-c"],
                    "args": [script],
                    "volumeMounts": [
                        {"name": "work", "mountPath": "/work"},
                        {"name": "test-data", "mountPath": "/data", "readOnly": True},
                    ],
                    "securityContext": MINIO_UPLOADER_SECURITY_CONTEXT,
                    "env": [
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
        ) as upload_pod,
    ):
        try:
            upload_pod.wait_for_status(status="Succeeded", timeout=300)
        except TimeoutExpiredError:
            collect_pod_information(pod=upload_pod)
            raise

    LOGGER.info("Synthetic AutoRAG data uploaded to MinIO")


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
        "embedding_models": ["smoke-embedding-model"],
        "generation_models": ["smoke-generation-model"],
        "optimization_max_rag_patterns": AUTORAG_MAX_RAG_PATTERNS,
        "optimization_metric": AUTORAG_OPTIMIZATION_METRIC,
    }

    run_name = f"autorag-smoke-{uuid.uuid4().hex[:8]}"
    try:
        return create_pipeline_run_managed(
            api_url=dspa_api_url,
            headers=dspa_auth_headers,
            pipeline_id=autorag_managed_pipeline["pipeline_id"],
            pipeline_version_id=autorag_managed_pipeline["pipeline_version_id"],
            run_name=run_name,
            parameters=parameters,
            ca_bundle=dspa_ca_bundle_file,
        )
    except requests.HTTPError as error:
        response = error.response
        if (
            response is None
            or response.status_code != 400
            or "parameter vector_db_secret_name is not optional" not in response.text
        ):
            raise

    LOGGER.info("Managed pipeline requires vector_db_secret_name; retrying run submission")
    parameters["vector_db_secret_name"] = parameters.pop("db_secret_name")
    return create_pipeline_run_managed(
        api_url=dspa_api_url,
        headers=dspa_auth_headers,
        pipeline_id=autorag_managed_pipeline["pipeline_id"],
        pipeline_version_id=autorag_managed_pipeline["pipeline_version_id"],
        run_name=run_name,
        parameters=parameters,
        ca_bundle=dspa_ca_bundle_file,
    )
