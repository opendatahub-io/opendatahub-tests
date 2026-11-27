"""Managed AutoRAG run inputs shared by smoke and upgrade checks."""

import json
import shlex
import uuid
from pathlib import Path
from typing import Any

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.pod import Pod
from timeout_sampler import TimeoutExpiredError

from tests.pipelines_components.constants import (
    AUTORAG_INPUT_DATA_KEY,
    AUTORAG_MAX_RAG_PATTERNS,
    AUTORAG_OPTIMIZATION_METRIC,
    AUTORAG_TEST_DATA_KEY,
    AUTORAG_VECTOR_IO_PROVIDER_ID,
    DSPA_NAME,
    DSPA_S3_BUCKET,
    DSPA_S3_SECRET,
    MINIO_MC_IMAGE,
    MINIO_UPLOADER_LABELS,
    MINIO_UPLOADER_SECURITY_CONTEXT,
)
from tests.pipelines_components.utils import create_pipeline_run_managed, get_pipeline_version
from utilities.general import collect_pod_information

LOGGER = structlog.get_logger(name=__name__)


def upload_autorag_data(client: DynamicClient, namespace: str | None) -> None:
    """Upload a local document and matching benchmark to the DSPA's MinIO.

    Args:
        client: Kubernetes client.
        namespace: Namespace containing the DSPA and its MinIO.
    """
    assert namespace is not None, "DSPA namespace must have a name"
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
    endpoint = f"http://minio-{DSPA_NAME}.{namespace}.svc.cluster.local:9000"
    script = (
        "export MC_CONFIG_DIR=/work/.mc && "
        "mc alias set dst $DST_ENDPOINT $DST_ACCESS_KEY $DST_SECRET_KEY && "
        f"mc mb --ignore-existing dst/{bucket} && "
        f"mc cp /data/sample.md dst/{bucket}/{input_key} && "
        f"mc cp /data/benchmark_data.json dst/{bucket}/{benchmark_key}"
    )

    with (
        ConfigMap(
            client=client,
            name=f"autorag-data-{uuid.uuid4().hex[:8]}",
            namespace=namespace,
            data={"sample.md": source_document, "benchmark_data.json": benchmark},
        ) as test_data,
        Pod(
            client=client,
            name=f"autorag-data-uploader-{uuid.uuid4().hex[:8]}",
            namespace=namespace,
            label=MINIO_UPLOADER_LABELS,
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
                        {"name": "DST_ENDPOINT", "value": endpoint},
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


def submit_autorag_run(
    api_url: str,
    headers: dict[str, str],
    ca_bundle: str,
    pipeline: dict[str, str],
    s3_secret_name: str | None,
    maas_secret_name: str | None,
    db_secret_name: str | None,
    run_name: str,
) -> str:
    """Submit the managed run, retrying older pipeline parameter names if required.

    Args:
        api_url: KFP API base URL.
        headers: API authentication headers.
        ca_bundle: TLS CA bundle file.
        pipeline: Managed pipeline and version identifiers.
        s3_secret_name: DSPA object-storage Secret.
        maas_secret_name: Secret with MaaS and legacy OGX connection fields.
        db_secret_name: Vector database connection Secret.
        run_name: Display name for the run.

    Returns:
        The submitted KFP run ID.
    """
    assert s3_secret_name and maas_secret_name and db_secret_name, "AutoRAG connection Secrets must have names"
    pipeline_version = get_pipeline_version(
        api_url=api_url,
        headers=headers,
        pipeline_id=pipeline["pipeline_id"],
        pipeline_version_id=pipeline["pipeline_version_id"],
        ca_bundle=ca_bundle,
    )
    parameter_definitions = (
        pipeline_version
        .get("pipeline_spec", {})
        .get("pipeline_spec", {})
        .get("root", {})
        .get("inputDefinitions", {})
        .get("parameters", {})
    )
    if not isinstance(parameter_definitions, dict) or not parameter_definitions:
        raise ValueError(f"Pipeline version {pipeline['pipeline_version_id']} does not declare root parameters")

    parameter_values: dict[str, Any] = {
        "input_data_secret_name": s3_secret_name,
        "input_data_bucket_name": DSPA_S3_BUCKET,
        "input_data_keys": [AUTORAG_INPUT_DATA_KEY],
        "input_data_key": AUTORAG_INPUT_DATA_KEY,
        "test_data_secret_name": s3_secret_name,
        "test_data_bucket_name": DSPA_S3_BUCKET,
        "test_data_key": AUTORAG_TEST_DATA_KEY,
        "maas_secret_name": maas_secret_name,
        "ogx_secret_name": maas_secret_name,
        "db_secret_name": db_secret_name,
        "vector_db_secret_name": db_secret_name,
        "vector_io_provider_id": AUTORAG_VECTOR_IO_PROVIDER_ID,
        "embedding_models": ["smoke-embedding-model"],
        "generation_models": ["smoke-generation-model"],
        "optimization_max_rag_patterns": AUTORAG_MAX_RAG_PATTERNS,
        "optimization_metric": AUTORAG_OPTIMIZATION_METRIC,
    }
    parameters = {name: value for name, value in parameter_values.items() if name in parameter_definitions}
    required_parameters = {
        name
        for name, definition in parameter_definitions.items()
        if not definition.get("isOptional", False) and definition.get("defaultValue") is None
    }
    missing_parameters = sorted(required_parameters - parameters.keys())
    if missing_parameters:
        raise ValueError(f"AutoRAG smoke test does not provide required parameters: {', '.join(missing_parameters)}")

    return create_pipeline_run_managed(
        api_url=api_url,
        headers=headers,
        pipeline_id=pipeline["pipeline_id"],
        pipeline_version_id=pipeline["pipeline_version_id"],
        run_name=run_name,
        parameters=parameters,
        ca_bundle=ca_bundle,
    )
