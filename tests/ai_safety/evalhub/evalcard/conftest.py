from __future__ import annotations

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
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from pytest_testconfig import config as py_config

from tests.ai_safety.evalhub.constants import (
    EVALHUB_COLLECTIONS_PATH,
    EVALHUB_TENANT_LABEL_KEY,
    EVALHUB_TENANT_LABEL_VALUE,
    EVALHUB_VLLM_EMULATOR_PORT,
)
from tests.ai_safety.evalhub.evalcard.constants import EVALCARD_OCI_REPO, EVALCARD_OCI_TAG
from tests.ai_safety.evalhub.utils import (
    MLflowWithWorkspaces,
    build_evalhub_job_payload,
    build_headers,
    submit_evalhub_job,
    validate_evalhub_job_completed,
    wait_for_evalhub_job,
)

LOGGER = structlog.get_logger(name=__name__)

MLFLOW_SERVICE_PORT = 8443


@pytest.fixture(scope="class")
def evalcard_mlflow_instance(
    admin_client: DynamicClient,
) -> Generator[MLflow, Any, Any]:
    """Deploy an MLflow instance in the applications namespace for EvalCard tracking."""
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
def evalhub_mt_cr(  # noqa: UFN001
    admin_client: DynamicClient,
    model_namespace: Namespace,
    evalcard_mlflow_instance: MLflow,
) -> Generator[EvalHub, Any, Any]:
    """Override the shared evalhub_mt_cr fixture to add MLflow tracking."""
    apps_ns = py_config["applications_namespace"]
    mlflow_uri = f"https://mlflow.{apps_ns}.svc.cluster.local:{MLFLOW_SERVICE_PORT}/mlflow"
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


def _build_model_url(emulator_service: Service, tenant_namespace: str) -> str:
    return f"http://{emulator_service.name}.{tenant_namespace}.svc.cluster.local:{EVALHUB_VLLM_EMULATOR_PORT}/v1"


@pytest.fixture(scope="class")
def evalcard_single_task_job(
    tenant_a_token: str,
    tenant_a_namespace: Namespace,
    evalhub_mt_ca_bundle_file: str,
    evalhub_mt_route: Route,
    evalhub_vllm_emulator_service: Service,
) -> tuple[str, dict]:
    """Submit a single-task job with MLflow experiment; wait for completion."""
    payload = build_evalhub_job_payload(
        model_service_name=evalhub_vllm_emulator_service.name,
        tenant_namespace=tenant_a_namespace.name,
        job_name="evalcard-single-task",
    )
    payload["experiment"] = {"name": "evalcard-single-task-experiment"}

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
    return job_id, job_result


@pytest.fixture(scope="class")
def evalcard_collection_id(
    tenant_a_token: str,
    tenant_a_namespace: Namespace,
    evalhub_mt_ca_bundle_file: str,
    evalhub_mt_route: Route,
) -> Generator[str, Any, Any]:
    """Create a collection with multiple benchmarks; delete on teardown."""
    headers = build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name)
    base = f"https://{evalhub_mt_route.host}{EVALHUB_COLLECTIONS_PATH}"

    collection_payload = {
        "name": "evalcard-fvt-collection",
        "description": "Collection for EvalCard FVT",
        "category": "test",
        "benchmarks": [
            {
                "id": "arc_easy",
                "provider_id": "lm_evaluation_harness",
                "parameters": {
                    "num_examples": 10,
                    "tokenizer": "google/flan-t5-small",
                },
                "weight": 0.6,
                "pass_criteria": {"threshold": 0.3},
                "primary_score": {"metric": "accuracy", "lower_is_better": False},
            },
        ],
    }
    resp = requests.post(
        url=base,
        headers=headers,
        json=collection_payload,
        verify=evalhub_mt_ca_bundle_file,
        timeout=30,
    )
    assert resp.status_code == 201, f"Expected 201 for collection create, got {resp.status_code}: {resp.text}"
    collection_id = resp.json()["resource"]["id"]
    LOGGER.info(f"Created evalcard FVT collection: {collection_id}")

    yield collection_id

    requests.delete(
        url=f"{base}/{collection_id}?hard_delete=true",
        headers=headers,
        verify=evalhub_mt_ca_bundle_file,
        timeout=10,
    )


@pytest.fixture(scope="class")
def evalcard_collection_job(
    tenant_a_token: str,
    tenant_a_namespace: Namespace,
    evalhub_mt_ca_bundle_file: str,
    evalhub_mt_route: Route,
    evalhub_vllm_emulator_service: Service,
    evalcard_collection_id: str,
) -> tuple[str, dict]:
    """Submit a collection job with MLflow experiment; wait for completion."""
    model_url = _build_model_url(
        emulator_service=evalhub_vllm_emulator_service,
        tenant_namespace=tenant_a_namespace.name,
    )
    payload = {
        "name": "evalcard-collection-job",
        "model": {"url": model_url, "name": "emulatedModel"},
        "collection": {"id": evalcard_collection_id},
        "experiment": {"name": "evalcard-collection-experiment"},
    }

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
    return job_id, job_result


@pytest.fixture(scope="class")
def evalcard_complete_job(
    tenant_a_token: str,
    tenant_a_namespace: Namespace,
    evalhub_mt_ca_bundle_file: str,
    evalhub_mt_route: Route,
    evalhub_vllm_emulator_service: Service,
) -> tuple[str, dict]:
    """Submit a production-style job with experiment, pass_criteria, and weights."""
    model_url = _build_model_url(
        emulator_service=evalhub_vllm_emulator_service,
        tenant_namespace=tenant_a_namespace.name,
    )
    payload = {
        "name": "evalcard-complete",
        "model": {"url": model_url, "name": "emulatedModel"},
        "benchmarks": [
            {
                "id": "arc_easy",
                "provider_id": "lm_evaluation_harness",
                "parameters": {
                    "num_examples": 10,
                    "tokenizer": "google/flan-t5-small",
                },
                "primary_score": {"metric": "accuracy", "lower_is_better": False},
                "pass_criteria": {"threshold": 0.3},
                "weight": 0.6,
            },
        ],
        "experiment": {"name": "evalcard-complete-experiment"},
    }

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
    return job_id, job_result


@pytest.fixture(scope="class")
def evalcard_oci_credentials_secret(
    admin_client: DynamicClient,
    oci_registry_host: str,
    tenant_a_namespace: Namespace,
) -> Generator[Secret, Any, Any]:
    """Create OCI registry credentials secret for eval card export."""
    import json
    from base64 import b64encode

    dockerconfig = {
        "auths": {
            oci_registry_host: {
                "auth": "",
                "email": "user@example.com",
            }
        }
    }

    data_dict = {
        ".dockerconfigjson": b64encode(json.dumps(dockerconfig).encode()).decode(),
        "OCI_HOST": b64encode(oci_registry_host.encode()).decode(),
    }

    with Secret(
        client=admin_client,
        name="evalcard-oci-credentials",
        namespace=tenant_a_namespace.name,
        data_dict=data_dict,
        type="kubernetes.io/dockerconfigjson",
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def evalcard_oci_job(
    tenant_a_token: str,
    tenant_a_namespace: Namespace,
    evalhub_mt_ca_bundle_file: str,
    evalhub_mt_route: Route,
    evalhub_vllm_emulator_service: Service,
    evalcard_oci_credentials_secret: Secret,
    oci_registry_pod_with_minio,
) -> tuple[str, dict]:
    """Submit a job configured to export evalcard to an OCI registry."""
    payload = build_evalhub_job_payload(
        model_service_name=evalhub_vllm_emulator_service.name,
        tenant_namespace=tenant_a_namespace.name,
        job_name="evalcard-oci-export",
    )
    payload["experiment"] = {"name": "evalcard-oci-experiment"}
    payload["evalcard_export"] = {
        "oci": {
            "registry": {
                "name": evalcard_oci_credentials_secret.name,
                "key": "OCI_HOST",
            },
            "repository": EVALCARD_OCI_REPO,
            "tag": EVALCARD_OCI_TAG,
            "dockerConfigJson": {
                "name": evalcard_oci_credentials_secret.name,
                "key": ".dockerconfigjson",
            },
            "verifySSL": False,
        },
    }

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
    return job_id, job_result
