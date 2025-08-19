from typing import Self
import time

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job
from ocp_resources.route import Route
from model_registry.types import ArtifactState, RegisteredModelState
from tests.model_registry.async_job.constants import (
    ASYNC_UPLOAD_JOB_NAME,
)
from tests.model_registry.async_job.utils import (
    get_latest_job_pod,
    pull_manifest_from_oci_registry,
)
from tests.model_registry.constants import MODEL_DICT
from utilities.constants import MinIo, OCIRegistry
from model_registry import ModelRegistry as ModelRegistryClient
from simple_logger.logger import get_logger
from tests.model_registry.async_job.constants import MODEL_SYNC_CONFIG, REPO_NAME, TAG

LOGGER = get_logger(name=__name__)

MODEL_NAME = f"async-test-model-{int(time.time())}"
MODEL_DATA = {
    **MODEL_DICT,
    "model_name": MODEL_NAME,
    "model_storage_key": MODEL_SYNC_CONFIG["SOURCE_AWS_KEY"],
    "model_storage_path": "path/to/test/model",
}


@pytest.mark.parametrize(
    "minio_pod, oci_registry_pod_with_minio",
    [
        pytest.param(
            MinIo.PodConfig.MODEL_MESH_MINIO_CONFIG,
            OCIRegistry.PodConfig.REGISTRY_BASE_CONFIG,
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures(
    "updated_dsc_component_state_scope_session",
    "model_registry_namespace",
    "model_registry_metadata_db_resources",
    "minio_pod",
    "create_test_data_in_minio_from_image",
    "s3_secret_for_async_job",
    "oci_secret_for_async_job",
    "oci_registry_pod_with_minio",
    "registered_model_from_image",
)
@pytest.mark.custom_namespace
@pytest.mark.parametrize(
    "registered_model_from_image",
    [
        pytest.param(MODEL_DATA, id="test_model_from_image"),
    ],
    indirect=True,
)
class TestAsyncUploadE2E:
    """Test for async upload job with real MinIO, OCI registry, and Model Registry"""

    def test_async_upload_job(
        self: Self,
        admin_client: DynamicClient,
        model_sync_async_job: Job,
        model_registry_client: list[ModelRegistryClient],
        oci_registry_route: Route,
    ) -> None:
        """Test for async upload job execution using model from KSERVE_MINIO_IMAGE"""
        LOGGER.info("Starting async upload job test with KSERVE_MINIO_IMAGE")
        # Wait for job to create a pod
        job_pod = get_latest_job_pod(admin_client=admin_client, job=model_sync_async_job)
        assert job_pod.name.startswith(ASYNC_UPLOAD_JOB_NAME)

        # Verify OCI registry contains the uploaded artifact
        registry_host = oci_registry_route.instance.spec.host
        registry_url = f"http://{registry_host}"

        LOGGER.info(f"Verifying artifact in OCI registry: {registry_url}/v2/{REPO_NAME}/manifests/{TAG}")

        # Check if the manifest exists in the OCI registry
        manifest = pull_manifest_from_oci_registry(registry_url=registry_url, repo=REPO_NAME, tag=TAG)

        LOGGER.info("Manifest found in OCI registry")
        LOGGER.info(f"Manifest schema version: {manifest.get('schemaVersion')}")
        LOGGER.info(f"Manifest media type: {manifest.get('mediaType')}")

        # Verify the manifest has the expected structure
        assert "manifests" in manifest, "Manifest should contain manifests section"
        assert len(manifest["manifests"]) > 0, "Manifest should have at least one manifest"
        LOGGER.info(f"Manifest contains {len(manifest['manifests'])} layer(s)")

        # Verify model registry metadata was updated
        LOGGER.info("Verifying model registry model and artifact")
        client = model_registry_client[0]
        model = client.get_registered_model(name=MODEL_NAME)
        assert model.state == RegisteredModelState.LIVE

        model_artifact = client.get_model_artifact(name=MODEL_NAME, version=MODEL_DATA["model_version"])

        # Validate model artifact attributes
        assert model_artifact.name == MODEL_NAME
        assert model_artifact.state == ArtifactState.LIVE
        assert model_artifact.uri == f"oci://{registry_host}/{REPO_NAME}"
        assert model_artifact.storage_key == MODEL_DATA["model_storage_key"]
        assert model_artifact.storage_path == MODEL_DATA["model_storage_path"]

        LOGGER.info("Async upload job test with KSERVE_MINIO_IMAGE: PASSED")
