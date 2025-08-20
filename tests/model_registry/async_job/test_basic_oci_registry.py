import pytest
import json

from ocp_resources.route import Route

from utilities.constants import OCIRegistry, MinIo
from simple_logger.logger import get_logger
from tests.model_registry.async_job.utils import (
    push_blob_to_oci_registry,
    create_manifest,
    push_manifest_to_oci_registry,
    pull_manifest_from_oci_registry,
)

LOGGER = get_logger(name=__name__)


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
@pytest.mark.usefixtures("minio_pod", "oci_registry_pod_with_minio", "oci_registry_route")
class TestOciRegistry:
    """
    Temporary test for OCI registry deployment functionality using MinIO backend
    It will be replaced with a more comprehensive e2e test as part of https://issues.redhat.com/browse/RHOAISTRAT-456
    """

    @pytest.mark.skip(reason="To remove soon")
    def test_oci_registry_push_and_pull_operations(
        self,
        oci_registry_route: Route,
    ) -> None:
        """Test pushing and pulling content to/from the OCI registry with MinIO backend."""

        registry_host = oci_registry_route.instance.spec.host
        registry_url = f"http://{registry_host}"

        LOGGER.info(f"Testing OCI registry at: {registry_url}")
        test_repo = "test/simple-artifact"
        test_tag = "v1.0"
        test_data = b"Hello from OCI registry test! This could be model data stored in MinIO."

        blob_digest = push_blob_to_oci_registry(registry_url=registry_url, data=test_data, repo=test_repo)

        config_data = {"architecture": "amd64", "os": "linux"}
        config_json = json.dumps(config_data, separators=(",", ":")).encode("utf-8")
        config_digest = push_blob_to_oci_registry(registry_url=registry_url, data=config_json, repo=test_repo)

        manifest = create_manifest(
            blob_digest=blob_digest, config_json=config_json, config_digest=config_digest, data=test_data
        )

        push_manifest_to_oci_registry(registry_url=registry_url, manifest=manifest, repo=test_repo, tag=test_tag)

        manifest_get = pull_manifest_from_oci_registry(registry_url=registry_url, repo=test_repo, tag=test_tag)

        assert manifest_get["schemaVersion"] == 2
        assert manifest_get["config"]["digest"] == config_digest
        assert len(manifest_get["layers"]) == 1
        assert manifest_get["layers"][0]["digest"] == blob_digest
