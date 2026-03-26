"""Tests for model signing integration with async upload pipeline.

Flow (Option B):
1. Sign model locally (model.sig created)
2. Upload signed model (model + model.sig) to MinIO
3. Async job converts S3 model to OCI image
4. Sign the OCI image after upload
5. Verify OCI image signature
"""

from pathlib import Path

import pytest
import structlog
from model_registry.signing import Signer

from tests.model_registry.model_registry.async_job.constants import ASYNC_UPLOAD_JOB_NAME
from tests.model_registry.model_registry.python_client.signing.utils import check_model_signature_file
from utilities.constants import MinIo

LOGGER = structlog.get_logger(name=__name__)

pytestmark = pytest.mark.usefixtures("skip_if_not_managed_cluster", "tas_connection_type")


@pytest.mark.parametrize(
    "minio_pod",
    [pytest.param(MinIo.PodConfig.MODEL_REGISTRY_MINIO_CONFIG)],
    indirect=True,
)
@pytest.mark.usefixtures(
    "updated_dsc_component_state_scope_session",
    "model_registry_namespace",
    "model_registry_metadata_db_resources",
    "model_registry_instance",
    "minio_pod",
    "oci_registry_pod",
    "oci_registry_service",
    "ai_hub_oci_registry_route",
    "set_environment_variables",
)
@pytest.mark.custom_namespace
@pytest.mark.downstream_only
@pytest.mark.tier3
class TestAsyncSigningE2E:
    """
    End-to-end test: sign model locally, upload via async job to OCI, then sign and verify OCI image.
    """

    @pytest.mark.dependency(name="test_model_signed_before_upload")
    def test_model_signed_before_upload(self, signed_model_dir: Path):
        """Verify model was signed locally before upload to MinIO."""
        assert check_model_signature_file(model_dir=str(signed_model_dir))
        LOGGER.info(f"Model signed successfully at {signed_model_dir}")

    @pytest.mark.dependency(
        name="test_async_job_uploads_signed_model",
        depends=["test_model_signed_before_upload"],
    )
    def test_async_job_uploads_signed_model(
        self,
        signing_job_pod,
    ):
        """Verify async job completes and uploads signed model to OCI registry."""
        assert signing_job_pod.name.startswith(f"{ASYNC_UPLOAD_JOB_NAME}-signing")
        LOGGER.info(f"Async upload job completed, pod: {signing_job_pod.name}")

    @pytest.mark.dependency(depends=["test_async_job_uploads_signed_model"])
    def test_job_log_signing_disabled(
        self,
        signing_job_pod,
    ):
        """Verify the job pod log indicates signing is disabled (external signing flow)."""
        expected_message = "Signing is disabled"
        assert expected_message in signing_job_pod.log(), f"Expected '{expected_message}' not found in job pod log"

    @pytest.mark.dependency(
        name="test_sign_oci_image",
        depends=["test_async_job_uploads_signed_model"],
    )
    def test_sign_oci_image(self, signer: Signer, oci_image_with_digest: str):
        """Sign the OCI image that was created by the async upload job."""
        LOGGER.info(f"Signing OCI image: {oci_image_with_digest}")
        signer.sign_image(image=oci_image_with_digest)
        LOGGER.info("OCI image signed successfully")

    @pytest.mark.dependency(depends=["test_sign_oci_image"])
    def test_verify_oci_image(self, signer: Signer, oci_image_with_digest: str):
        """Verify the signed OCI image."""
        LOGGER.info(f"Verifying OCI image: {oci_image_with_digest}")
        signer.verify_image(image=oci_image_with_digest)
        LOGGER.info("OCI image verified successfully")
