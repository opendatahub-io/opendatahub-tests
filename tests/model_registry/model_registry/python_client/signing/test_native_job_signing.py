"""Tests for native signing in async upload job (PR #2337).

Flow:
1. Upload unsigned model to MinIO
2. Async job downloads model, signs it, uploads to OCI, and signs the OCI image
3. Verify the OCI image signature externally

This tests the job's built-in signing capability via MODEL_SYNC_SIGN=true,
where the job itself handles both model signing and OCI image signing internally.
"""

import pytest
import structlog
from model_registry.signing import Signer

from tests.model_registry.model_registry.async_job.constants import ASYNC_UPLOAD_JOB_NAME
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
class TestNativeJobSigningE2E:
    """
    End-to-end test: async job signs model and OCI image internally via MODEL_SYNC_SIGN=true.

    Unlike TestAsyncSigningE2E (Option B) where signing happens externally,
    this test verifies that the async upload job can handle signing natively
    when configured with Sigstore environment variables.
    """

    @pytest.mark.dependency(name="test_native_signing_job_completes")
    def test_native_signing_job_completes(
        self,
        native_signing_job_pod,
    ):
        """Verify the async job completes with native signing enabled."""
        assert native_signing_job_pod.name.startswith(f"{ASYNC_UPLOAD_JOB_NAME}-native-signing")
        LOGGER.info(f"Native signing async job completed, pod: {native_signing_job_pod.name}")

    @pytest.mark.dependency(depends=["test_native_signing_job_completes"])
    def test_job_log_contains_signed_image(
        self,
        native_signing_job_pod,
    ):
        """Verify the job pod log contains the 'Signed image successfully' message."""
        expected_message = "Signed image successfully: "
        assert expected_message in native_signing_job_pod.log(), (
            f"Expected '{expected_message}' not found in job pod log"
        )

    @pytest.mark.dependency(depends=["test_native_signing_job_completes"])
    def test_job_log_contains_verified_image(
        self,
        native_signing_job_pod,
    ):
        """Verify the job pod log contains the 'Verified image successfully' message."""
        expected_message = "Verified image successfully: "
        assert expected_message in native_signing_job_pod.log(), (
            f"Expected '{expected_message}' not found in job pod log"
        )

    @pytest.mark.dependency(depends=["test_native_signing_job_completes"])
    def test_verify_natively_signed_oci_image(
        self,
        signer: Signer,
        native_signing_oci_image_with_digest: str,
    ):
        """Verify the OCI image that was signed by the async job itself.

        The job should have signed the image internally using cosign with
        the identity token and Sigstore service URLs. We verify externally
        to confirm the signature is valid.
        """
        LOGGER.info(f"Verifying natively signed OCI image: {native_signing_oci_image_with_digest}")
        signer.verify_image(image=native_signing_oci_image_with_digest)
        LOGGER.info("Natively signed OCI image verified successfully")
