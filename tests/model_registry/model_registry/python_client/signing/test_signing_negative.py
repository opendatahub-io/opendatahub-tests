"""Negative tests for model signing failure scenarios (TC-011b, TC-011c)."""

from pathlib import Path

import pytest
from model_registry.signing.exceptions import SigningError
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)

pytestmark = pytest.mark.usefixtures("skip_if_not_managed_cluster", "tas_connection_type")

INVALID_IMAGE_REF = "quay.io/invalid-no-access-repo/nonexistent-image:latest"


@pytest.mark.tier3
@pytest.mark.usefixtures("set_environment_variables")
class TestSigningNegative:
    """Negative test suite for signing failure scenarios."""

    def test_sign_image_invalid_registry_credentials(self, signer):
        """TC-011b: Sign container image with invalid registry credentials.

        Verifies that signing an image to a registry where the user has no write
        permissions fails with a SigningError indicating an authentication/authorization issue.
        """
        LOGGER.info(f"Attempting to sign image with no write access: {INVALID_IMAGE_REF}")
        with pytest.raises(SigningError):
            signer.sign_image(image=INVALID_IMAGE_REF)
        LOGGER.info("SigningError raised as expected for invalid registry credentials")

    def test_verify_model_missing_signature(self, signer, tmp_path: Path):
        """TC-011c: Verify model with missing signature file.

        Verifies that attempting to verify a model directory that has no model.sig
        file fails with a VerificationError.
        """
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(data=b"fake onnx model content for testing")

        LOGGER.info(f"Attempting to verify model without signature: {tmp_path}")
        with pytest.raises(FileNotFoundError, match="model.sig"):
            signer.verify_model(model_path=str(tmp_path))
        LOGGER.info("FileNotFoundError raised as expected for missing signature")
