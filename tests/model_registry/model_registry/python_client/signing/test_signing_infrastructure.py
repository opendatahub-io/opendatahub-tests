"""Tests for TAS signing infrastructure setup and readiness."""

import json
from typing import Self

import pytest
import requests
import structlog
from ocp_resources.config_map import ConfigMap

from tests.model_registry.model_registry.python_client.signing.constants import (
    SIGNING_OCI_REPO_NAME,
)
from tests.model_registry.model_registry.python_client.signing.utils import check_model_signature_file
from utilities.resources.securesign import Securesign

LOGGER = structlog.get_logger(name=__name__)

pytestmark = pytest.mark.usefixtures("skip_if_not_managed_cluster", "tas_connection_type")


class TestSigningInfrastructure:
    """
    Test suite to verify TAS signing infrastructure is ready for model signing operations."
    """

    @pytest.mark.dependency(name="test_securesign_service_urls")
    def test_securesign_service_urls(self: Self, securesign_instance: Securesign):
        """Verify Securesign instance has all required service URLs with HTTPS."""
        instance = securesign_instance.instance.to_dict()

        status = instance.get("status", {})
        assert status, "Securesign instance has no status"

        required_services = ["fulcio", "rekor", "tuf", "tsa"]
        for service in required_services:
            service_status = status.get(service, {})
            url = service_status.get("url")
            assert url, f"Service '{service}' has no URL in Securesign status"
            assert url.startswith("https://"), f"Service '{service}' URL is not HTTPS: {url}"
            LOGGER.info(f"{service.upper()} service available: {url}")

    @pytest.mark.dependency(name="test_connection_type_env_vars")
    def test_connection_type_env_vars(self: Self, tas_connection_type: ConfigMap):
        """Verify TAS Connection Type has all required environment variables."""
        data = dict(tas_connection_type.instance.data)
        fields = json.loads(data["fields"])

        env_var_to_url = {}
        for field in fields:
            env_var = field["envVar"]
            default_value = field.get("properties", {}).get("defaultValue")
            if default_value:
                env_var_to_url[env_var] = default_value

        required_env_vars = ["SIGSTORE_FULCIO_URL", "SIGSTORE_REKOR_URL", "SIGSTORE_TUF_URL", "SIGSTORE_TSA_URL"]
        for env_var in required_env_vars:
            assert env_var in env_var_to_url, f"Missing required environment variable: {env_var}"
            url = env_var_to_url[env_var]
            assert url, f"Environment variable {env_var} has empty value"
            LOGGER.info(f"{env_var} configured: {url[:50]}...")

    @pytest.mark.dependency(name="test_oidc_issuer")
    def test_oidc_issuer(self: Self, oidc_issuer_url: str):
        """Verify OIDC issuer URL is configured."""
        assert oidc_issuer_url, "OIDC issuer URL is empty"
        LOGGER.info(f"OIDC issuer configured: {oidc_issuer_url}")


@pytest.mark.usefixtures("set_environment_variables", "downloaded_model_dir")
class TestModelSigning:
    """
    Test suite for model signing and verification.
    """

    @pytest.mark.dependency(
        name="test_model_sign",
        depends=["test_securesign_service_urls", "test_connection_type_env_vars", "test_oidc_issuer"],
    )
    def test_model_sign(self, signed_model):
        """
        Test model signing functionality.
        """

        LOGGER.info(f"Testing model signing in directory: {signed_model}")
        assert signed_model
        has_signature = check_model_signature_file(model_dir=str(signed_model))
        assert has_signature, "model.sig file not found after signing"

    @pytest.mark.dependency(
        depends=[
            "test_securesign_service_urls",
            "test_connection_type_env_vars",
            "test_oidc_issuer",
            "test_model_sign",
        ],
    )
    def test_model_verify(self, signer, downloaded_model_dir):
        """
        Test model verification functionality.
        """
        LOGGER.info("Testing model verification")
        LOGGER.info(f"Verifying signed model in directory: {downloaded_model_dir}")
        signer.verify_model(model_path=str(downloaded_model_dir))
        LOGGER.info("Model verified successfully")


@pytest.mark.usefixtures("set_environment_variables", "oci_registry_pod", "copied_model_to_oci_registry")
class TestImageSigning:
    """
    Test suite for signing OCI images in an OCI registry.
    """

    @pytest.mark.dependency(
        name="test_image_sign",
        depends=["test_securesign_service_urls", "test_connection_type_env_vars", "test_oidc_issuer"],
    )
    def test_image_sign(self, signer, copied_model_to_oci_registry, ai_hub_oci_registry_host):
        """
        Test image signing functionality.
        """

        LOGGER.info(f"Testing model signing for image: {copied_model_to_oci_registry}")
        assert copied_model_to_oci_registry
        registry_url = f"https://{ai_hub_oci_registry_host}"
        tags_url = f"{registry_url}/v2/{SIGNING_OCI_REPO_NAME}/tags/list"

        tags_before = requests.get(tags_url, verify=False, timeout=10).json()
        LOGGER.info(f"Tags before signing: {json.dumps(tags_before, indent=2)}")
        assert tags_before["tags"] == ["latest"], (
            f"Expected only ['latest'] tag before signing, got: {tags_before['tags']}"
        )

        signer.sign_image(image=str(copied_model_to_oci_registry))
        LOGGER.info("Model signed successfully")

        tags_after = requests.get(tags_url, verify=False, timeout=10).json()
        LOGGER.info(f"Tags after signing: {json.dumps(tags_after, indent=2)}")
        digest = copied_model_to_oci_registry.split("@")[-1]
        expected_sig_tag = f"{digest.replace(':', '-')}.sig"
        assert expected_sig_tag in tags_after["tags"], (
            f"Signature tag '{expected_sig_tag}' not found in registry tags: {tags_after['tags']}"
        )

    @pytest.mark.dependency(
        depends=[
            "test_securesign_service_urls",
            "test_connection_type_env_vars",
            "test_oidc_issuer",
            "test_image_sign",
        ],
    )
    def test_image_verify(self, signer, copied_model_to_oci_registry):
        """
        Test image verification functionality.
        """
        LOGGER.info(f"Verifying signed image: {copied_model_to_oci_registry}")
        signer.verify_image(image=str(copied_model_to_oci_registry))
        LOGGER.info("Image verified successfully")
