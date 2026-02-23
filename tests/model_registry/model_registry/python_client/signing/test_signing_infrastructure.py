"""Tests for TAS signing infrastructure setup and readiness."""

from typing import Self
import json
import pytest
from ocp_resources.config_map import ConfigMap
from simple_logger.logger import get_logger
from utilities.resources.securesign import Securesign

LOGGER = get_logger(name=__name__)


@pytest.mark.usefixtures("tas_connection_type")
class TestSigningInfrastructure:
    """Test suite to verify TAS signing infrastructure is ready for model signing operations."""

    def test_signing_environment_ready(
        self: Self,
        securesign_instance: Securesign,
        tas_connection_type: ConfigMap,
        oidc_issuer_url: str,
    ):
        """Temporary test to verify the complete signing environment is ready.

        This test validates that all required components for model signing are
        properly configured and ready:
        - TAS operator is installed
        - Securesign instance is ready
        - All Sigstore components (Fulcio, Rekor, TUF, TSA) are available
        - Service URLs are accessible via Connection Type
        - OIDC issuer is configured

        Replace this test with a more comprehensive test suite for model signing.

        Args:
            admin_client: Kubernetes dynamic client
            securesign_instance: Securesign instance from fixture
            tas_connection_type: TAS Connection Type ConfigMap from fixture
            oidc_issuer_url: OIDC issuer URL from cluster
        """
        LOGGER.info("Verifying Model Signing Environment Readiness")

        # 1. Verify Securesign instance status has all required service URLs
        instance = securesign_instance.instance.to_dict()

        status = instance.get("status", {})
        assert status, "Securesign instance has no status"

        # Check each service has a URL
        required_services = ["fulcio", "rekor", "tuf", "tsa"]
        for service in required_services:
            service_status = status.get(service, {})
            url = service_status.get("url")
            assert url, f"Service '{service}' has no URL in Securesign status"
            assert url.startswith("https://"), f"Service '{service}' URL is not HTTPS: {url}"
            LOGGER.info(f"✓ {service.upper()} service available: {url}")

        # 2. Verify Connection Type has all required environment variables
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
            LOGGER.info(f"✓ {env_var} configured: {url[:50]}...")

        # 3. Verify OIDC issuer is set
        assert oidc_issuer_url, "OIDC issuer URL is empty"
        LOGGER.info(f"✓ OIDC issuer configured: {oidc_issuer_url}")

        LOGGER.info("Environment is ready for model signing operations!")
