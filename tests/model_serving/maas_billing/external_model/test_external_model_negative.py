from __future__ import annotations

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import UnprocessibleEntityError
from ocp_resources.exceptions import MissingRequiredArgumentError
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret

from utilities.resources.external_model import ExternalModel
from utilities.resources.external_provider import ExternalProvider

LOGGER = structlog.get_logger(name=__name__)

NON_EXISTENT_MODEL_PATH = "non-existent-model-xyz"
TYPO_PROVIDER = "opeanai"


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "external_model_credential_secret",
    "external_provider_cr",
)
class TestExternalModelNegative:
    """Negative tests for ExternalModel CRD validation and gateway error handling."""

    @pytest.mark.skip
    @pytest.mark.tier3
    def test_typo_provider_rejected_by_crd(
        self,
        admin_client: DynamicClient,
        maas_unprivileged_model_namespace: Namespace,
        external_model_credential_secret: Secret,
    ) -> None:
        """Given an ExternalProvider with a typo in provider, when it is created, then the API rejects it."""
        with pytest.raises(UnprocessibleEntityError):
            ExternalProvider(
                client=admin_client,
                name="e2e-typo-provider",
                namespace=maas_unprivileged_model_namespace.name,
                provider=TYPO_PROVIDER,
                endpoint="api.openai.com",
                auth={
                    "type": "simple",
                    "secretRef": {"name": external_model_credential_secret.name},
                },
                teardown=True,
            ).deploy()

        LOGGER.info(f"ExternalProvider with provider '{TYPO_PROVIDER}' correctly rejected by CRD validation")

    @pytest.mark.tier3
    def test_missing_external_provider_refs_rejected(
        self,
        admin_client: DynamicClient,
        maas_unprivileged_model_namespace: Namespace,
    ) -> None:
        """Given an ExternalModel without externalProviderRefs, when it is created, then validation fails."""
        with pytest.raises(MissingRequiredArgumentError):
            ExternalModel(
                client=admin_client,
                name="e2e-missing-provider-refs",
                namespace=maas_unprivileged_model_namespace.name,
                teardown=True,
            ).deploy()

        LOGGER.info("ExternalModel without externalProviderRefs correctly rejected")

    @pytest.mark.tier3
    def test_invalid_endpoint_format_rejected_by_crd(
        self,
        admin_client: DynamicClient,
        maas_unprivileged_model_namespace: Namespace,
        external_model_credential_secret: Secret,
    ) -> None:
        """Given an ExternalProvider with an invalid endpoint format, when it is created, then the API rejects it."""
        with pytest.raises(UnprocessibleEntityError):
            ExternalProvider(
                client=admin_client,
                name="e2e-bad-endpoint",
                namespace=maas_unprivileged_model_namespace.name,
                provider="openai",
                endpoint="https://not-a-valid-fqdn!@#",
                auth={
                    "type": "simple",
                    "secretRef": {"name": external_model_credential_secret.name},
                },
                teardown=True,
            ).deploy()

        LOGGER.info("ExternalProvider with invalid endpoint format correctly rejected by CRD validation")

    @pytest.mark.skip
    @pytest.mark.tier3
    def test_invalid_api_format_rejected_by_crd(
        self,
        admin_client: DynamicClient,
        maas_unprivileged_model_namespace: Namespace,
        external_provider_cr: ExternalProvider,
    ) -> None:
        """Given an ExternalModel with an invalid apiFormat, when it is created, then the API rejects it."""
        with pytest.raises(UnprocessibleEntityError):
            ExternalModel(
                client=admin_client,
                name="e2e-bad-api-format",
                namespace=maas_unprivileged_model_namespace.name,
                external_provider_refs=[
                    {
                        "ref": {"name": external_provider_cr.name},
                        "targetModel": "gpt-3.5-turbo",
                        "apiFormat": "openai",
                    }
                ],
                teardown=True,
            ).deploy()

        LOGGER.info("ExternalModel with invalid apiFormat correctly rejected by CRD validation")

    @pytest.mark.tier3
    def test_request_to_nonexistent_model_returns_not_found(
        self,
        request_session_http: requests.Session,
        maas_scheme: str,
        maas_host: str,
        maas_unprivileged_model_namespace: Namespace,
    ) -> None:
        """Given a model name that does not exist, when a chat request is sent, then the gateway returns 404 or 403."""
        url = (
            f"{maas_scheme}://{maas_host}"
            f"/{maas_unprivileged_model_namespace.name}"
            f"/{NON_EXISTENT_MODEL_PATH}/v1/chat/completions"
        )
        response = request_session_http.post(
            url=url,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer INVALID-KEY-12345",
            },
            json={
                "model": NON_EXISTENT_MODEL_PATH,
                "messages": [{"role": "user", "content": "hello"}],
            },
            timeout=60,
        )
        assert response.status_code in (403, 404), (
            f"Expected 403/404 for non-existent model, got {response.status_code}: "
            f"{(response.text or '')[:200]}"
        )
        LOGGER.info(
            f"Request to non-existent model '{NON_EXISTENT_MODEL_PATH}' "
            f"correctly returned {response.status_code}"
        )
