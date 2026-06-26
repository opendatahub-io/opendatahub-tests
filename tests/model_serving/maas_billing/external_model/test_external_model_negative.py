from __future__ import annotations

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import UnprocessibleEntityError
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret

from tests.model_serving.maas_billing.external_model.utils import (
    external_provider_ref,
)
from tests.model_serving.maas_billing.utils import build_maas_headers
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

    @pytest.mark.tier3
    @pytest.mark.parametrize(
        "name, provider, endpoint",
        [
            pytest.param(
                "e2e-typo-provider",
                TYPO_PROVIDER,
                "api.openai.com",
                marks=pytest.mark.skip(reason="CRD does not yet reject invalid provider enum values"),
                id="test_typo_provider_rejected_by_crd",
            ),
            pytest.param(
                "e2e-bad-endpoint",
                "openai",
                "https://not-a-valid-fqdn!@#",
                id="test_invalid_endpoint_format_rejected_by_crd",
            ),
        ],
    )
    def test_external_provider_invalid_config_rejected_by_crd(
        self,
        admin_client: DynamicClient,
        maas_unprivileged_model_namespace: Namespace,
        external_model_credential_secret: Secret,
        name: str,
        provider: str,
        endpoint: str,
    ) -> None:
        """Given an ExternalProvider with invalid configuration, when it is created, then the API rejects it."""
        with pytest.raises(UnprocessibleEntityError):
            ExternalProvider(
                client=admin_client,
                name=name,
                namespace=maas_unprivileged_model_namespace.name,
                provider=provider,
                endpoint=endpoint,
                auth={
                    "type": "simple",
                    "secretRef": {"name": external_model_credential_secret.name},
                },
                teardown=True,
            ).deploy()
        LOGGER.info(
            f"ExternalProvider with provider='{provider}', endpoint='{endpoint}' correctly rejected by CRD validation"
        )

    @pytest.mark.tier3
    @pytest.mark.parametrize(
        "name, external_provider_refs",
        [
            pytest.param(
                "e2e-missing-provider-refs",
                [],
                id="test_missing_external_provider_refs_rejected",
            ),
            pytest.param(
                "e2e-bad-api-format",
                "invalid_api_format",
                marks=pytest.mark.skip(reason="CRD does not yet reject invalid apiFormat enum values"),
                id="test_invalid_api_format_rejected_by_crd",
            ),
        ],
    )
    def test_external_model_invalid_config_rejected_by_crd(
        self,
        admin_client: DynamicClient,
        maas_unprivileged_model_namespace: Namespace,
        external_provider_cr: ExternalProvider,
        name: str,
        external_provider_refs: list[dict[str, object]] | str,
    ) -> None:
        """Given an ExternalModel with invalid configuration, when it is created, then the API rejects it."""
        if external_provider_refs == "invalid_api_format":
            bad_ref = external_provider_ref(provider_name=external_provider_cr.name)
            bad_ref["apiFormat"] = "not-a-valid-api-format"
            provider_refs = [bad_ref]
        else:
            provider_refs = external_provider_refs

        with pytest.raises(UnprocessibleEntityError):
            ExternalModel(
                client=admin_client,
                name=name,
                namespace=maas_unprivileged_model_namespace.name,
                external_provider_refs=provider_refs,
                teardown=True,
            ).deploy()
        LOGGER.info(f"ExternalModel '{name}' with invalid config correctly rejected by CRD validation")


@pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "external_model_credential_secret",
    "external_provider_cr",
    "external_model_cr",
    "external_model_ref",
    "external_model_auth_policy",
    "external_model_subscription",
)
class TestExternalModelNegativeGateway:
    """Gateway negative tests requiring a valid API key."""

    @pytest.mark.tier3
    def test_request_to_nonexistent_model_returns_not_found(
        self,
        request_session_http: requests.Session,
        maas_scheme: str,
        maas_host: str,
        maas_unprivileged_model_namespace: Namespace,
        external_model_api_key: str,
    ) -> None:
        """Given a valid API key and unknown model path, when a chat request is sent, then the gateway returns 404."""
        url = (
            f"{maas_scheme}://{maas_host}"
            f"/{maas_unprivileged_model_namespace.name}"
            f"/{NON_EXISTENT_MODEL_PATH}/v1/chat/completions"
        )
        response = request_session_http.post(
            url=url,
            headers=build_maas_headers(token=external_model_api_key),
            json={
                "model": NON_EXISTENT_MODEL_PATH,
                "messages": [{"role": "user", "content": "hello"}],
            },
            timeout=60,
        )
        assert response.status_code == 404, (
            f"Expected 404 for non-existent model route, got {response.status_code}: {(response.text or '')[:200]}"
        )
        LOGGER.info(
            f"Request to non-existent model '{NON_EXISTENT_MODEL_PATH}' correctly returned {response.status_code}"
        )
