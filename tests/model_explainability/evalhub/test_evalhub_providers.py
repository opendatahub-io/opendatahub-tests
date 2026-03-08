import pytest
import requests
from ocp_resources.namespace import Namespace
from ocp_resources.role_binding import RoleBinding
from ocp_resources.route import Route

from tests.model_explainability.evalhub.utils import get_evalhub_provider, list_evalhub_providers


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-providers"},
        ),
    ],
    indirect=True,
)
@pytest.mark.sanity
@pytest.mark.model_explainability
class TestEvalHubProviders:
    """Tests for the EvalHub providers API using a scoped non-admin ServiceAccount."""

    def test_scoped_user_can_list_providers(
        self,
        evalhub_providers_response: dict,
    ) -> None:
        """Verify that a scoped user with providers access can list providers."""
        assert "items" in evalhub_providers_response, "Response missing 'items' field"
        assert isinstance(evalhub_providers_response["items"], list), "'items' must be a list"
        assert "total_count" in evalhub_providers_response, "Response missing 'total_count' field"
        assert "limit" in evalhub_providers_response, "Response missing 'limit' field"

    def test_list_providers_has_registered_providers(
        self,
        evalhub_providers_response: dict,
    ) -> None:
        """Verify that at least one provider is registered."""
        assert evalhub_providers_response["total_count"] > 0, "Expected at least one registered provider"
        assert len(evalhub_providers_response["items"]) > 0, "Expected at least one provider in items"

    def test_provider_has_required_fields(
        self,
        evalhub_providers_response: dict,
    ) -> None:
        """Verify that each provider contains the expected resource metadata and config fields."""
        for provider in evalhub_providers_response["items"]:
            assert "resource" in provider, f"Provider missing 'resource': {provider}"
            assert "id" in provider["resource"], f"Provider resource missing 'id': {provider}"
            assert provider["resource"]["id"], "Provider ID must not be empty"
            assert "name" in provider, f"Provider missing 'name': {provider}"
            assert "benchmarks" in provider, f"Provider missing 'benchmarks': {provider}"

    def test_provider_benchmarks_have_required_fields(
        self,
        evalhub_providers_response: dict,
    ) -> None:
        """Verify that benchmarks within each provider have id, name, and category."""
        for provider in evalhub_providers_response["items"]:
            provider_name = provider.get("name", "unknown")
            for benchmark in provider.get("benchmarks", []):
                assert "id" in benchmark, f"Benchmark in provider '{provider_name}' missing 'id'"
                assert "name" in benchmark, f"Benchmark in provider '{provider_name}' missing 'name'"
                assert "category" in benchmark, f"Benchmark in provider '{provider_name}' missing 'category'"

    def test_lm_evaluation_harness_provider_exists(
        self,
        evalhub_providers_response: dict,
    ) -> None:
        """Verify that the lm_evaluation_harness provider is registered and has benchmarks."""
        provider_ids = [provider["resource"]["id"] for provider in evalhub_providers_response["items"]]
        assert "lm_evaluation_harness" in provider_ids, (
            f"Expected 'lm_evaluation_harness' in providers, got: {provider_ids}"
        )

        lmeval_provider = next(
            provider
            for provider in evalhub_providers_response["items"]
            if provider["resource"]["id"] == "lm_evaluation_harness"
        )
        assert len(lmeval_provider["benchmarks"]) > 0, (
            "lm_evaluation_harness provider should have at least one benchmark"
        )

    def test_get_single_provider(
        self,
        evalhub_providers_response: dict,
        evalhub_scoped_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
        model_namespace: Namespace,
    ) -> None:
        """Verify that a single provider can be retrieved by ID."""
        assert evalhub_providers_response.get("items") and len(evalhub_providers_response["items"]) > 0, (
            "No providers registered; cannot test single-provider retrieval"
        )
        first_provider_id = evalhub_providers_response["items"][0]["resource"]["id"]

        data = get_evalhub_provider(
            host=evalhub_route.host,
            token=evalhub_scoped_token,
            ca_bundle_file=evalhub_ca_bundle_file,
            provider_id=first_provider_id,
            tenant=model_namespace.name,
        )

        assert data["resource"]["id"] == first_provider_id
        assert "name" in data
        assert "benchmarks" in data

    def test_get_nonexistent_provider_returns_error(
        self,
        model_namespace: Namespace,
        evalhub_scoped_token: str,
        evalhub_providers_role_binding: RoleBinding,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
    ) -> None:
        """Verify that requesting a non-existent provider ID returns 404."""
        with pytest.raises(requests.exceptions.HTTPError) as excinfo:
            get_evalhub_provider(
                host=evalhub_route.host,
                token=evalhub_scoped_token,
                ca_bundle_file=evalhub_ca_bundle_file,
                provider_id="nonexistent-provider-id",
                tenant=model_namespace.name,
            )
        assert excinfo.value.response.status_code == 404


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-providers"},
        ),
    ],
    indirect=True,
)
@pytest.mark.sanity
@pytest.mark.model_explainability
class TestEvalHubProvidersUnauthorised:
    """Tests verifying that a user without providers RBAC is denied access."""

    def test_list_providers_denied_without_role_binding(
        self,
        model_namespace: Namespace,
        evalhub_unauthorised_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
    ) -> None:
        """Verify that a user without providers ClusterRole binding gets 403."""
        with pytest.raises(requests.exceptions.HTTPError, match="403"):
            list_evalhub_providers(
                host=evalhub_route.host,
                token=evalhub_unauthorised_token,
                ca_bundle_file=evalhub_ca_bundle_file,
                tenant=model_namespace.name,
            )

    def test_get_provider_denied_without_role_binding(
        self,
        model_namespace: Namespace,
        evalhub_unauthorised_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
        evalhub_providers_response: dict,
    ) -> None:
        """Verify that a user without providers ClusterRole binding cannot get a provider."""
        provider_id = evalhub_providers_response["items"][0]["resource"]["id"]
        with pytest.raises(requests.exceptions.HTTPError, match="403"):
            get_evalhub_provider(
                host=evalhub_route.host,
                token=evalhub_unauthorised_token,
                ca_bundle_file=evalhub_ca_bundle_file,
                provider_id=provider_id,
                tenant=model_namespace.name,
            )
