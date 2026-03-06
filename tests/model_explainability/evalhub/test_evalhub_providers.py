import pytest
import requests
from ocp_resources.namespace import Namespace
from ocp_resources.role_binding import RoleBinding
from ocp_resources.route import Route

from tests.model_explainability.evalhub.utils import (
    get_evalhub_provider,
    list_evalhub_providers,
)


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

    def test_list_providers_returns_paginated_response(
        self,
        model_namespace: Namespace,
        evalhub_scoped_token: str,
        evalhub_providers_role_binding: RoleBinding,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
    ) -> None:
        """Verify that a scoped user with providers access can list providers."""
        data = list_evalhub_providers(
            host=evalhub_route.host,
            token=evalhub_scoped_token,
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant=model_namespace.name,
        )

        assert "items" in data, "Response missing 'items' field"
        assert isinstance(data["items"], list), "'items' must be a list"
        assert "total_count" in data, "Response missing 'total_count' field"
        assert "limit" in data, "Response missing 'limit' field"

    def test_list_providers_has_registered_providers(
        self,
        model_namespace: Namespace,
        evalhub_scoped_token: str,
        evalhub_providers_role_binding: RoleBinding,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
    ) -> None:
        """Verify that at least one provider is registered."""
        data = list_evalhub_providers(
            host=evalhub_route.host,
            token=evalhub_scoped_token,
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant=model_namespace.name,
        )

        assert data["total_count"] > 0, "Expected at least one registered provider"
        assert len(data["items"]) > 0, "Expected at least one provider in items"

    def test_provider_has_required_fields(
        self,
        model_namespace: Namespace,
        evalhub_scoped_token: str,
        evalhub_providers_role_binding: RoleBinding,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
    ) -> None:
        """Verify that each provider contains the expected resource metadata and config fields."""
        data = list_evalhub_providers(
            host=evalhub_route.host,
            token=evalhub_scoped_token,
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant=model_namespace.name,
        )

        for provider in data["items"]:
            assert "resource" in provider, f"Provider missing 'resource': {provider}"
            assert "id" in provider["resource"], f"Provider resource missing 'id': {provider}"
            assert provider["resource"]["id"], "Provider ID must not be empty"
            assert "name" in provider, f"Provider missing 'name': {provider}"
            assert "benchmarks" in provider, f"Provider missing 'benchmarks': {provider}"

    def test_provider_benchmarks_have_required_fields(
        self,
        model_namespace: Namespace,
        evalhub_scoped_token: str,
        evalhub_providers_role_binding: RoleBinding,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
    ) -> None:
        """Verify that benchmarks within each provider have id, name, and category."""
        data = list_evalhub_providers(
            host=evalhub_route.host,
            token=evalhub_scoped_token,
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant=model_namespace.name,
        )

        for provider in data["items"]:
            provider_name = provider.get("name", "unknown")
            for benchmark in provider.get("benchmarks", []):
                assert "id" in benchmark, f"Benchmark in provider '{provider_name}' missing 'id'"
                assert "name" in benchmark, f"Benchmark in provider '{provider_name}' missing 'name'"
                assert "category" in benchmark, f"Benchmark in provider '{provider_name}' missing 'category'"

    def test_lm_evaluation_harness_provider_exists(
        self,
        model_namespace: Namespace,
        evalhub_scoped_token: str,
        evalhub_providers_role_binding: RoleBinding,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
    ) -> None:
        """Verify that the lm_evaluation_harness provider is registered and has benchmarks."""
        data = list_evalhub_providers(
            host=evalhub_route.host,
            token=evalhub_scoped_token,
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant=model_namespace.name,
        )

        provider_ids = [p["resource"]["id"] for p in data["items"]]
        assert "lm_evaluation_harness" in provider_ids, (
            f"Expected 'lm_evaluation_harness' in providers, got: {provider_ids}"
        )

        lmeval_provider = next(p for p in data["items"] if p["resource"]["id"] == "lm_evaluation_harness")
        assert len(lmeval_provider["benchmarks"]) > 0, (
            "lm_evaluation_harness provider should have at least one benchmark"
        )

    def test_get_single_provider(
        self,
        model_namespace: Namespace,
        evalhub_scoped_token: str,
        evalhub_providers_role_binding: RoleBinding,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
    ) -> None:
        """Verify that a single provider can be retrieved by ID."""
        providers = list_evalhub_providers(
            host=evalhub_route.host,
            token=evalhub_scoped_token,
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant=model_namespace.name,
        )
        first_provider_id = providers["items"][0]["resource"]["id"]

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
        """Verify that requesting a non-existent provider ID returns an HTTP error."""
        with pytest.raises(requests.exceptions.HTTPError):
            get_evalhub_provider(
                host=evalhub_route.host,
                token=evalhub_scoped_token,
                ca_bundle_file=evalhub_ca_bundle_file,
                provider_id="nonexistent-provider-id",
                tenant=model_namespace.name,
            )


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
    ) -> None:
        """Verify that a user without providers ClusterRole binding cannot get a provider."""
        with pytest.raises(requests.exceptions.HTTPError, match="403"):
            get_evalhub_provider(
                host=evalhub_route.host,
                token=evalhub_unauthorised_token,
                ca_bundle_file=evalhub_ca_bundle_file,
                provider_id="lm_evaluation_harness",
                tenant=model_namespace.name,
            )
