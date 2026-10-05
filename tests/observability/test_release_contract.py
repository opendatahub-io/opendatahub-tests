"""Integrated release-contract assertions for observability routes and namespace isolation."""

from collections.abc import Callable

import pytest

from tests.observability.assertions import (
    assert_authorization_response,
    assert_capability_is_available,
    assert_capability_unavailable,
    assert_namespace_isolation,
    assert_query_contract,
)
from tests.observability.authorization import SubjectAccessReviewResult
from tests.observability.contract import ContractRecord, ReleaseContract, ensure_authorization_reviewed
from tests.observability.fixtures import NamespacePair, wait_for_source_metric
from tests.observability.personas import Persona
from tests.observability.preflight import PreflightReport
from tests.observability.query import RawQueryClient, RawQueryResult, build_contract_request

pytestmark = [pytest.mark.tier1, pytest.mark.metrics]


class TestObservabilityReleaseContract:
    """Validate release dashboard queries, source telemetry, capabilities, and namespace boundaries.

    The release runner supplies route bases, independent persona tokens, and the model fixture configuration.
    Missing inputs are reported as explicit blocked/failed fixture outcomes rather than skipped product assertions.
    """

    def test_admin_data_contract(
        self,
        release_contract: ReleaseContract,
        release_preflight: PreflightReport,
        observability_namespaces: NamespacePair,
        observability_models: list[object],
        observability_fixture_resources: dict[str, object],
        observability_query_clients: dict[str, RawQueryClient],
        observability_personas: tuple[Persona, ...],
        observability_persona_tokens: dict[str, str],
        observability_evidence: Callable[..., None],
    ) -> None:
        """Given ready source fixtures and an administrator token, validate every shipped data panel response."""
        del release_preflight, observability_models
        assert observability_fixture_resources["namespaces"]
        admin = next(persona for persona in observability_personas if persona.name == "cluster-admin")
        for record in release_contract.records:
            if record.capability != "shipped" or record.authorization_response != "not-applicable":
                continue
            client = _client_for_record(datasource=record.datasource, clients=observability_query_clients)
            result = client.query(
                request=build_contract_request(
                    contract=record,
                    persona=admin.name,
                    principal=admin.principal,
                    requested_namespace=observability_namespaces.namespace_a.name,
                    fixture_namespace=observability_namespaces.namespace_a.name,
                    variables={"namespace": observability_namespaces.namespace_a.name or "", "model": "model-a"},
                ),
                bearer_token=observability_persona_tokens[admin.name],
            )
            _assert_query_with_evidence(
                test_identifier="test_admin_data_contract",
                contract_record=record,
                persona=admin,
                fixture_resources=observability_fixture_resources,
                query=result,
                evidence=observability_evidence,
            )

    def test_gpu_source_and_recording_series_are_active(
        self,
        release_contract: ReleaseContract,
        observability_namespaces: NamespacePair,
        observability_models: list[object],
        observability_query_clients: dict[str, RawQueryClient],
        observability_personas: tuple[Persona, ...],
        observability_persona_tokens: dict[str, str],
        observability_fixture_resources: dict[str, object],
        observability_evidence: Callable[..., None],
    ) -> None:
        """Given a scheduled GPU model, prove source and recording telemetry before dashboard assertions."""
        del observability_models
        admin = next(persona for persona in observability_personas if persona.name == "cluster-admin")
        identifiers = {"accelerator-dcgm-source", "accelerator-gpu-utilization", "accelerator-memory-used"}
        for record in (item for item in release_contract.records if item.identifier in identifiers):
            client = _client_for_record(datasource=record.datasource, clients=observability_query_clients)
            request = build_contract_request(
                contract=record,
                persona=admin.name,
                principal=admin.principal,
                requested_namespace=observability_namespaces.namespace_a.name,
                fixture_namespace=observability_namespaces.namespace_a.name,
                variables={"namespace": observability_namespaces.namespace_a.name or "", "model": "model-a"},
            )
            result = wait_for_source_metric(
                query=lambda request=request, client=client: client.query(
                    request=request,
                    bearer_token=observability_persona_tokens[admin.name],
                )
            )
            _assert_query_with_evidence(
                test_identifier="test_gpu_source_and_recording_series_are_active",
                contract_record=record,
                persona=admin,
                fixture_resources=observability_fixture_resources,
                query=result,
                evidence=observability_evidence,
            )
            has_positive_value = False
            for series in result.series:
                for _timestamp, value in series.values:
                    try:
                        if float(value) > 0:
                            has_positive_value = True
                            break
                    except TypeError, ValueError:
                        continue
                if has_positive_value:
                    break
            assert has_positive_value, f"{record.identifier}: expected a numeric telemetry value greater than zero"

    def test_maas_usage_metrics_are_separate_from_unshipped_capabilities(
        self,
        release_contract: ReleaseContract,
        observability_namespaces: NamespacePair,
        observability_models: list[object],
        observability_query_clients: dict[str, RawQueryClient],
        observability_personas: tuple[Persona, ...],
        observability_persona_tokens: dict[str, str],
        observability_fixture_resources: dict[str, object],
        observability_evidence: Callable[..., None],
    ) -> None:
        """Given MaaS telemetry, validate request/token series and independently declare breakdown/showback status."""
        del observability_models
        admin = next(persona for persona in observability_personas if persona.name == "cluster-admin")
        for record in (
            item for item in release_contract.records if item.dashboard == "maas" and item.capability == "shipped"
        ):
            client = _client_for_record(datasource=record.datasource, clients=observability_query_clients)
            result = client.query(
                request=build_contract_request(
                    contract=record,
                    persona=admin.name,
                    principal=admin.principal,
                    requested_namespace=observability_namespaces.namespace_a.name,
                    fixture_namespace=observability_namespaces.namespace_a.name,
                    variables={"namespace": observability_namespaces.namespace_a.name or "", "model": "model-a"},
                ),
                bearer_token=observability_persona_tokens[admin.name],
            )
            assert_capability_is_available(contract=record)
            _assert_query_with_evidence(
                test_identifier="test_maas_usage_metrics_are_separate_from_unshipped_capabilities",
                contract_record=record,
                persona=admin,
                fixture_resources=observability_fixture_resources,
                query=result,
                evidence=observability_evidence,
            )

        for record in release_contract.records:
            if record.dashboard == "maas" and record.capability != "shipped":
                assert_capability_unavailable(contract=record)

    def test_gpu_aas_regression_is_explicitly_environment_blocked(
        self,
        release_contract: ReleaseContract,
    ) -> None:
        """Given no reviewed GPUaaS route dependency, report an environment block instead of a product pass."""
        assert_capability_unavailable(contract=release_contract.record(identifier="gpu-aas-regression"))

    @pytest.mark.parametrize(
        "identifier",
        [
            pytest.param("namespace-proxy", id="test_namespace_proxy"),
            pytest.param("tenancy-endpoint", id="test_tenancy_endpoint"),
            pytest.param("data-science-thanos", id="test_data_science_thanos"),
        ],
    )
    def test_namespace_queries_require_reviewed_authorization_contract(
        self,
        identifier: str,
        release_contract: ReleaseContract,
        observability_namespaces: NamespacePair,
        observability_personas: tuple[Persona, ...],
        observability_persona_tokens: dict[str, str],
        observability_query_clients: dict[str, RawQueryClient],
        observability_fixture_resources: dict[str, object],
        observability_evidence: Callable[..., None],
    ) -> None:
        """Given restricted personas, refuse tampered namespace assertions until denial behavior is reviewed."""
        record = release_contract.record(identifier=identifier)
        ensure_authorization_reviewed(record=record)
        client = _client_for_record(datasource=record.datasource, clients=observability_query_clients)
        namespace_a = observability_namespaces.namespace_a.name or ""
        namespace_b = observability_namespaces.namespace_b.name or ""
        for persona in observability_personas:
            for requested_namespace in (namespace_a, namespace_b, "tampered-observability-namespace"):
                result = client.query(
                    request=build_contract_request(
                        contract=record,
                        persona=persona.name,
                        principal=persona.principal,
                        requested_namespace=requested_namespace,
                        fixture_namespace=namespace_a,
                        variables={"namespace": requested_namespace, "model": "model-a"},
                    ),
                    bearer_token=observability_persona_tokens[persona.name],
                )
                try:
                    assert_authorization_response(result=result, expected=record.authorization_response)
                    assert_namespace_isolation(result=result, allowed_namespaces=set(persona.namespaces))
                except AssertionError:
                    observability_evidence(
                        test_identifier=f"test_namespace_queries_require_reviewed_authorization_contract[{identifier}]",
                        contract_record=record,
                        persona=persona,
                        fixture_resources=observability_fixture_resources,
                        query=result,
                        failure_category="authorization-contract",
                    )
                    raise
                else:
                    observability_evidence(
                        test_identifier=f"test_namespace_queries_require_reviewed_authorization_contract[{identifier}]",
                        contract_record=record,
                        persona=persona,
                        fixture_resources=observability_fixture_resources,
                        query=result,
                    )

    def test_persona_sar_baseline_has_no_evaluation_errors(
        self,
        observability_sar_baseline: tuple[SubjectAccessReviewResult, ...],
    ) -> None:
        """Given independent persona identities, verify every baseline SAR was evaluated by Kubernetes."""
        assert all(result.evaluation_error is None for result in observability_sar_baseline)


def _client_for_record(datasource: str, clients: dict[str, RawQueryClient]) -> RawQueryClient:
    try:
        return clients[datasource]
    except KeyError as error:
        raise AssertionError(f"no release-runner route configured for datasource {datasource!r}") from error


def _assert_query_with_evidence(
    *,
    test_identifier: str,
    contract_record: ContractRecord,
    persona: Persona,
    fixture_resources: dict[str, object],
    query: RawQueryResult,
    evidence: Callable[..., None],
) -> None:
    """Assert one query and retain a sanitized pass or failure record."""
    try:
        assert_query_contract(result=query, contract=contract_record)
    except AssertionError:
        evidence(
            test_identifier=test_identifier,
            contract_record=contract_record,
            persona=persona,
            fixture_resources=fixture_resources,
            query=query,
            failure_category="query-contract",
        )
        raise
    else:
        evidence(
            test_identifier=test_identifier,
            contract_record=contract_record,
            persona=persona,
            fixture_resources=fixture_resources,
            query=query,
        )
