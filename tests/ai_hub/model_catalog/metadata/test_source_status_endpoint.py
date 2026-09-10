from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from timeout_sampler import retry

from tests.ai_hub.model_catalog.constants import REDHAT_AI_CATALOG_ID
from tests.ai_hub.utils import execute_delete_call_with_retry, execute_get_command, execute_get_command_with_retry

STATUS_PATH_TEMPLATE: str = "{base}sources/{source_id}/status"
UNKNOWN_SOURCE_ID: str = "does_not_exist_xyz"
ERROR_SOURCE_ID: str = "mixed_models_catalog"
ERROR_SOURCE_YAML: str = f"""
catalogs:
  - name: Source Status Error Catalog
    id: {ERROR_SOURCE_ID}
    type: hf
    enabled: true
    includedModels:
      - "jonburdo/public-test-model-1"
      - "jonburdo/private-test-model-1"
    labels:
      - {ERROR_SOURCE_ID}
"""

pytestmark = [pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace")]


@pytest.fixture(scope="class")
def source_status_base_url(model_catalog_rest_url: list[str]) -> str:
    """Return the v1 base URL used by the source-status endpoint."""
    return model_catalog_rest_url[0].replace("/v1alpha1/", "/v1/")


def get_source_status(base_url: str, headers: dict[str, str], source_id: str) -> dict[str, Any]:
    """Get the persisted status of a catalog source, allowing an empty response."""
    return execute_get_command(
        url=STATUS_PATH_TEMPLATE.format(base=base_url, source_id=source_id),
        headers=headers,
    )


def clear_source_status(base_url: str, headers: dict[str, str], source_id: str) -> int:
    """Clear the persisted status of a catalog source."""
    response = execute_delete_call_with_retry(
        url=STATUS_PATH_TEMPLATE.format(base=base_url, source_id=source_id),
        headers=headers,
    )
    return response.status_code


class SourceStatusNotRestored(Exception):
    pass


@retry(wait_timeout=120, sleep=5, exceptions_dict={SourceStatusNotRestored: []})
def wait_for_source_status_restored(base_url: str, headers: dict[str, str], source_id: str) -> dict[str, Any]:
    """Wait for a catalog source's persisted status to be repopulated after a restart."""
    status = get_source_status(base_url=base_url, headers=headers, source_id=source_id)
    if status.get("status"):
        return status
    raise SourceStatusNotRestored(f"Status for {source_id} not yet restored: {status}")


class TestGetSourceStatusEndpoint:
    """Verify source-status GET behavior for RHOAIENG-88203."""

    @pytest.mark.tier1
    def test_get_status_available_source(
        self,
        source_status_base_url: str,
        model_registry_rest_headers: dict[str, str],
    ) -> None:
        """Given an available source, when its status is read, then it is available without an error."""
        status = get_source_status(
            base_url=source_status_base_url,
            headers=model_registry_rest_headers,
            source_id=REDHAT_AI_CATALOG_ID,
        )

        assert status.get("status") == "available", f"Unexpected status for {REDHAT_AI_CATALOG_ID}: {status}"
        assert not status.get("error"), f"Available source should have no error: {status}"

    @pytest.mark.parametrize(
        "asset_type",
        [
            pytest.param(None, id="test_models"),
            pytest.param("mcp_servers", id="test_mcp", marks=pytest.mark.skip_on_disconnected),
        ],
    )
    @pytest.mark.tier1
    def test_get_status_agrees_with_listing(
        self,
        asset_type: str | None,
        source_status_base_url: str,
        model_registry_rest_headers: dict[str, str],
    ) -> None:
        """Given model or MCP sources, when their statuses are read, then they match the source listing."""
        params = {"assetType": asset_type} if asset_type else None
        listing = execute_get_command_with_retry(
            url=f"{source_status_base_url}sources",
            headers=model_registry_rest_headers,
            params=params,
        )
        sources = listing.get("items", [])
        assert sources, f"No sources returned for assetType={asset_type}"

        mismatches: list[str] = []
        for source in sources:
            status = get_source_status(
                base_url=source_status_base_url,
                headers=model_registry_rest_headers,
                source_id=source["id"],
            )
            if status.get("status") != source.get("status") or status.get("error") != source.get("error"):
                mismatches.append(
                    f"{source['id']}: /sources=({source.get('status')!r}, {source.get('error')!r}) "
                    f"/status=({status.get('status')!r}, {status.get('error')!r})"
                )
        assert not mismatches, f"Status endpoint disagrees with /sources: {mismatches}"

    @pytest.mark.tier1
    def test_get_status_unknown_source_returns_empty(
        self,
        source_status_base_url: str,
        model_registry_rest_headers: dict[str, str],
    ) -> None:
        """Given an unknown source, when its status is read, then the successful response is empty."""
        status = get_source_status(
            base_url=source_status_base_url,
            headers=model_registry_rest_headers,
            source_id=UNKNOWN_SOURCE_ID,
        )

        assert not status.get("status"), f"Unknown source should have empty status: {status}"
        assert not status.get("error"), f"Unknown source should have empty error: {status}"


@pytest.mark.skip_on_disconnected
@pytest.mark.parametrize(
    "updated_catalog_config_map",
    [pytest.param({"sources_yaml": ERROR_SOURCE_YAML}, id="test_hf_error_source")],
    indirect=["updated_catalog_config_map"],
)
class TestGetErrorSourceStatus:
    """Verify error source-status GET behavior for RHOAIENG-88203."""

    @pytest.mark.tier2
    def test_get_status_error_source_matches_listing(
        self,
        updated_catalog_config_map: ConfigMap,
        source_status_base_url: str,
        model_registry_rest_headers: dict[str, str],
    ) -> None:
        """Given a source that fails to load, when its status is read, then it matches the listed error."""
        listing = execute_get_command_with_retry(
            url=f"{source_status_base_url}sources",
            headers=model_registry_rest_headers,
        )
        matched_sources = [source for source in listing["items"] if source["id"] == ERROR_SOURCE_ID]
        assert matched_sources, f"Error source {ERROR_SOURCE_ID} not found in /sources: {listing}"
        listed_source = matched_sources[0]
        assert listed_source.get("status") and listed_source["status"] != "available", (
            f"Expected a non-available status for the error source: {listed_source}"
        )

        status = get_source_status(
            base_url=source_status_base_url,
            headers=model_registry_rest_headers,
            source_id=ERROR_SOURCE_ID,
        )
        assert status.get("status") == listed_source.get("status"), (
            f"Status endpoint {status.get('status')!r} != /sources {listed_source.get('status')!r}"
        )
        assert status.get("error") == listed_source.get("error"), (
            f"Status endpoint error {status.get('error')!r} != /sources {listed_source.get('error')!r}"
        )
        assert status.get("error"), f"Error source status should include an error message: {status}"


class TestClearSourceStatusEndpoint:
    """Verify source-status DELETE behavior for RHOAIENG-88203."""

    @pytest.fixture(scope="class")
    def restore_catalog_status(
        self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        source_status_base_url: str,
        model_registry_rest_headers: dict[str, str],
    ) -> Generator[None]:
        """Restore persisted catalog statuses after status-clearing tests."""
        from tests.ai_hub.model_catalog.utils import wait_for_model_catalog_api
        from tests.ai_hub.utils import wait_for_model_catalog_pod_ready_after_deletion

        yield
        wait_for_model_catalog_pod_ready_after_deletion(
            client=admin_client,
            model_registry_namespace=model_registry_namespace,
        )
        wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)
        wait_for_source_status_restored(
            base_url=source_status_base_url,
            headers=model_registry_rest_headers,
            source_id=REDHAT_AI_CATALOG_ID,
        )

    @pytest.mark.tier2
    def test_clear_unknown_source_is_noop(
        self,
        source_status_base_url: str,
        model_registry_rest_headers: dict[str, str],
    ) -> None:
        """Given an unknown source, when its status is cleared, then the operation succeeds and remains empty."""
        status_code = clear_source_status(
            base_url=source_status_base_url,
            headers=model_registry_rest_headers,
            source_id=UNKNOWN_SOURCE_ID,
        )
        assert status_code == 204, f"Expected 204 clearing unknown source, got {status_code}"

        status = get_source_status(
            base_url=source_status_base_url,
            headers=model_registry_rest_headers,
            source_id=UNKNOWN_SOURCE_ID,
        )
        assert not status.get("status") and not status.get("error"), f"Unknown source still has status: {status}"

    @pytest.mark.tier3
    def test_clear_then_get_returns_empty(
        self,
        restore_catalog_status: None,
        source_status_base_url: str,
        model_registry_rest_headers: dict[str, str],
    ) -> None:
        """Given a persisted source status, when it is cleared, then the following read is empty."""
        before = get_source_status(
            base_url=source_status_base_url,
            headers=model_registry_rest_headers,
            source_id=REDHAT_AI_CATALOG_ID,
        )
        assert before.get("status"), f"Expected {REDHAT_AI_CATALOG_ID} to have a persisted status: {before}"

        status_code = clear_source_status(
            base_url=source_status_base_url,
            headers=model_registry_rest_headers,
            source_id=REDHAT_AI_CATALOG_ID,
        )
        assert status_code == 204, f"Expected 204 clearing status, got {status_code}"

        status = get_source_status(
            base_url=source_status_base_url,
            headers=model_registry_rest_headers,
            source_id=REDHAT_AI_CATALOG_ID,
        )
        assert not status.get("status"), f"Status should be empty right after clearing: {status}"
        assert not status.get("error"), f"Error should be empty right after clearing: {status}"
