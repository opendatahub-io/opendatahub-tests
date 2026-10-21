"""Public HTTP error contracts for serving runtime discovery (RHOAIENG-96064)."""

import re

import pytest
import requests

from tests.ai_hub.serving_runtime_catalog.constants import ABSENT_RUNTIME_ID, LABEL_ALPHA, SOURCE_ALPHA
from tests.ai_hub.serving_runtime_catalog.utils import list_catalog_items, runtime_by_name

pytestmark = [pytest.mark.tier3, pytest.mark.usefixtures("updated_dsc_component_state_scope_session")]


@pytest.mark.usefixtures("runtime_sources")
class TestRuntimeCatalogErrors:
    """Malformed requests must report their specific HTTP error instead of returning discovery data."""

    @pytest.mark.parametrize(
        "params, message_field",
        [
            pytest.param({"filterQuery": "INVALID SYNTAX !!!"}, "invalid filter query", id="test_invalid_filter"),
            pytest.param({"pageSize": "abc"}, "pageSize", id="test_nonnumeric_page_size"),
            pytest.param({"pageSize": "0"}, "pageSize", id="test_zero_page_size"),
            pytest.param({"pageSize": "-1"}, "pageSize", id="test_negative_page_size"),
            pytest.param({"nextPageToken": "bogus_token"}, "nextPageToken", id="test_invalid_page_token"),
            pytest.param({"orderBy": "unknown"}, "orderBy", id="test_invalid_ordering"),
            pytest.param({"orderBy": "RECOMMENDED"}, "orderBy", id="test_unsupported_ordering"),
            pytest.param({"sortOrder": "unknown"}, "sortOrder", id="test_invalid_sort_order"),
        ],
    )
    @pytest.mark.parametrize(
        "versions", [pytest.param(False, id="test_runtime_list"), pytest.param(True, id="test_version_list")]
    )
    def test_invalid_query_returns_bad_request(
        self,
        runtime_api_url: str,
        runtime_rest_headers: dict[str, str],
        params: dict[str, str],
        message_field: str,
        versions: bool,
    ) -> None:
        """Given an invalid list query, when requested, then HTTP 400 identifies the invalid input."""
        endpoint = f"{runtime_api_url}serving_runtimes"
        if versions:
            parent = runtime_by_name(
                url=runtime_api_url, headers=runtime_rest_headers, source_id=SOURCE_ALPHA, name="catalog-alpha"
            )
            endpoint = f"{endpoint}/{parent['id']}/versions"
        response = requests.get(url=endpoint, headers=runtime_rest_headers, params=params, verify=False, timeout=30)
        assert response.status_code == 400, f"Expected HTTP 400, got {response.status_code}"
        error = response.json()
        assert error["message"], "Error response has no diagnostic message"
        assert message_field in error["message"]

    def test_source_and_label_are_mutually_exclusive(
        self, runtime_api_url: str, runtime_rest_headers: dict[str, str]
    ) -> None:
        """Given both source selectors, when listing runtimes, then HTTP 400 explains their incompatibility."""
        response = requests.get(
            url=f"{runtime_api_url}serving_runtimes",
            headers=runtime_rest_headers,
            params={"source": SOURCE_ALPHA, "sourceLabel": LABEL_ALPHA},
            verify=False,
            timeout=30,
        )
        assert response.status_code == 400
        message = response.json()["message"]
        assert re.search(r"\bsource\b", message), f"Message does not name source: {message}"
        assert "sourceLabel" in message, f"Message does not name sourceLabel: {message}"

    @pytest.mark.parametrize(
        "suffix",
        [
            pytest.param("", id="test_detail"),
            pytest.param("/versions", id="test_versions"),
            pytest.param("/logo", id="test_logo"),
        ],
    )
    def test_nonexistent_runtime_returns_not_found(
        self, runtime_api_url: str, runtime_rest_headers: dict[str, str], suffix: str
    ) -> None:
        """Given an unused runtime ID, when its resources are requested, then HTTP 404 is returned."""
        items = list_catalog_items(url=f"{runtime_api_url}serving_runtimes", headers=runtime_rest_headers)
        assert ABSENT_RUNTIME_ID not in {item["id"] for item in items}, "Chosen absent runtime ID exists"
        response = requests.get(
            url=f"{runtime_api_url}serving_runtimes/{ABSENT_RUNTIME_ID}{suffix}",
            headers=runtime_rest_headers,
            verify=False,
            timeout=30,
            allow_redirects=False,
        )
        assert response.status_code == 404
