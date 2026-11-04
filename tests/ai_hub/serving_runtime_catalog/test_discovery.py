"""Runtime discovery API coverage for RHOAIENG-96061 and RHAISTRAT-1988."""

from typing import Any

import pytest
import requests

from tests.ai_hub.serving_runtime_catalog.constants import (
    LABEL_ALPHA,
    LABEL_BETA,
    LOGO_BYTES,
    LOGO_URL,
    SOURCE_ALPHA,
    SOURCE_BETA,
    SOURCE_UNLABELED,
    SUPPORT_LEVELS,
    TEST_CATALOGS,
)
from tests.ai_hub.serving_runtime_catalog.utils import (
    assert_metadata,
    list_catalog_items,
    runtime_by_name,
    runtime_snapshot,
)
from tests.ai_hub.utils import execute_get_command

pytestmark = [pytest.mark.usefixtures("updated_dsc_component_state_scope_session")]


@pytest.mark.usefixtures("runtime_sources")
class TestRuntimeDiscovery:
    """Exact discovery results from controlled sources, including cross-source name collisions."""

    @pytest.mark.tier1
    def test_runtime_details_and_version_ownership(
        self, runtime_api_url: str, runtime_rest_headers: dict[str, str]
    ) -> None:
        """Given colliding source names, when details and versions are read, then each record retains its owner."""
        snapshot = runtime_snapshot(url=runtime_api_url, headers=runtime_rest_headers, source_ids=list(TEST_CATALOGS))
        runtime_ids = [runtime["id"] for runtime in snapshot.values()]
        version_ids: list[str] = []
        assert len(set(runtime_ids)) == len(runtime_ids)
        for (source_id, name), runtime in snapshot.items():
            detail = execute_get_command(
                url=f"{runtime_api_url}serving_runtimes/{runtime['id']}", headers=runtime_rest_headers
            )
            expected = next(entry for entry in TEST_CATALOGS[source_id]["serving_runtimes"] if entry["name"] == name)
            assert detail["id"] == runtime["id"]
            assert detail["sourceId"] == source_id
            assert_metadata(actual=detail, expected=expected)
            for version in runtime["versions"]:
                assert version["artifactType"] == "serving-runtime-version"
                version_ids.append(version["id"])
        assert len(set(version_ids)) == len(version_ids), "A version belongs to more than one runtime"

    @pytest.mark.tier2
    @pytest.mark.parametrize(
        "query, names",
        [
            pytest.param({"name": "catalog-alpha"}, {"catalog-alpha"}, id="test_exact_name"),
            pytest.param({"name": "catalog-b%"}, {"catalog-bravo"}, id="test_name_pattern"),
            pytest.param({"q": "CATALOG-ALPHA"}, {"catalog-alpha"}, id="test_keyword_name"),
            pytest.param({"q": "QUARTZ"}, {"catalog-alpha"}, id="test_keyword_display_name"),
            pytest.param({"q": "ACME"}, {"catalog-alpha", "catalog-charlie"}, id="test_keyword_provider"),
            pytest.param({"q": "ZEPHYR"}, {"catalog-alpha"}, id="test_keyword_description"),
            pytest.param({"q": "no-runtime-matches-this"}, set(), id="test_unmatched_keyword"),
            pytest.param({"q": "%"}, set(), id="test_keyword_percent_is_literal"),
            pytest.param({"filterQuery": "name = 'catalog-alpha'"}, {"catalog-alpha"}, id="test_filter_name"),
            pytest.param(
                {"filterQuery": "provider = 'Acme inference'"},
                {"catalog-alpha", "catalog-charlie"},
                id="test_scalar_filter",
            ),
            pytest.param({"filterQuery": "capabilities.requiresGPU = true"}, {"catalog-alpha"}, id="test_gpu_filter"),
            pytest.param(
                {"filterQuery": "capabilities.multiModel = true"}, {"catalog-bravo"}, id="test_multimodel_filter"
            ),
            pytest.param(
                {"filterQuery": "supportedModelFormats = 'onnx'"},
                {"catalog-alpha", "catalog-bravo"},
                id="test_model_format_filter",
            ),
            pytest.param({"filterQuery": "tags = 'generative'"}, {"catalog-alpha"}, id="test_tag_filter"),
            pytest.param(
                {"filterQuery": "capabilities.supportedAccelerators = 'nvidia.com/gpu'"},
                {"catalog-alpha"},
                id="test_accelerator_filter",
            ),
            pytest.param(
                {"filterQuery": "test_owner.string_value = 'QE'"},
                {"catalog-alpha"},
                id="test_custom_property_filter",
            ),
            pytest.param(
                {"filterQuery": "provider = 'Acme inference' AND capabilities.requiresGPU = false"},
                {"catalog-charlie"},
                id="test_combined_filter",
            ),
        ],
    )
    def test_runtime_queries(
        self,
        runtime_api_url: str,
        runtime_rest_headers: dict[str, str],
        query: dict[str, str],
        names: set[str],
    ) -> None:
        """Given varied runtimes, when queried, then exactly the matching identities are returned."""
        items = list_catalog_items(
            url=f"{runtime_api_url}serving_runtimes",
            headers=runtime_rest_headers,
            params={"source": SOURCE_ALPHA, **query},
        )
        assert {(item["sourceId"], item["name"]) for item in items} == {(SOURCE_ALPHA, name) for name in names}
        assert len(items) == len(names), "Discovery returned duplicate records"

    @pytest.mark.tier2
    @pytest.mark.parametrize(
        "query, identities",
        [
            pytest.param({"source": SOURCE_BETA}, {(SOURCE_BETA, "catalog-alpha")}, id="test_source_id"),
            pytest.param(
                {"source": f"{SOURCE_BETA},{SOURCE_UNLABELED}"},
                {(SOURCE_BETA, "catalog-alpha"), (SOURCE_UNLABELED, "catalog-unlabeled")},
                id="test_multiple_sources",
            ),
            pytest.param(
                {"sourceLabel": LABEL_ALPHA.swapcase()},
                {(SOURCE_ALPHA, "catalog-alpha"), (SOURCE_ALPHA, "catalog-bravo"), (SOURCE_ALPHA, "catalog-charlie")},
                id="test_label_case_insensitive",
            ),
            pytest.param(
                {"sourceLabel": f"{LABEL_ALPHA},{LABEL_BETA}"},
                {
                    (SOURCE_ALPHA, "catalog-alpha"),
                    (SOURCE_ALPHA, "catalog-bravo"),
                    (SOURCE_ALPHA, "catalog-charlie"),
                    (SOURCE_BETA, "catalog-alpha"),
                },
                id="test_multiple_labels",
            ),
            pytest.param(
                {"sourceLabel": "null"}, {(SOURCE_UNLABELED, "catalog-unlabeled")}, id="test_unlabeled_source"
            ),
            pytest.param({"sourceLabel": "no-runtime-source-label"}, set(), id="test_unknown_label"),
            pytest.param({"source": "no-runtime-source"}, set(), id="test_unknown_source"),
        ],
    )
    def test_source_selection(
        self,
        runtime_api_url: str,
        runtime_rest_headers: dict[str, str],
        query: dict[str, str],
        identities: set[tuple[str, str]],
    ) -> None:
        """Given labeled and unlabeled sources, when selected, then only the intended test assets are returned."""
        items = list_catalog_items(
            url=f"{runtime_api_url}serving_runtimes",
            headers=runtime_rest_headers,
            params={"filterQuery": "tags = 'catalog-test'", **query},
        )
        assert {(item["sourceId"], item["name"]) for item in items} == identities
        assert len(items) == len(identities)

    @pytest.mark.tier2
    @pytest.mark.parametrize(
        "expression, versions",
        [
            pytest.param("version = '2.0.0'", {"2.0.0"}, id="test_version_pin"),
            *[
                pytest.param(f"supportLevel = '{level}'", {f"{index + 1}.0.0"}, id=f"test_support_level_{level}")
                for index, level in enumerate(SUPPORT_LEVELS)
            ],
            pytest.param("deprecated = true", {"4.0.0"}, id="test_deprecated_versions"),
            pytest.param(
                "deprecated = false AND supportLevel = 'supported'", {"1.0.0"}, id="test_combined_version_filter"
            ),
            pytest.param("version = '99.0.0'", set(), id="test_unmatched_version"),
        ],
    )
    def test_version_filters(
        self, runtime_api_url: str, runtime_rest_headers: dict[str, str], expression: str, versions: set[str]
    ) -> None:
        """Given multiple runtime versions, when filtered, then only selected versions of that parent are returned."""
        parent = runtime_by_name(
            url=runtime_api_url, headers=runtime_rest_headers, source_id=SOURCE_ALPHA, name="catalog-alpha"
        )
        items = list_catalog_items(
            url=f"{runtime_api_url}serving_runtimes/{parent['id']}/versions",
            headers=runtime_rest_headers,
            params={"filterQuery": expression},
        )
        assert {item["version"] for item in items} == versions
        assert len(items) == len(versions)

    @pytest.mark.tier2
    @pytest.mark.parametrize(
        "sort_order", [pytest.param("ASC", id="test_ascending"), pytest.param("DESC", id="test_descending")]
    )
    @pytest.mark.parametrize(
        "entity", [pytest.param("runtimes", id="test_runtimes"), pytest.param("versions", id="test_versions")]
    )
    def test_filtered_ordered_pagination(
        self, runtime_api_url: str, runtime_rest_headers: dict[str, str], sort_order: str, entity: str
    ) -> None:
        """Given filtered entities, when traversing ordered pages, then every match appears once in order."""
        endpoint = f"{runtime_api_url}serving_runtimes"
        params: dict[str, Any] = {"orderBy": "NAME", "sortOrder": sort_order, "pageSize": 1}
        if entity == "runtimes":
            params.update({"source": f"{SOURCE_ALPHA},{SOURCE_BETA}", "filterQuery": "provider != 'Beta inference'"})
            expected = ["catalog-alpha", "catalog-alpha", "catalog-charlie"]
            field = "name"
        else:
            parent = runtime_by_name(
                url=runtime_api_url, headers=runtime_rest_headers, source_id=SOURCE_ALPHA, name="catalog-alpha"
            )
            endpoint = f"{endpoint}/{parent['id']}/versions"
            params["filterQuery"] = "deprecated = false"
            expected = ["1.0.0", "2.0.0", "3.0.0"]
            field = "version"
        first_page = execute_get_command(url=endpoint, headers=runtime_rest_headers, params=params)
        assert len(first_page["items"]) == 1
        assert first_page.get("nextPageToken"), "Expected multiple pages"
        items = list_catalog_items(url=endpoint, headers=runtime_rest_headers, params=params)
        assert [item[field] for item in items] == sorted(expected, reverse=sort_order == "DESC")
        assert len({item["id"] for item in items}) == len(expected)
        if entity == "runtimes":
            assert {(item["sourceId"], item["name"]) for item in items} == {
                (SOURCE_ALPHA, "catalog-alpha"),
                (SOURCE_BETA, "catalog-alpha"),
                (SOURCE_ALPHA, "catalog-charlie"),
            }

    @pytest.mark.tier2
    def test_filter_options(self, runtime_api_url: str, runtime_rest_headers: dict[str, str]) -> None:
        """Given persisted properties, when filter options are read, then controlled values are present."""
        filters = execute_get_command(
            url=f"{runtime_api_url}serving_runtimes/filter_options", headers=runtime_rest_headers
        )["filters"]
        assert {"Acme inference", "Beta inference", "Separate source"} <= set(filters["provider"]["values"])
        assert set(filters["capabilities.requiresGPU"]["values"]) == {False, True}
        assert set(filters["capabilities.multiModel"]["values"]) == {False, True}

    @pytest.mark.tier2
    @pytest.mark.parametrize(
        "name, status_code",
        [
            pytest.param("catalog-alpha", 200, id="test_inline_logo"),
            pytest.param("catalog-bravo", 302, id="test_external_logo"),
            pytest.param("catalog-charlie", 404, id="test_missing_logo"),
        ],
    )
    def test_runtime_logos(
        self, runtime_api_url: str, runtime_rest_headers: dict[str, str], name: str, status_code: int
    ) -> None:
        """Given inline, remote, or absent logos, when requested, then bytes, redirects, or not-found are returned."""
        parent = runtime_by_name(url=runtime_api_url, headers=runtime_rest_headers, source_id=SOURCE_ALPHA, name=name)
        response = requests.get(
            url=f"{runtime_api_url}serving_runtimes/{parent['id']}/logo",
            headers=runtime_rest_headers,
            verify=False,
            timeout=30,
            allow_redirects=False,
        )
        assert response.status_code == status_code
        if status_code == 200:
            assert response.content == LOGO_BYTES
            assert response.headers["Content-Type"].split(";")[0] == "image/svg+xml"
        elif status_code == 302:
            assert response.headers["Location"] == LOGO_URL
