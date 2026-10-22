"""Source reload and validation integration coverage for RHOAIENG-96060/96062/96063."""

import re
from copy import deepcopy
from sys import exc_info

import pytest
import yaml
from ocp_resources.config_map import ConfigMap
from ocp_resources.pod import Pod
from timeout_sampler import TimeoutSampler

from tests.ai_hub.model_catalog.metadata.utils import clear_source_status, get_source_status
from tests.ai_hub.serving_runtime_catalog.constants import SOURCE_ALPHA, SOURCE_BETA, TEST_CATALOGS, TEST_SOURCES
from tests.ai_hub.serving_runtime_catalog.utils import (
    LOGGER,
    catalog_pod_identity,
    list_catalog_items,
    list_runtime_sources,
    refreshed_runtime_configmap,
    runtime_catalog_patch,
    runtime_snapshot,
    stable_records,
    wait_for_runtime_data,
    wait_for_source_status,
    wait_for_test_sources,
)
from tests.ai_hub.utils import execute_get_command

pytestmark = [pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "runtime_sources")]


@pytest.mark.tier2
class TestRuntimeSourceLifecycle:
    """Runtime and version records reconcile without affecting sibling sources."""

    def test_unchanged_reload_preserves_identity(
        self,
        model_catalog_pod: Pod,
        runtime_source_configmap: ConfigMap,
        runtime_api_url: str,
        runtime_source_url: str,
        runtime_rest_headers: dict[str, str],
    ) -> None:
        """Given loaded data, when equivalent YAML reloads, then IDs remain stable and no duplicate records appear.

        The catalog process must not restart, so a restored `available` status proves the reload happened.
        """
        before = runtime_snapshot(url=runtime_api_url, headers=runtime_rest_headers, source_ids=list(TEST_CATALOGS))
        pod_identity = catalog_pod_identity(pod=model_catalog_pod)
        try:
            assert (
                clear_source_status(base_url=runtime_source_url, headers=runtime_rest_headers, source_id=SOURCE_ALPHA)
                == 204
            )
            assert catalog_pod_identity(pod=model_catalog_pod) == pod_identity, "Catalog restarted after status clear"
            unchanged = runtime_snapshot(
                url=runtime_api_url, headers=runtime_rest_headers, source_ids=list(TEST_CATALOGS)
            )
            assert stable_records(snapshot=unchanged) == stable_records(snapshot=before), (
                "Clearing status changed runtime records"
            )
            with refreshed_runtime_configmap(
                configmap=runtime_source_configmap,
                data={
                    f"{SOURCE_ALPHA}.yaml": "# Equivalent metadata reload\n"
                    + yaml.safe_dump(TEST_CATALOGS[SOURCE_ALPHA])
                },
            ):
                wait_for_source_status(
                    source_url=runtime_source_url,
                    headers=runtime_rest_headers,
                    source_id=SOURCE_ALPHA,
                    expected="available",
                )
                assert catalog_pod_identity(pod=model_catalog_pod) == pod_identity, (
                    "Catalog restarted, so the status was not restored by reloading the source"
                )
                after = wait_for_runtime_data(url=runtime_api_url, headers=runtime_rest_headers, catalogs=TEST_CATALOGS)
                assert {key: item["id"] for key, item in after.items()} == {
                    key: item["id"] for key, item in before.items()
                }
                assert {
                    (key, version["version"]): version["id"]
                    for key, item in after.items()
                    for version in item["versions"]
                } == {
                    (key, version["version"]): version["id"]
                    for key, item in before.items()
                    for version in item["versions"]
                }
        finally:
            original_exception = exc_info()[1]
            try:
                status = get_source_status(
                    base_url=runtime_source_url, headers=runtime_rest_headers, source_id=SOURCE_ALPHA
                )
                if status.get("status") != "available":
                    with refreshed_runtime_configmap(
                        configmap=runtime_source_configmap,
                        data={
                            f"{SOURCE_ALPHA}.yaml": "# Restore runtime source status\n"
                            + yaml.safe_dump(TEST_CATALOGS[SOURCE_ALPHA])
                        },
                    ):
                        wait_for_source_status(
                            source_url=runtime_source_url,
                            headers=runtime_rest_headers,
                            source_id=SOURCE_ALPHA,
                            expected="available",
                        )
                wait_for_test_sources(url=runtime_api_url, source_url=runtime_source_url, headers=runtime_rest_headers)
            except Exception:
                if original_exception is None:
                    raise
                LOGGER.exception("Runtime source status recovery failed after an earlier error")

    @pytest.mark.parametrize(
        "change",
        [
            pytest.param("update_metadata", id="test_runtime_and_version_update"),
            pytest.param("add_version", id="test_version_added"),
            pytest.param("remove_version", id="test_version_removed"),
            pytest.param("remove_runtime", id="test_runtime_removed"),
        ],
    )
    def test_runtime_data_reconciliation(
        self,
        runtime_source_configmap: ConfigMap,
        runtime_api_url: str,
        runtime_source_url: str,
        runtime_rest_headers: dict[str, str],
        change: str,
    ) -> None:
        """Given loaded sources, when metadata changes, then records reconcile and surviving IDs remain stable."""
        before = runtime_snapshot(url=runtime_api_url, headers=runtime_rest_headers, source_ids=list(TEST_CATALOGS))
        expected = deepcopy(x=TEST_CATALOGS)
        runtimes = expected[SOURCE_ALPHA]["serving_runtimes"]
        versions = runtimes[0]["versions"]
        if change == "update_metadata":
            runtimes[0]["description"] = "Updated runtime description"
            runtimes[0]["provider"] = "Reconciled test provider"
            versions[0]["defaultArgs"] = ["--port", "9090"]
            versions[0]["recommendedResources"]["recommended"]["memory"] = "8Gi"
        elif change == "add_version":
            version = deepcopy(versions[0])
            version["version"] = "5.0.0"
            versions.append(version)
        elif change == "remove_version":
            versions.pop(0)
        else:
            runtimes.pop(0)
        with runtime_catalog_patch(
            configmap=runtime_source_configmap,
            url=runtime_api_url,
            source_url=runtime_source_url,
            headers=runtime_rest_headers,
            catalogs={SOURCE_ALPHA: expected[SOURCE_ALPHA]},
        ):
            after = wait_for_runtime_data(url=runtime_api_url, headers=runtime_rest_headers, catalogs=expected)
            for key in before.keys() & after.keys():
                assert after[key]["id"] == before[key]["id"], f"Runtime identity changed for {key}"
                before_versions = {version["version"]: version["id"] for version in before[key]["versions"]}
                for version in after[key]["versions"]:
                    if version["version"] in before_versions:
                        assert version["id"] == before_versions[version["version"]]
            beta_key = (SOURCE_BETA, "catalog-alpha")
            assert stable_records(snapshot=after)[beta_key] == stable_records(snapshot=before)[beta_key]
            if change == "update_metadata":
                for options in TimeoutSampler(
                    wait_timeout=300,
                    sleep=5,
                    func=execute_get_command,
                    url=f"{runtime_api_url}serving_runtimes/filter_options",
                    headers=runtime_rest_headers,
                ):
                    values = options["filters"]["provider"]["values"]
                    if "Reconciled test provider" in values:
                        break

    @pytest.mark.parametrize(
        "operation",
        [pytest.param("disable", id="test_disable_source"), pytest.param("remove", id="test_remove_source")],
    )
    def test_source_disable_and_removal(
        self,
        runtime_source_configmap: ConfigMap,
        runtime_api_url: str,
        runtime_source_url: str,
        runtime_rest_headers: dict[str, str],
        operation: str,
    ) -> None:
        """Given an enabled source, when disabled or removed, then its assets disappear and other sources remain."""
        original_items = list_catalog_items(url=f"{runtime_api_url}serving_runtimes", headers=runtime_rest_headers)
        expected_ids = {item["id"] for item in original_items if item["sourceId"] != SOURCE_ALPHA}
        sources = deepcopy(x=TEST_SOURCES)
        if operation == "disable":
            sources[0]["enabled"] = False
        else:
            sources.pop(0)
        expected = deepcopy(x=TEST_CATALOGS)
        expected[SOURCE_ALPHA] = {"serving_runtimes": []}
        with runtime_catalog_patch(
            configmap=runtime_source_configmap,
            url=runtime_api_url,
            source_url=runtime_source_url,
            headers=runtime_rest_headers,
            sources=sources,
        ):
            wait_for_runtime_data(url=runtime_api_url, headers=runtime_rest_headers, catalogs=expected)
            for items in TimeoutSampler(
                wait_timeout=300,
                sleep=5,
                func=list_catalog_items,
                url=f"{runtime_api_url}serving_runtimes",
                headers=runtime_rest_headers,
            ):
                if {item["id"] for item in items} == expected_ids:
                    break
            listed = list_runtime_sources(source_url=runtime_source_url, headers=runtime_rest_headers)
            selected = [source for source in listed if source["id"] == SOURCE_ALPHA]
            if operation == "disable":
                assert len(selected) == 1 and selected[0]["enabled"] is False
                wait_for_source_status(
                    source_url=runtime_source_url,
                    headers=runtime_rest_headers,
                    source_id=SOURCE_ALPHA,
                    expected="disabled",
                )
            else:
                assert not selected


@pytest.mark.tier3
class TestRuntimeSourceValidation:
    """Invalid source data must preserve previously published runtime families and versions."""

    @pytest.mark.parametrize(
        "invalid_case, field",
        [
            pytest.param("missing_name", "name", id="test_missing_runtime_name"),
            pytest.param("missing_version", "version", id="test_missing_version"),
            pytest.param("missing_image", "image", id="test_missing_image"),
            pytest.param("unknown_field", "unexpectedField", id="test_unknown_field"),
            pytest.param("duplicate_runtime", "name", id="test_duplicate_runtime"),
            pytest.param("duplicate_version", "version", id="test_duplicate_version"),
            pytest.param("invalid_support", "supportLevel", id="test_invalid_support_level"),
            pytest.param("invalid_template", "servingRuntimeTemplate", id="test_invalid_template"),
            pytest.param("secret_default", "defaultValue", id="test_secret_environment_default"),
        ],
    )
    def test_invalid_reload_preserves_valid_data(
        self,
        runtime_source_configmap: ConfigMap,
        runtime_api_url: str,
        runtime_source_url: str,
        runtime_rest_headers: dict[str, str],
        invalid_case: str,
        field: str,
    ) -> None:
        """Given valid data, when an invalid replacement loads, then status identifies the field and data survives."""
        before = runtime_snapshot(url=runtime_api_url, headers=runtime_rest_headers, source_ids=list(TEST_CATALOGS))
        document = deepcopy(x=TEST_CATALOGS[SOURCE_ALPHA])
        runtime = document["serving_runtimes"][0]
        version = runtime["versions"][0]
        if invalid_case == "missing_name":
            del runtime["name"]
        elif invalid_case == "missing_version":
            del version["version"]
        elif invalid_case == "missing_image":
            del version["image"]
        elif invalid_case == "unknown_field":
            runtime["unexpectedField"] = True
        elif invalid_case == "duplicate_runtime":
            document["serving_runtimes"].append(deepcopy(runtime))
        elif invalid_case == "duplicate_version":
            runtime["versions"].append(deepcopy(version))
        elif invalid_case == "invalid_support":
            version["supportLevel"] = "unsupported-value"
        elif invalid_case == "invalid_template":
            version["servingRuntimeTemplate"] = "{invalid json"
        else:
            version["env"][1]["defaultValue"] = "invalid-test-placeholder"
        with runtime_catalog_patch(
            configmap=runtime_source_configmap,
            url=runtime_api_url,
            source_url=runtime_source_url,
            headers=runtime_rest_headers,
            catalogs={SOURCE_ALPHA: document},
        ):
            status = wait_for_source_status(
                source_url=runtime_source_url,
                headers=runtime_rest_headers,
                source_id=SOURCE_ALPHA,
                expected="error",
            )
            assert SOURCE_ALPHA in status["error"]
            assert re.search(rf"\b{re.escape(field)}\b", status["error"]), f"Error does not name {field}"
            after = runtime_snapshot(url=runtime_api_url, headers=runtime_rest_headers, source_ids=list(TEST_CATALOGS))
            assert stable_records(snapshot=after) == stable_records(snapshot=before), (
                "An invalid reload changed previously valid records"
            )
