"""Source lifecycle and metadata assertions shared by runtime catalog tests."""

import json
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from typing import Any
from uuid import uuid4

import requests
import structlog
import yaml
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.config_map import ConfigMap
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_hub.constants import CATALOG_CONTAINER
from tests.ai_hub.model_catalog.metadata.utils import get_source_status
from tests.ai_hub.model_catalog.utils import get_all_catalog_items
from tests.ai_hub.serving_runtime_catalog.constants import TEST_CATALOGS, TEST_SOURCES
from tests.ai_hub.utils import TransientUnauthorizedError, execute_get_command, get_model_catalog_pod

LOGGER = structlog.get_logger(name=__name__)


JSON_FIELDS: tuple[str, ...] = ("servingRuntimeTemplate", "llmInferenceServiceConfig")
UNORDERED_FIELDS: tuple[str, ...] = ("tags", "protocolVersions", "supportedAccelerators")


def as_json(value: Any) -> Any:
    """Decode a JSON string, leaving already-structured values unchanged.

    Args:
        value: A JSON document encoded as a string, or an already-decoded value.

    Returns:
        The decoded document, or `value` itself when it is not a string.
    """
    return json.loads(value) if isinstance(value, str) else value


def assert_metadata(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Compare supplied metadata while allowing API-generated fields.

    Args:
        actual: API resource, including server-generated identity fields.
        expected: Metadata supplied by the YAML source.
    """
    for field, value in expected.items():
        if field == "versions":
            continue
        assert field in actual, f"Missing {field} on {actual.get('name', actual.get('version'))}"
        if field in JSON_FIELDS:
            assert as_json(value=actual[field]) == as_json(value=value), f"Different {field}"
        elif isinstance(value, dict):
            assert_metadata(actual=actual[field], expected=value)
        elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
            assert len(actual[field]) == len(value), f"Different {field} lengths"
            for actual_item, expected_item in zip(actual[field], value, strict=True):
                assert_metadata(actual=actual_item, expected=expected_item)
        elif field in UNORDERED_FIELDS:
            assert sorted(actual[field]) == sorted(value), f"Different {field}: {actual[field]!r} != {value!r}"
        else:
            assert actual[field] == value, f"Different {field}: {actual[field]!r} != {value!r}"


def list_catalog_items(url: str, headers: dict[str, str], params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Read every page of a catalog list endpoint, surfacing the underlying HTTP error.

    The retry helper behind `get_all_catalog_items` wraps every non-connection error in `TimeoutExpiredError`.
    Unwrapping it here lets callers and samplers classify errors such as 401 and 404 by their real type.

    Args:
        url: Catalog list endpoint URL.
        headers: Authentication headers.
        params: Filters applied to every page.

    Returns:
        Items from all pages in API order.

    Raises:
        Exception: The last error seen while retrying, so callers can classify it.
    """
    try:
        return get_all_catalog_items(url=url, headers=headers, params=params)
    except TimeoutExpiredError as error:
        if error.last_exp is not None:
            raise error.last_exp from error
        raise


def runtime_by_name(url: str, headers: dict[str, str], source_id: str, name: str) -> dict[str, Any]:
    """Look up the single runtime with a given name in a source.

    Args:
        url: Serving runtime API base URL.
        headers: Authentication headers.
        source_id: Source that owns the runtime.
        name: Runtime name.

    Returns:
        The runtime record.
    """
    items = list_catalog_items(
        url=f"{url}serving_runtimes", headers=headers, params={"source": source_id, "name": name}
    )
    assert len(items) == 1, f"Expected one runtime {name!r} in {source_id}, found {len(items)}"
    return items[0]


def list_runtime_sources(source_url: str, headers: dict[str, str]) -> list[dict[str, Any]]:
    """List the serving runtime sources.

    Args:
        source_url: Shared source API base URL.
        headers: Authentication headers.

    Returns:
        Source records for the serving runtime asset type.
    """
    return execute_get_command(url=f"{source_url}sources", headers=headers, params={"assetType": "serving_runtimes"})[
        "items"
    ]


def stable_records(snapshot: dict[tuple[str, str], dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Drop server-maintained timestamps so snapshots compare on user-visible data only.

    Args:
        snapshot: Runtime records with versions, as returned by `runtime_snapshot`.

    Returns:
        A copy of the snapshot without `*TimeSinceEpoch` fields on runtimes and versions.
    """

    def without_timestamps(record: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in record.items() if not key.endswith("TimeSinceEpoch")}

    return {
        key: {
            **without_timestamps(record=runtime),
            "versions": [without_timestamps(record=version) for version in runtime["versions"]],
        }
        for key, runtime in snapshot.items()
    }


def runtime_snapshot(
    url: str,
    headers: dict[str, str],
    source_ids: list[str],
    expected_keys: set[tuple[str, str]] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Fetch runtime and version records for specific sources.

    Args:
        url: Serving runtime API base URL.
        headers: Authentication headers.
        source_ids: Source IDs to include.
        expected_keys: Runtime identities (source ID, runtime name) the listing must match. When set, a
            mismatch fails before any version is fetched.

    Returns:
        Runtime records with versions, keyed by source ID and runtime name.
    """
    items = list_catalog_items(url=f"{url}serving_runtimes", headers=headers, params={"source": ",".join(source_ids)})
    keys = [(item["sourceId"], item["name"]) for item in items]
    assert len(set(keys)) == len(keys), f"Duplicate runtimes in {keys}"
    if expected_keys is not None:
        assert set(keys) == expected_keys, f"Runtime identities: {set(keys)} != {expected_keys}"
    with ThreadPoolExecutor() as executor:
        versions = list(
            executor.map(
                lambda item: list_catalog_items(url=f"{url}serving_runtimes/{item['id']}/versions", headers=headers),
                items,
            )
        )
    for item, item_versions in zip(items, versions, strict=True):
        item["versions"] = item_versions
    return dict(zip(keys, items, strict=True))


def wait_for_runtime_data(
    url: str, headers: dict[str, str], catalogs: dict[str, dict[str, Any]]
) -> dict[tuple[str, str], dict[str, Any]]:
    """Wait until runtime and version metadata exactly represents the supplied sources.

    Args:
        url: Serving runtime API base URL.
        headers: Authentication headers.
        catalogs: Expected YAML documents, including empty documents for removed sources.

    Returns:
        Converged runtime snapshot.
    """
    expected = {
        (source_id, runtime["name"]): runtime
        for source_id, document in catalogs.items()
        for runtime in document["serving_runtimes"]
    }

    def matching_snapshot() -> dict[tuple[str, str], dict[str, Any]]:
        """Read one consistent snapshot matching the expected source metadata."""
        snapshot = runtime_snapshot(url=url, headers=headers, source_ids=list(catalogs), expected_keys=set(expected))
        for key, runtime in expected.items():
            actual = snapshot[key]
            assert_metadata(actual=actual, expected=runtime)
            versions = {version["version"]: version for version in actual["versions"]}
            assert len(versions) == len(actual["versions"]), f"Duplicate versions for {key}"
            assert versions.keys() == {version["version"] for version in runtime["versions"]}
            assert actual.get("versionCount") == len(versions), (
                f"Wrong versionCount for {key}: {actual.get('versionCount')!r}"
            )
            for version in runtime["versions"]:
                assert_metadata(actual=versions[version["version"]], expected=version)
        return snapshot

    sampler = TimeoutSampler(
        wait_timeout=300,
        sleep=5,
        func=matching_snapshot,
        exceptions_dict={
            AssertionError: [],
            requests.exceptions.ConnectionError: [],
            TransientUnauthorizedError: [],
            ResourceNotFoundError: [lambda error: ", 404:" in str(error) and "/versions" in str(error)],
        },
        log_context=f"Serving runtime data for {', '.join(catalogs)}",
    )
    return next(iter(sampler))


def wait_for_source_status(source_url: str, headers: dict[str, str], source_id: str, expected: str) -> dict[str, Any]:
    """Wait for a source's operational status.

    Args:
        source_url: Shared source API base URL.
        headers: Authentication headers.
        source_id: Source whose status is observed.
        expected: Expected operational status.

    Returns:
        Persisted status response.
    """

    def matching_status() -> dict[str, Any]:
        """Read the source status once, failing until it equals the expected status."""
        status = get_source_status(base_url=source_url, headers=headers, source_id=source_id)
        assert status.get("status") == expected, f"Source {source_id} status {status.get('status')!r} != {expected!r}"
        return status

    sampler = TimeoutSampler(
        wait_timeout=300,
        sleep=5,
        func=matching_status,
        exceptions_dict={
            AssertionError: [],
            requests.exceptions.ConnectionError: [],
            TransientUnauthorizedError: [],
            ResourceNotFoundError: [lambda error: ", 404:" in str(error)],
        },
        log_context=f"Source {source_id} status {expected}",
    )
    return next(iter(sampler))


def wait_for_original_sources(
    url: str,
    source_url: str,
    headers: dict[str, str],
    baseline: list[dict[str, Any]],
    original_sources: list[dict[str, Any]],
) -> None:
    """Wait until test data is gone and the pre-existing runtimes and sources are healthy again.

    Args:
        url: Serving runtime API base URL.
        source_url: Shared source API base URL.
        headers: Authentication headers.
        baseline: Runtime records listed before the test sources were installed.
        original_sources: Runtime sources listed before the test sources were installed.
    """
    wait_for_runtime_data(
        url=url, headers=headers, catalogs={source_id: {"serving_runtimes": []} for source_id in TEST_CATALOGS}
    )
    settled_source_ids = {source["id"] for source in original_sources if source.get("status") == "available"}
    expected_identities = {
        (item["sourceId"], item["name"]) for item in baseline if item["sourceId"] in settled_source_ids
    }

    def settled_identities() -> set[tuple[str, str]]:
        """List runtimes of the sources that were healthy before setup, failing until they match the baseline."""
        items = list_catalog_items(url=f"{url}serving_runtimes", headers=headers)
        identities = {(item["sourceId"], item["name"]) for item in items if item["sourceId"] in settled_source_ids}
        assert identities == expected_identities, f"Runtime identities: {identities} != {expected_identities}"
        return identities

    next(
        iter(
            TimeoutSampler(
                wait_timeout=300,
                sleep=5,
                func=settled_identities,
                exceptions_dict={
                    AssertionError: [],
                    requests.exceptions.ConnectionError: [],
                    TransientUnauthorizedError: [],
                },
                log_context="Original runtime sources restored",
            )
        )
    )
    for source in original_sources:
        if source.get("status") == "available":
            wait_for_source_status(source_url=source_url, headers=headers, source_id=source["id"], expected="available")


@contextmanager
def refreshed_runtime_configmap(configmap: ConfigMap, data: dict[str, str]) -> Generator[None]:
    """Temporarily patch source data and refresh its pod mounts on entry and exit.

    Args:
        configmap: User source ConfigMap mounted by the catalog pods.
        data: Replacement ConfigMap data entries.

    Yields:
        Control after patching the ConfigMap and triggering a mount refresh.
    """
    pods = get_model_catalog_pod(client=configmap.client, model_registry_namespace=configmap.namespace)
    assert pods, "No catalog pods found"
    annotation = "tests.opendatahub.io/runtime-configmap-refresh"
    patches = {pod: {"metadata": {"annotations": {annotation: str(uuid4())}}} for pod in pods}
    # Restore annotations after the ConfigMap, triggering a refresh of the restored data.
    with ExitStack() as pod_annotations:
        try:
            with ResourceEditor(patches={configmap: {"data": data}}):
                annotation_editor = ResourceEditor(patches=patches)
                pod_annotations.push(exit=annotation_editor)
                annotation_editor.update(backup_resources=True)
                yield
        except BaseException:
            try:
                pod_annotations.close()
            except Exception:
                LOGGER.exception("Catalog pod annotation restore failed after an earlier error")
            raise


@contextmanager
def configured_runtime_sources(
    configmap: ConfigMap, url: str, source_url: str, headers: dict[str, str]
) -> Generator[None]:
    """Install controlled sources and restore the original configuration even on setup failure.

    Cleanup errors raised after an earlier failure are logged so they do not hide the original error.

    Args:
        configmap: Operator-created user source ConfigMap.
        url: Serving runtime API base URL.
        source_url: Shared source API base URL.
        headers: Authentication headers.

    Yields:
        Control after all test sources are available.
    """
    original = yaml.safe_load(configmap.instance.data["sources.yaml"]) or {}
    original_sources = list_runtime_sources(source_url=source_url, headers=headers)
    assert not {source["id"] for source in original_sources}.intersection(TEST_CATALOGS), (
        "Test source IDs already exist"
    )
    baseline = list_catalog_items(url=f"{url}serving_runtimes", headers=headers)
    updated = deepcopy(x=original)
    updated["serving_runtime_catalogs"] = (updated.get("serving_runtime_catalogs") or []) + deepcopy(TEST_SOURCES)
    data = {"sources.yaml": yaml.safe_dump(updated)}
    data.update({f"{source_id}.yaml": yaml.safe_dump(document) for source_id, document in TEST_CATALOGS.items()})
    try:
        with refreshed_runtime_configmap(configmap=configmap, data=data):
            for source_id in TEST_CATALOGS:
                wait_for_source_status(
                    source_url=source_url, headers=headers, source_id=source_id, expected="available"
                )
            wait_for_runtime_data(url=url, headers=headers, catalogs=TEST_CATALOGS)
            yield
    except BaseException:
        try:
            wait_for_original_sources(
                url=url, source_url=source_url, headers=headers, baseline=baseline, original_sources=original_sources
            )
        except Exception:
            LOGGER.exception("Runtime source cleanup failed after an earlier error")
        raise
    else:
        wait_for_original_sources(
            url=url, source_url=source_url, headers=headers, baseline=baseline, original_sources=original_sources
        )


def catalog_pod_identity(pod: Pod) -> tuple[str, int]:
    """Identify the running catalog process, so tests can detect a pod or container restart.

    Args:
        pod: Model catalog pod.

    Returns:
        Pod UID and restart count of the catalog container.
    """
    instance = pod.instance
    restarts = next(
        status.restartCount for status in instance.status.containerStatuses if status.name == CATALOG_CONTAINER
    )
    return instance.metadata.uid, restarts


def wait_for_test_sources(url: str, source_url: str, headers: dict[str, str]) -> None:
    """Wait for the test sources to serve their original data and report `available`.

    Args:
        url: Serving runtime API base URL.
        source_url: Shared source administration API base URL.
        headers: Authentication headers.
    """
    wait_for_runtime_data(url=url, headers=headers, catalogs=TEST_CATALOGS)
    for source_id in TEST_CATALOGS:
        wait_for_source_status(source_url=source_url, headers=headers, source_id=source_id, expected="available")


@contextmanager
def runtime_catalog_patch(
    configmap: ConfigMap,
    url: str,
    source_url: str,
    headers: dict[str, str],
    catalogs: dict[str, dict[str, Any]] | None = None,
    sources: list[dict[str, Any]] | None = None,
) -> Generator[None]:
    """Temporarily change test sources, then wait for their original data and health.

    Cleanup errors raised after an earlier failure are logged so they do not hide the original error.

    Args:
        configmap: User source ConfigMap containing installed test sources.
        url: Serving runtime API base URL.
        source_url: Shared source administration API base URL.
        headers: Authentication headers.
        catalogs: Replacement YAML data, keyed by test source ID.
        sources: Replacement test source definitions, preserving non-test sources.

    Yields:
        Control immediately after patching, so tests can assert reconciliation.
    """
    data = {f"{source_id}.yaml": yaml.safe_dump(document) for source_id, document in (catalogs or {}).items()}
    if sources is not None:
        configuration = yaml.safe_load(configmap.instance.data["sources.yaml"])
        configuration["serving_runtime_catalogs"] = [
            source for source in configuration["serving_runtime_catalogs"] if source["id"] not in TEST_CATALOGS
        ] + sources
        data["sources.yaml"] = yaml.safe_dump(configuration)
    try:
        with refreshed_runtime_configmap(configmap=configmap, data=data):
            yield
    except BaseException:
        try:
            wait_for_test_sources(url=url, source_url=source_url, headers=headers)
        except Exception:
            LOGGER.exception("Runtime source restore failed after an earlier error")
        raise
    else:
        wait_for_test_sources(url=url, source_url=source_url, headers=headers)
