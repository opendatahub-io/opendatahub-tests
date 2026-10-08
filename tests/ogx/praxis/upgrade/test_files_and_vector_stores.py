"""Files and Vector Stores API parity across the 3.5 -> 3.6 upgrade to Praxis.

The pre-upgrade test seeds files through the Files API and a vector store through
the Vector Stores API, then records `GET /v1/files/{id}` and
`GET /v1/vector_stores/{id}` for each id into a ConfigMap. The upgrade itself --
which lets rhods-operator reconcile the Gateway and HTTPRoute resources so Praxis
takes over the public API -- is performed by the Jenkins job that runs these
tests.

The post-upgrade test re-issues the identical requests against the same external
hostname and diffs each response against its baseline, so a dropped or rewritten
id is caught as a field difference rather than as a generic request failure.
Nothing in the client changes between the two runs: `ogx_test_route` pins the
route name across upgrade phases, so both runs resolve the same URL.
"""

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ogx_client import OgxClient
from ogx_client.types.vector_store import VectorStore

from tests.ogx.praxis.constants import (
    FILES_API_PATH,
    NAMESPACE_PARAMS,
    OGX_SERVER_PARAMS,
    VECTOR_STORES_API_PATH,
)
from tests.ogx.praxis.upgrade.constants import (
    FILES_AND_VECTOR_STORES_CONFIG_MAP_KEY,
    SEED_FILES_COUNT,
)
from tests.ogx.praxis.upgrade.utils import (
    ApiBaseline,
    capture_api_baseline,
    format_field_diff,
    load_baseline_section,
    retrieve_file_fields,
    retrieve_vector_store_fields,
    save_baseline_section,
    seed_files,
)
from tests.ogx.praxis.utils import (
    http_routes_matching_path,
    pod_logs,
    serving_pods_for_path,
)

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="class")
def files_and_vector_stores_baseline(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> ApiBaseline:
    """The Files and Vector Stores baseline section written by the pre-upgrade run."""
    baseline: ApiBaseline = load_baseline_section(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        section=FILES_AND_VECTOR_STORES_CONFIG_MAP_KEY,
    )
    return baseline


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server, vector_store",
    [
        pytest.param(
            NAMESPACE_PARAMS,
            OGX_SERVER_PARAMS,
            {"vector_io_provider": "pgvector"},
        ),
    ],
    indirect=True,
)
@pytest.mark.ogx
@pytest.mark.tier1
class TestPreUpgradeFilesAndVectorStores:
    """Record the 3.5 Files and Vector Stores responses that must survive the upgrade."""

    @pytest.mark.pre_upgrade
    def test_api_baseline_captured(
        self,
        unprivileged_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        ogx_client: OgxClient,
        vector_store: VectorStore,
    ) -> None:
        """Verify the pre-upgrade Files and Vector Stores responses are readable and recorded.

        Given: A 3.5 cluster with OGX serving the public API.
        When: Files are uploaded, a vector store is created, and every id is read back through
            GET /v1/files/{id} and GET /v1/vector_stores/{id}.
        Then: Every id returns a populated response, and the bodies are persisted as the baseline.
        """
        # The vector store comes from the shared `vector_store` fixture because it owns the
        # post-upgrade reuse and the `teardown_resources` gating; the files are seeded here so
        # the data the parity assertions depend on is visible in the test itself.
        seeded_file_ids = seed_files(ogx_client=ogx_client, count=SEED_FILES_COUNT)

        baseline = capture_api_baseline(
            ogx_client=ogx_client,
            file_ids=seeded_file_ids,
            vector_store_ids=[vector_store.id],
        )

        save_baseline_section(
            client=unprivileged_client,
            namespace=unprivileged_model_namespace.name,
            section=FILES_AND_VECTOR_STORES_CONFIG_MAP_KEY,
            payload=baseline,
        )
        LOGGER.info(
            f"Captured pre-upgrade baseline for {len(baseline['files'])} file(s) "
            f"and {len(baseline['vector_stores'])} vector store(s)"
        )


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server",
    [
        pytest.param(NAMESPACE_PARAMS, OGX_SERVER_PARAMS),
    ],
    indirect=True,
)
@pytest.mark.ogx
@pytest.mark.tier1
class TestPostUpgradeFilesAndVectorStores:
    """Verify the 3.5 ids still resolve identically once Praxis serves the API."""

    @pytest.mark.post_upgrade
    def test_file_responses_unchanged(
        self,
        ogx_client: OgxClient,
        files_and_vector_stores_baseline: ApiBaseline,
    ) -> None:
        """Verify every pre-upgrade file id returns an identical body after the upgrade.

        Given: A cluster upgraded from 3.5 to 3.6, with files created before the upgrade.
        When: GET /v1/files/{id} is re-issued for each recorded id, unchanged.
        Then: Each returns HTTP 200 with id, bytes, filename, created_at and status unchanged.
        """
        baseline_files = files_and_vector_stores_baseline["files"]
        assert baseline_files, "Pre-upgrade baseline recorded no files"

        differences = []
        for file_id, before in baseline_files.items():
            after = retrieve_file_fields(ogx_client=ogx_client, file_id=file_id)
            if diff := format_field_diff(resource="File", resource_id=file_id, before=before, after=after):
                differences.append(diff)

        assert not differences, "Files API did not return identical bodies after the upgrade: " + "; ".join(differences)
        LOGGER.info(f"{len(baseline_files)} file(s) returned identical bodies after the upgrade")

    @pytest.mark.post_upgrade
    def test_vector_store_responses_unchanged(
        self,
        ogx_client: OgxClient,
        files_and_vector_stores_baseline: ApiBaseline,
    ) -> None:
        """Verify every pre-upgrade vector store id returns an identical body after the upgrade.

        Given: A cluster upgraded from 3.5 to 3.6, with a vector store created before the upgrade.
        When: GET /v1/vector_stores/{id} is re-issued for each recorded id, unchanged.
        Then: Each returns HTTP 200 with id, name, created_at and status unchanged.
        """
        baseline_vector_stores = files_and_vector_stores_baseline["vector_stores"]
        assert baseline_vector_stores, "Pre-upgrade baseline recorded no vector stores"

        differences = []
        for vector_store_id, before in baseline_vector_stores.items():
            after = retrieve_vector_store_fields(ogx_client=ogx_client, vector_store_id=vector_store_id)
            if diff := format_field_diff(
                resource="Vector store", resource_id=vector_store_id, before=before, after=after
            ):
                differences.append(diff)

        assert not differences, "Vector Stores API did not return identical bodies after the upgrade: " + "; ".join(
            differences
        )
        LOGGER.info(f"{len(baseline_vector_stores)} vector store(s) returned identical bodies after the upgrade")

    @pytest.mark.post_upgrade
    def test_ids_listed_by_the_api(
        self,
        ogx_client: OgxClient,
        files_and_vector_stores_baseline: ApiBaseline,
    ) -> None:
        """Verify no pre-upgrade id was dropped from the collection endpoints.

        Given: A cluster upgraded from 3.5 to 3.6.
        When: The Files and Vector Stores collections are listed.
        Then: Every pre-upgrade id is still present, so none was silently discarded.
        """
        listed_file_ids = {file.id for file in ogx_client.files.list().data}
        missing_files = sorted(set(files_and_vector_stores_baseline["files"]) - listed_file_ids)
        assert not missing_files, f"File ids missing from GET {FILES_API_PATH} after the upgrade: {missing_files}"

        listed_vector_store_ids = {store.id for store in ogx_client.vector_stores.list().data}
        missing_vector_stores = sorted(set(files_and_vector_stores_baseline["vector_stores"]) - listed_vector_store_ids)
        assert not missing_vector_stores, (
            f"Vector store ids missing from GET {VECTOR_STORES_API_PATH} after the upgrade: {missing_vector_stores}"
        )

    @pytest.mark.post_upgrade
    def test_requests_served_by_praxis(
        self,
        admin_client: DynamicClient,
        ogx_client: OgxClient,
        files_and_vector_stores_baseline: ApiBaseline,
    ) -> None:
        """Verify the post-upgrade Files requests are handled by the Praxis workload.

        Given: An upgraded cluster whose Gateway routes were reconciled by rhods-operator.
        When: A recorded file id is read back and the serving pods for /v1/files are inspected.
        Then: The path is owned by a single route whose backing pods log the request.
        """
        file_id = next(iter(files_and_vector_stores_baseline["files"]))

        routes = http_routes_matching_path(client=admin_client, path=FILES_API_PATH)
        assert len(routes) == 1, (
            f"Expected exactly one HTTPRoute declaring {FILES_API_PATH} after the upgrade, "
            f"found {[(route.namespace, route.name) for route in routes]}"
        )

        serving_pods = serving_pods_for_path(client=admin_client, http_route=routes[0])
        assert serving_pods, f"HTTPRoute {routes[0].namespace}/{routes[0].name} resolves to no running pods"

        ogx_client.files.retrieve(file_id=file_id)

        logs = "\n".join(pod_logs(pod=pod) for pod in serving_pods)
        assert file_id in logs, (
            f"File id {file_id} was not found in the logs of the pods serving {FILES_API_PATH} "
            f"({[pod.name for pod in serving_pods]}), so the request was not served by that workload"
        )
