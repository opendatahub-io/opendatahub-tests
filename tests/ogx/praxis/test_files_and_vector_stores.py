"""Files and Vector Stores API parity across an OGX -> Praxis migration.

The module seeds files and a vector store through OGX, captures their API
responses, and then enables Praxis mode to run the database migration. The tests
re-issue the same requests through Praxis and diff each response against its
in-memory baseline, so a dropped or rewritten id is reported as a field
difference rather than as a generic request failure.
"""

from collections.abc import Callable

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ogx_client import OgxClient
from ogx_client.types.vector_store import VectorStore

from tests.ogx.praxis.constants import (
    FILES_API_PATH,
    NAMESPACE_PARAMS,
    OGX_SERVER_PARAMS,
    SEED_FILES_COUNT,
    VECTOR_STORES_API_PATH,
)
from tests.ogx.praxis.utils import (
    ApiBaseline,
    capture_api_baseline,
    format_field_diff,
    http_routes_matching_path,
    pod_logs,
    retrieve_file_fields,
    retrieve_vector_store_fields,
    seed_files,
    serving_pods_for_path,
)
from utilities.resources.ogx_server import OgxServer

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="class")
def files_and_vector_stores_baseline(
    ogx_client: OgxClient,
    vector_store: VectorStore,
    praxis_migration_runner: Callable[[], OgxServer],
) -> ApiBaseline:
    """Seed OGX data, capture its baseline, and run the Praxis migration once."""
    seeded_file_ids = seed_files(ogx_client=ogx_client, count=SEED_FILES_COUNT)
    baseline = capture_api_baseline(
        ogx_client=ogx_client,
        file_ids=seeded_file_ids,
        vector_store_ids=[vector_store.id],
    )
    LOGGER.info(
        f"Captured pre-migration baseline for {len(baseline['files'])} file(s) "
        f"and {len(baseline['vector_stores'])} vector store(s)"
    )

    praxis_migration_runner()
    LOGGER.info("Completed OGX to Praxis migration")
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
class TestFilesAndVectorStoresMigration:
    """Verify OGX Files and Vector Stores survive migration to Praxis."""

    def test_file_responses_unchanged(
        self,
        files_and_vector_stores_baseline: ApiBaseline,
        praxis_client: OgxClient,
    ) -> None:
        """Verify every pre-migration file id returns an identical body after migration.

        Given: Files created in OGX before migration to Praxis.
        When: GET /v1/files/{id} is re-issued for each recorded id, unchanged.
        Then: Each returns HTTP 200 with id, bytes, filename, created_at and status unchanged.
        """
        baseline_files = files_and_vector_stores_baseline["files"]
        assert baseline_files, "Pre-migration baseline recorded no files"

        differences = []
        for file_id, before in baseline_files.items():
            after = retrieve_file_fields(ogx_client=praxis_client, file_id=file_id)
            if diff := format_field_diff(resource="File", resource_id=file_id, before=before, after=after):
                differences.append(diff)

        assert not differences, "Files API did not return identical bodies after migration: " + "; ".join(differences)
        LOGGER.info(f"{len(baseline_files)} file(s) returned identical bodies after migration")

    def test_vector_store_responses_unchanged(
        self,
        files_and_vector_stores_baseline: ApiBaseline,
        praxis_client: OgxClient,
    ) -> None:
        """Verify every pre-migration vector store id returns an identical body after migration.

        Given: A vector store created in OGX before migration to Praxis.
        When: GET /v1/vector_stores/{id} is re-issued for each recorded id, unchanged.
        Then: Each returns HTTP 200 with id, name, created_at and status unchanged.
        """
        baseline_vector_stores = files_and_vector_stores_baseline["vector_stores"]
        assert baseline_vector_stores, "Pre-migration baseline recorded no vector stores"

        differences = []
        for vector_store_id, before in baseline_vector_stores.items():
            after = retrieve_vector_store_fields(ogx_client=praxis_client, vector_store_id=vector_store_id)
            if diff := format_field_diff(
                resource="Vector store", resource_id=vector_store_id, before=before, after=after
            ):
                differences.append(diff)

        assert not differences, "Vector Stores API did not return identical bodies after migration: " + "; ".join(
            differences
        )
        LOGGER.info(f"{len(baseline_vector_stores)} vector store(s) returned identical bodies after migration")

    def test_ids_listed_by_the_api(
        self,
        files_and_vector_stores_baseline: ApiBaseline,
        praxis_client: OgxClient,
    ) -> None:
        """Verify no pre-migration id was dropped from the collection endpoints.

        Given: Files and a vector store migrated from OGX to Praxis.
        When: The Files and Vector Stores collections are listed.
        Then: Every pre-migration id is still present, so none was silently discarded.
        """
        listed_file_ids = {file.id for file in praxis_client.files.list().data}
        missing_files = sorted(set(files_and_vector_stores_baseline["files"]) - listed_file_ids)
        assert not missing_files, f"File ids missing from GET {FILES_API_PATH} after migration: {missing_files}"

        listed_vector_store_ids = {store.id for store in praxis_client.vector_stores.list().data}
        missing_vector_stores = sorted(
            set(files_and_vector_stores_baseline["vector_stores"]) - listed_vector_store_ids
        )
        assert not missing_vector_stores, (
            f"Vector store ids missing from GET {VECTOR_STORES_API_PATH} after migration: {missing_vector_stores}"
        )

    def test_requests_served_by_praxis(
        self,
        admin_client: DynamicClient,
        files_and_vector_stores_baseline: ApiBaseline,
        praxis_client: OgxClient,
    ) -> None:
        """Verify post-migration Files requests are handled by the Praxis workload.

        Given: A migrated OGXServer whose Gateway routes were reconciled by rhods-operator.
        When: A recorded file id is read back and the serving pods for /v1/files are inspected.
        Then: The path is owned by a single route whose backing pods log the request.
        """
        file_id = next(iter(files_and_vector_stores_baseline["files"]))

        routes = http_routes_matching_path(client=admin_client, path=FILES_API_PATH)
        assert len(routes) == 1, (
            f"Expected exactly one HTTPRoute declaring {FILES_API_PATH} after migration, "
            f"found {[(route.namespace, route.name) for route in routes]}"
        )

        serving_pods = serving_pods_for_path(client=admin_client, http_route=routes[0], path=FILES_API_PATH)
        assert serving_pods, f"HTTPRoute {routes[0].namespace}/{routes[0].name} resolves to no running pods"

        praxis_client.files.retrieve(file_id=file_id)

        logs = "\n".join(pod_logs(pod=pod) for pod in serving_pods)
        assert file_id in logs, (
            f"File id {file_id} was not found in the logs of the pods serving {FILES_API_PATH} "
            f"({[pod.name for pod in serving_pods]}), so the request was not served by that workload"
        )
