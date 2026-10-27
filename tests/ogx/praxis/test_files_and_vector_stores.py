"""Files and Vector Stores migration coverage for OGX to Praxis."""

from collections.abc import Callable
from typing import Any

import pytest
from ogx_client import OgxClient
from ogx_client.types.vector_store import VectorStore

from tests.ogx.datasets import IBM_2025_Q4_EARNINGS
from utilities.resources.ogx_server import OgxServer

NAMESPACE_PARAMS: dict[str, Any] = {"name": "test-ogx-praxis-files-migration", "randomize_name": True}
OGX_SERVER_PARAMS: dict[str, Any] = {
    "ogx_storage_size": "2Gi",
    "vector_io_provider": "pgvector",
    "files_provider": "local",
    "cpu_requests": "500m",
    "cpu_limits": "2",
}


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server, vector_store",
    [
        pytest.param(
            NAMESPACE_PARAMS,
            OGX_SERVER_PARAMS,
            {"vector_io_provider": "pgvector", "dataset": IBM_2025_Q4_EARNINGS},
        ),
    ],
    indirect=True,
)
@pytest.mark.ogx
@pytest.mark.tier1
class TestFilesAndVectorStoresMigration:
    """Verify Files and Vector Stores are accessible after migration."""

    def test_files_and_vector_stores_accessible_after_migration(
        self,
        ogx_client: OgxClient,
        ogx_praxis_client: OgxClient,
        vector_store: VectorStore,
        praxis_migration_runner: Callable[[], OgxServer],
    ) -> None:
        """Verify seeded Files and Vector Stores are accessible after migration.

        Given: An isolated OGXServer with a vector store seeded from the IBM earnings dataset.
        When: Praxis migration mode is enabled and its migration Job completes.
        Then: Praxis lists the same File and Vector Store IDs as OGX did before migration.
        """
        source_file_ids = {file.id for file in ogx_client.files.list().data}
        source_vector_store_ids = {store.id for store in ogx_client.vector_stores.list().data}

        praxis_migration_runner()

        target_file_ids = {file.id for file in ogx_praxis_client.files.list().data}
        target_vector_store_ids = {store.id for store in ogx_praxis_client.vector_stores.list().data}

        assert target_file_ids == source_file_ids, "Praxis File IDs differ from the File IDs seeded in OGX"
        assert target_vector_store_ids == source_vector_store_ids, (
            "Praxis Vector Store IDs differ from the Vector Store IDs seeded in OGX"
        )
