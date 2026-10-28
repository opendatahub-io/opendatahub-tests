"""Files and Vector Stores migration coverage for OGX to Praxis."""

from collections.abc import Callable
from typing import Any

import pytest
from ocp_resources.deployment import Deployment
from ogx_client import OgxClient
from ogx_client.types.vector_store import VectorStore

from tests.ogx.datasets import IBM_2025_Q4_EARNINGS
from tests.ogx.praxis.constants import (
    MIGRATION_FALLBACK_OWNER_SUBJECT,
    MIGRATION_FALLBACK_TENANT,
    PRAXIS_GATEWAY_POD_LABELS,
)
from tests.ogx.praxis.migration import wait_for_deployment_rollout
from utilities.resources.ogx_server import OgxServer

NAMESPACE_PARAMS: dict[str, Any] = {"name": "test-ogx-praxis-files-migration", "randomize_name": True}
OGX_SERVER_PARAMS: dict[str, Any] = {
    "ogx_storage_size": "2Gi",
    "vector_io_provider": "pgvector",
    "files_provider": "local",
    "cpu_requests": "500m",
    "cpu_limits": "2",
    "network": {
        "policy": {
            "ingress": [
                {
                    "ports": [{"port": 8321, "protocol": "TCP"}],
                    "from": [
                        {"namespaceSelector": {"matchLabels": {"network.openshift.io/policy-group": "ingress"}}},
                        {"namespaceSelector": {"matchLabels": {"network.openshift.io/policy-group": "host-network"}}},
                        {"podSelector": {"matchLabels": PRAXIS_GATEWAY_POD_LABELS}},
                    ],
                }
            ]
        }
    },
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
@pytest.skip(
    reason=(
        "Skipping. PraxisMode sets server.tenancy.mode to multi-tenant,",
        "which makes previously seeded Files and Vector Stores inaccessible.",
        "Un-skip this test after the issue with multi-tenancy is resolved.",
    )
)
class TestFilesAndVectorStoresMigration:
    """Verify Files and Vector Stores are accessible after migration."""

    def test_files_and_vector_stores_accessible_after_migration(
        self,
        ogx_client: OgxClient,
        ogx_praxis_client: OgxClient,
        ogx_server: OgxServer,
        ogx_server_deployment: Deployment,
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

        prev_gen = int(ogx_server.instance.metadata.generation)
        praxis_migration_runner()
        wait_for_deployment_rollout(deployment=ogx_server_deployment, previous_generation=prev_gen)

        extra_headers = {
            "x-user-id": MIGRATION_FALLBACK_OWNER_SUBJECT,
            "x-tenant-id": MIGRATION_FALLBACK_TENANT,
        }

        target_files = ogx_praxis_client.files.list(extra_headers=extra_headers).data
        target_vector_stores = ogx_praxis_client.vector_stores.list(extra_headers=extra_headers).data
        target_file_ids = {file.id for file in target_files}
        target_vector_store_ids = {store.id for store in target_vector_stores}

        assert target_file_ids == source_file_ids, "Praxis File IDs differ from the File IDs seeded in OGX"
        assert target_vector_store_ids == source_vector_store_ids, (
            "Praxis Vector Store IDs differ from the Vector Store IDs seeded in OGX"
        )
