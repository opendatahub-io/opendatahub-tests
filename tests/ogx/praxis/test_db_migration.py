"""Isolated OGX -> Praxis database migration coverage."""

from collections.abc import Callable
from typing import Any

import pytest
from ocp_resources.pod import Pod
from ogx_client import OgxClient

from tests.ogx.constants import ModelInfo
from tests.ogx.praxis.constants import (
    OGX_POSTGRES_DATABASE,
    PRAXIS_POSTGRES_DATABASE,
    SEED_CONVERSATIONS_COUNT,
    SEED_RESPONSES_COUNT,
    SOURCE_CONVERSATIONS_TABLE,
    SOURCE_RESPONSES_TABLE,
    TARGET_CONVERSATIONS_TABLE,
    TARGET_RESPONSES_TABLE,
)
from tests.ogx.praxis.utils import (
    compare_rows,
    format_parity_failure,
    query_database,
    seed_conversations,
    seed_responses,
)
from utilities.resources.ogx_server import OgxServer

# `metadata` is a JSON column in OGX and a JSON-serialized text column in Praxis;
# routing both through `jsonb` canonicalizes key order and whitespace so the two
# representations are comparable.
_JSON_METADATA_COLUMN = "COALESCE(metadata::text::jsonb, '{}'::jsonb)::text AS metadata"


def _responses_query(table_name: str) -> str:
    """Return the responses SELECT for `table_name`."""
    return f"SELECT id, created_at, model FROM {table_name}"


def _conversations_query(table_name: str, id_column: str) -> str:
    """Return the conversations SELECT for `table_name`, whose id column differs between OGX and Praxis."""
    return f"SELECT {id_column}, created_at, {_JSON_METADATA_COLUMN} FROM {table_name}"


SOURCE_RESPONSES_QUERY = _responses_query(table_name=SOURCE_RESPONSES_TABLE)
SOURCE_CONVERSATIONS_QUERY = _conversations_query(table_name=SOURCE_CONVERSATIONS_TABLE, id_column="id")
TARGET_RESPONSES_QUERY = _responses_query(table_name=TARGET_RESPONSES_TABLE)
TARGET_CONVERSATIONS_QUERY = _conversations_query(table_name=TARGET_CONVERSATIONS_TABLE, id_column="conversation_id")

# This migration owns a namespace and OGXServer so it cannot share state with
# other OGX -> Praxis migration modules.
NAMESPACE_PARAMS: dict[str, Any] = {"name": "test-ogx-praxis-db-migration"}
OGX_SERVER_PARAMS: dict[str, Any] = {
    "ogx_storage_size": "2Gi",
    "vector_io_provider": "pgvector",
    "files_provider": "local",
    "cpu_requests": "500m",
    "cpu_limits": "2",
}


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server",
    [pytest.param(NAMESPACE_PARAMS, OGX_SERVER_PARAMS)],
    indirect=True,
)
@pytest.mark.ogx
class TestPraxisDbMigration:
    def test_responses_and_conversations_migrated_to_praxis(
        self,
        ogx_client: OgxClient,
        ogx_models: ModelInfo,
        ogx_postgres_pod: Pod,
        praxis_migration_runner: Callable[[], OgxServer],
        praxis_postgres_pod: Pod,
    ) -> None:
        """Verify seeded OGX responses and conversations are migrated to Praxis.

        Given: An isolated OGXServer with seeded responses and conversations.
        When: Praxis migration mode is enabled and its migration Job completes.
        Then: Every source row is present in Praxis with matching field values.
        """
        # Seed and read the source before running the migration job.
        conversation_ids = seed_conversations(ogx_client=ogx_client, count=SEED_CONVERSATIONS_COUNT)
        response_ids = seed_responses(
            ogx_client=ogx_client,
            model_id=ogx_models.model_id,
            count=SEED_RESPONSES_COUNT,
            conversation_id=conversation_ids[0],
        )

        source_responses = query_database(
            postgres_pod=ogx_postgres_pod,
            database=OGX_POSTGRES_DATABASE,
            select_statement=SOURCE_RESPONSES_QUERY,
        )
        source_conversations = query_database(
            postgres_pod=ogx_postgres_pod,
            database=OGX_POSTGRES_DATABASE,
            select_statement=SOURCE_CONVERSATIONS_QUERY,
        )

        # Run the migration job.
        praxis_migration_runner()

        target_responses = query_database(
            postgres_pod=praxis_postgres_pod,
            database=PRAXIS_POSTGRES_DATABASE,
            select_statement=TARGET_RESPONSES_QUERY,
        )
        target_conversations = query_database(
            postgres_pod=praxis_postgres_pod,
            database=PRAXIS_POSTGRES_DATABASE,
            select_statement=TARGET_CONVERSATIONS_QUERY,
        )

        # Verify the migration job's parity with the source.
        responses_parity = compare_rows(
            table=SOURCE_RESPONSES_TABLE,
            source_rows=source_responses,
            target_rows=target_responses,
            source_key="id",
            target_key="id",
            compared_fields=("created_at", "model"),
        )

        conversations_parity = compare_rows(
            table=SOURCE_CONVERSATIONS_TABLE,
            source_rows=source_conversations,
            target_rows=target_conversations,
            source_key="id",
            target_key="conversation_id",
            compared_fields=("created_at", "metadata"),
        )

        assert responses_parity.source_count, (
            f"No rows found in the source {SOURCE_RESPONSES_TABLE} table; migration parity cannot be asserted"
        )
        assert conversations_parity.source_count, (
            f"No rows found in the source {SOURCE_CONVERSATIONS_TABLE} table; migration parity cannot be asserted"
        )
        failures = [
            format_parity_failure(parity=responses_parity, allow_extra_in_target=False),
            format_parity_failure(parity=conversations_parity, allow_extra_in_target=True),
        ]
        assert not any(failures), "\n".join(failure for failure in failures if failure)
