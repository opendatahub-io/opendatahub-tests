from typing import Any

import yaml

from tests.ogx.praxis.constants import (
    MIGRATION_FALLBACK_TENANT,
    PRAXIS_GATEWAY_APP_LABEL,
    PRAXIS_GATEWAY_PORT,
    TARGET_CONVERSATION_ITEMS_TABLE,
    TARGET_CONVERSATIONS_TABLE,
    TARGET_RESPONSES_TABLE,
)
from tests.ogx.praxis.utils import praxis_connection_string

# Names internal to the Praxis configuration rendered by `praxis_gateway_config`.
# The upstream is deliberately unreachable: nothing the tests send to Praxis is
# meant to be proxied anywhere.
_PRAXIS_FILTER_CHAIN_NAME: str = "store-pipeline"
_PRAXIS_UNUSED_UPSTREAM_NAME: str = "unused-upstream"
_PRAXIS_UNUSED_UPSTREAM: str = "127.0.0.1:1"


def build_praxis_gateway_config(namespace: str) -> str:
    """Render the Praxis configuration that puts it in front of the target database.

    The pipeline is the one the OGX repository's `migration-e2e-praxis` workflow
    uses, reduced to what stamping the schema needs: the two store filters own
    the three tables the migration writes to, and the filters around them are the
    ones those two require to be reachable. The upstream cluster is never dialed,
    because a request for a missing resource is answered locally.

    The rendered document embeds the target DSN, so it must only ever be written
    into a Secret.

    Args:
        namespace: Namespace holding the target PostgreSQL service.

    Returns:
        The Praxis configuration as a YAML document.
    """
    postgres_backend: dict[str, Any] = {
        "backend": "postgres",
        "database_url": praxis_connection_string(namespace=namespace),
        # The DSN addresses a Service by name, which Praxis rejects as a private
        # target unless it is opted into, and the test PostgreSQL serves no TLS.
        "allow_private_database_url": True,
        "ssl_mode": "disable",
    }
    config: dict[str, Any] = {
        "insecure_options": {"allow_private_endpoints": True},
        "listeners": [
            {
                "name": PRAXIS_GATEWAY_APP_LABEL,
                "address": f"0.0.0.0:{PRAXIS_GATEWAY_PORT}",
                "filter_chains": [_PRAXIS_FILTER_CHAIN_NAME],
            }
        ],
        "filter_chains": [
            {
                "name": _PRAXIS_FILTER_CHAIN_NAME,
                "filters": [
                    {"filter": "state_owner", "mode": "single_tenant", "tenant_id": MIGRATION_FALLBACK_TENANT},
                    # Classifies the operation from the request head. `openai_conversations`
                    # requires it earlier in the same chain.
                    {"filter": "ai_operation"},
                    {
                        "filter": "openai_conversations",
                        "conversations_table": TARGET_CONVERSATIONS_TABLE,
                        "items_table": TARGET_CONVERSATION_ITEMS_TABLE,
                        **postgres_backend,
                    },
                    {"filter": "openai_responses_request"},
                    {
                        "filter": "openai_responses_store",
                        "responses_table": TARGET_RESPONSES_TABLE,
                        "conversations_table": TARGET_CONVERSATIONS_TABLE,
                        **postgres_backend,
                    },
                    # The probes are served locally and the migration never POSTs
                    # to Praxis, so no request reaches this cluster. It exists
                    # only because the response-store chain requires a route.
                    {
                        "filter": "router",
                        "routes": [{"path_prefix": "/", "cluster": _PRAXIS_UNUSED_UPSTREAM_NAME}],
                    },
                    {
                        "filter": "load_balancer",
                        "clusters": [{"name": _PRAXIS_UNUSED_UPSTREAM_NAME, "endpoints": [_PRAXIS_UNUSED_UPSTREAM]}],
                    },
                ],
            }
        ],
    }
    return yaml.safe_dump(data=config, sort_keys=False)
