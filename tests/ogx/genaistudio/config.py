"""Gen AI Studio Playground OGX configuration.

Replicates the configuration the odh-dashboard BFF generates when a user creates
a Gen AI Studio Playground, so tests can start OGX exactly the way the Playground
does and catch upstream breakage of that configuration (e.g. the removal of
``remote::passthrough`` in OGX 1.3.0, which no downstream test
caught at the time).

Source of truth — keep in sync when re-generating (odh-dashboard tag ``v3.6.0``,
package ``packages/gen-ai/bff``):

- ``internal/integrations/kubernetes/token_k8s_client.go``
  (``InstallOGXServer``: creates the config ConfigMap —
  ``GENAI_PLAYGROUND_CONFIGMAP_NAME`` — and the
  ``lsd-genai-playground`` OGXServer CR with ``spec.overrideConfig``)
- ``internal/integrations/kubernetes`` config builders
  (default Playground config + ``NewPassthroughProvider``)
- ``internal/constants`` (ConfigMap name/key and other BFF constants)

The generated config was validated to start on the RHOAI 3.6 GA image
(``rhoai/odh-ogx-core-rhel9:rhoai-3.6``, ogx 1.4.2+rhaiv.0) with the
``genai-bff-proxy`` / ``remote::passthrough`` provider registered.
"""

from typing import Any

from tests.ogx.constants import (
    GENAI_PLAYGROUND_CONFIGMAP_KEY,
    GENAI_PLAYGROUND_CONFIGMAP_NAME,
    GENAI_PLAYGROUND_FORWARD_HEADERS,
    GENAI_PLAYGROUND_PASSTHROUGH_PROVIDER_ID,
    GENAI_PLAYGROUND_PASSTHROUGH_PROVIDER_TYPE,
    GENAI_PLAYGROUND_PASSTHROUGH_STUB_BASE_URL,
    GENAI_PLAYGROUND_SERVER_NAME,
)

# Legacy in-cluster paths still emitted by the dashboard config (v3.6.0 keeps
# its state under .llama dirs); values must match the dashboard output exactly.
_PLAYGROUND_STATE_PATH_PREFIX = "/opt/app-root/src/.llama/distributions/rh"
_PLAYGROUND_EXTERNAL_PROVIDERS_DIR = "/opt/app-root/.llama/providers.d"


def build_genai_playground_config(
    namespace: str = "playground",
    passthrough_base_url: str | None = None,
) -> dict[str, Any]:
    """Build the Gen AI Studio Playground config.yaml (full dashboard fidelity).

    Mirrors the dashboard's default config builder (v3.6.0) plus the install-time
    additions: the ``genai-bff-proxy`` ``remote::passthrough`` inference provider
    and the default embedding model registration. Inference (LLM) models are
    intentionally NOT registered — the dashboard resolves them per-request
    through the passthrough provider.

    Args:
        namespace: Namespace embedded in the passthrough ``base_url`` (the
            dashboard builds ``.../genai-proxy/ns/<namespace>``).
        passthrough_base_url: Optional full override of the passthrough provider
            ``base_url`` (e.g. an in-cluster mock proxy). Defaults to a
            non-routable stub — OGX connects to remote providers lazily, so a
            stub is enough to verify start and provider registration.

    Returns:
        The full OGX config.yaml as a dict, for the config ConfigMap
        (``GENAI_PLAYGROUND_CONFIGMAP_NAME``) referenced by
        ``spec.overrideConfig``.
    """
    base_url = passthrough_base_url or f"{GENAI_PLAYGROUND_PASSTHROUGH_STUB_BASE_URL}/{namespace}"

    return {
        "version": "2",
        "distro_name": "rh",
        "apis": [
            "responses",
            "file_processors",
            "files",
            "inference",
            "tool_runtime",
            "vector_io",
        ],
        "providers": {
            "inference": [
                {
                    "provider_id": "sentence-transformers",
                    "provider_type": "inline::sentence-transformers",
                    "config": {"trust_remote_code": False},
                },
                {
                    "provider_id": GENAI_PLAYGROUND_PASSTHROUGH_PROVIDER_ID,
                    "provider_type": GENAI_PLAYGROUND_PASSTHROUGH_PROVIDER_TYPE,
                    "config": {
                        # OGX appends /v1 automatically (-> /v1/models, /v1/chat/completions, ...).
                        "base_url": base_url,
                        # Credentials flow per-request via X-OGX-Provider-Data,
                        # mapped to outbound headers through forward_headers.
                        "api_key": "",
                        "forward_headers": dict(GENAI_PLAYGROUND_FORWARD_HEADERS),
                    },
                },
            ],
            "responses": [
                {
                    "provider_id": "builtin",
                    "provider_type": "inline::builtin",
                    "config": {
                        "persistence": {
                            "responses": {
                                "table_name": "responses",
                                "backend": "sql_default",
                                "max_write_queue_size": 10000,
                                "num_writers": 4,
                            }
                        }
                    },
                }
            ],
            "file_processors": [
                {
                    "provider_id": "auto",
                    "provider_type": "inline::auto",
                    "config": {},
                }
            ],
            "files": [
                {
                    "provider_id": "localfs-files",
                    "provider_type": "inline::localfs",
                    "config": {
                        "storage_dir": f"{_PLAYGROUND_STATE_PATH_PREFIX}/files",
                        "metadata_store": {
                            "table_name": "files_metadata",
                            "backend": "sql_default",
                        },
                    },
                }
            ],
            "tool_runtime": [
                {
                    "provider_id": "file-search",
                    "provider_type": "inline::file-search",
                    "config": {},
                },
                {
                    "provider_id": "model-context-protocol",
                    "provider_type": "remote::model-context-protocol",
                    "config": {},
                },
            ],
        },
        "registered_resources": {
            "models": [
                {
                    "provider_id": "sentence-transformers",
                    "model_id": "sentence-transformers/ibm-granite/granite-embedding-125m-english",
                    "provider_model_id": "ibm-granite/granite-embedding-125m-english",
                    "model_type": "embedding",
                    "metadata": {"embedding_dimension": 768},
                }
            ],
            "vector_stores": [],
        },
        "metadata_store": {
            "type": "sqlite",
            "db_path": f"{_PLAYGROUND_STATE_PATH_PREFIX}/inference_store.db",
        },
        "storage": {
            "backends": {
                "kv_default": {"type": "kv_sqlite", "db_path": f"{_PLAYGROUND_STATE_PATH_PREFIX}/kvstore.db"},
                "sql_default": {"type": "sql_sqlite", "db_path": f"{_PLAYGROUND_STATE_PATH_PREFIX}/sql_store.db"},
            },
            "stores": {
                "metadata": {"namespace": "registry", "backend": "kv_default"},
                "inference": {
                    "table_name": "inference_store",
                    "backend": "sql_default",
                    "max_write_queue_size": 10000,
                    "num_writers": 4,
                },
                "conversations": {"table_name": "openai_conversations", "backend": "sql_default"},
                "prompts": {"table_name": "prompts", "backend": "sql_default"},
                "connectors": {"table_name": "connectors", "backend": "sql_default"},
            },
        },
        "vector_stores": {
            # VectorIO starts empty; the dashboard adds remote::pgvector only when
            # PostgreSQL is provisioned. default_provider_id stays empty otherwise.
            "default_provider_id": "",
            "default_embedding_model": {
                "provider_id": "sentence-transformers",
                "model_id": "ibm-granite/granite-embedding-125m-english",
            },
            "annotation_prompt_params": {"enable_annotations": True},
            "file_ingestion_params": {
                "default_chunk_size_tokens": 512,
                "default_chunk_overlap_tokens": 128,
            },
            "chunk_retrieval_params": {
                "chunk_multiplier": 5,
                "max_tokens_in_context": 4000,
                "default_reranker_strategy": "rrf",
                "rrf_impact_factor": 60.0,
                "weighted_search_alpha": 0.5,
                "default_search_mode": "vector",
            },
            "file_batch_params": {
                "max_concurrent_files_per_batch": 3,
                "file_batch_chunk_size": 10,
                "cleanup_interval_seconds": 86400,
            },
        },
        "external_providers_dir": _PLAYGROUND_EXTERNAL_PROVIDERS_DIR,
        "server": {"port": 8321},
    }


def build_genai_playground_server_config() -> dict[str, Any]:
    """Build the OGXServer spec for the Gen AI Studio Playground deployment.

    Mirrors what the dashboard's ``InstallOGXServer`` creates for the
    ``lsd-genai-playground`` OGXServer CR — ``spec.distribution.name: rh`` plus
    ``spec.overrideConfig`` pointing at the config ConfigMap —
    with the workload resources and route-ingress network policy the shared test
    fixtures need (the dashboard relies on default namespace network policy).

    Returns:
        OGXServerSpec configuration dict for ``create_ogx_server`` (keys:
        ``distribution``, ``overrideConfig``, ``workload``, ``network``).
    """
    return {
        "distribution": {"name": "rh"},
        # Full config.yaml override; the ConfigMap (with ogx.io/watch: "true"
        # label) must exist in the same namespace before the OGXServer is created.
        "overrideConfig": {
            "name": GENAI_PLAYGROUND_CONFIGMAP_NAME,
            "key": GENAI_PLAYGROUND_CONFIGMAP_KEY,
        },
        "workload": {
            "resources": {
                "requests": {"cpu": "1", "memory": "1Gi"},
                "limits": {"cpu": "2", "memory": "2Gi"},
            },
        },
        # Allow ingress to the OGX port from the router so the shared
        # ogx_test_route/ogx_client fixtures can reach the server.
        "network": {
            "policy": {
                "ingress": [
                    {
                        "ports": [{"port": 8321, "protocol": "TCP"}],
                        "from": [
                            {"namespaceSelector": {"matchLabels": {"network.openshift.io/policy-group": "ingress"}}},
                            {
                                "namespaceSelector": {
                                    "matchLabels": {"network.openshift.io/policy-group": "host-network"}
                                }
                            },
                        ],
                    }
                ]
            }
        },
    }


__all__ = [
    "GENAI_PLAYGROUND_SERVER_NAME",
    "build_genai_playground_config",
    "build_genai_playground_server_config",
]
