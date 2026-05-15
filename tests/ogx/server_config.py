from collections.abc import Callable
from typing import Any

import structlog

from tests.ogx.constants import (
    HTTPS_PROXY,
    LLS_CORE_EMBEDDING_MODEL,
    LLS_CORE_EMBEDDING_PROVIDER_MODEL_ID,
    LLS_CORE_INFERENCE_MODEL,
    LLS_CORE_VLLM_EMBEDDING_MAX_TOKENS,
    LLS_CORE_VLLM_EMBEDDING_TLS_VERIFY,
    LLS_CORE_VLLM_EMBEDDING_URL,
    LLS_CORE_VLLM_MAX_TOKENS,
    LLS_CORE_VLLM_TLS_VERIFY,
    LLS_CORE_VLLM_URL,
)

LOGGER = structlog.get_logger(name=__name__)


def build_ogx_server_config(
    vector_io_provider_deployment_config_factory: Callable[[str], list[dict[str, str]]],
    files_provider_config_factory: Callable[[str], list[dict[str, str]]],
    is_disconnected_cluster: bool,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Build OGXServer spec configuration matching the v1beta1 CRD.

    Returns a dict whose keys correspond to top-level OGXServerSpec fields
    (distribution, workload, tls). The caller is responsible for adding
    network policy configuration.

    Args:
        vector_io_provider_deployment_config_factory: Factory to deploy vector I/O providers
            and return their configuration environment variables.
        files_provider_config_factory: Factory to configure files storage providers
            and return their configuration environment variables.
        is_disconnected_cluster: Whether the target cluster is disconnected (air-gapped).
        params: Configuration parameters dict with optional keys:
            - inference_model: Override for INFERENCE_MODEL env var.
            - embedding_provider: Embedding provider ("vllm-embedding" or "sentence-transformers").
            - vector_io_provider: Vector I/O provider type (e.g. "milvus", "milvus-remote").
            - files_provider: Files storage provider ("local" or "s3").
            - ogx_storage_size: PVC storage size (e.g. "2Gi").

    Returns:
        OGXServerSpec configuration dict with distribution, workload, and optional tls sections.
    """

    env_vars = []
    tls_config: dict[str, Any] | None = None
    cpu_requests = "2"
    cpu_limits = "4"

    # INFERENCE_MODEL
    if params.get("inference_model"):
        inference_model = str(params.get("inference_model"))
    else:
        inference_model = LLS_CORE_INFERENCE_MODEL
    env_vars.append({"name": "INFERENCE_MODEL", "value": inference_model})

    env_vars.append(
        {
            "name": "VLLM_API_TOKEN",
            "valueFrom": {"secretKeyRef": {"name": "llamastack-distribution-secret", "key": "vllm-api-token"}},
        },
    )

    env_vars.append({"name": "VLLM_URL", "value": LLS_CORE_VLLM_URL})

    env_vars.append({"name": "VLLM_TLS_VERIFY", "value": LLS_CORE_VLLM_TLS_VERIFY})
    env_vars.append({"name": "VLLM_MAX_TOKENS", "value": LLS_CORE_VLLM_MAX_TOKENS})

    env_vars.append({"name": "FMS_ORCHESTRATOR_URL", "value": "http://localhost"})

    # EMBEDDING_MODEL
    embedding_provider = params.get("embedding_provider") or "vllm-embedding"

    if embedding_provider == "vllm-embedding":
        env_vars.append({"name": "EMBEDDING_MODEL", "value": LLS_CORE_EMBEDDING_MODEL})
        env_vars.append({"name": "EMBEDDING_PROVIDER_MODEL_ID", "value": LLS_CORE_EMBEDDING_PROVIDER_MODEL_ID})
        env_vars.append({"name": "VLLM_EMBEDDING_URL", "value": LLS_CORE_VLLM_EMBEDDING_URL})
        env_vars.append(
            {
                "name": "VLLM_EMBEDDING_API_TOKEN",
                "valueFrom": {
                    "secretKeyRef": {"name": "llamastack-distribution-secret", "key": "vllm-embedding-api-token"}
                },
            },
        )
        env_vars.append({"name": "VLLM_EMBEDDING_MAX_TOKENS", "value": LLS_CORE_VLLM_EMBEDDING_MAX_TOKENS})
        env_vars.append({"name": "VLLM_EMBEDDING_TLS_VERIFY", "value": LLS_CORE_VLLM_EMBEDDING_TLS_VERIFY})
    elif embedding_provider == "sentence-transformers":
        # Increase CPU limits to prevent timeouts when inserting files into vector stores
        cpu_requests = "4"
        cpu_limits = "8"

        # Enable sentence-transformers embedding model
        env_vars.append({"name": "ENABLE_SENTENCE_TRANSFORMERS", "value": "true"})
        env_vars.append({"name": "EMBEDDING_PROVIDER", "value": "sentence-transformers"})
        # Explicitly set EMBEDDING_MODEL and EMBEDDING_PROVIDER_MODEL_ID.
        # This overrides the default sentence-transformer model (nomic-embed-text-v1.5).
        env_vars.append({"name": "EMBEDDING_MODEL", "value": "ibm-granite/granite-embedding-125m-english"})
        env_vars.append({"name": "EMBEDDING_PROVIDER_MODEL_ID", "value": "ibm-granite/granite-embedding-125m-english"})

        if is_disconnected_cluster:
            # Workaround to fix sentence-transformer embeddings on disconnected (RHAIENG-1624)
            env_vars.append({"name": "SENTENCE_TRANSFORMERS_HOME", "value": "/opt/app-root/src/.cache/huggingface/hub"})
            env_vars.append({"name": "HF_HUB_OFFLINE", "value": "1"})
            env_vars.append({"name": "TRANSFORMERS_OFFLINE", "value": "1"})
            env_vars.append({"name": "HF_DATASETS_OFFLINE", "value": "1"})

    else:
        raise ValueError(f"Unsupported embeddings provider: {embedding_provider}")

    # POSTGRESQL environment variables for sql_default and kvstore_default
    env_vars.append({"name": "POSTGRES_HOST", "value": "vector-io-postgres-service"})
    env_vars.append({"name": "POSTGRES_PORT", "value": "5432"})
    env_vars.append(
        {
            "name": "POSTGRES_USER",
            "valueFrom": {"secretKeyRef": {"name": "llamastack-distribution-secret", "key": "postgres-user"}},
        },
    )
    env_vars.append(
        {
            "name": "POSTGRES_PASSWORD",
            "valueFrom": {"secretKeyRef": {"name": "llamastack-distribution-secret", "key": "postgres-password"}},
        },
    )
    env_vars.append({"name": "POSTGRES_DB", "value": "ps_db"})
    env_vars.append({"name": "POSTGRES_TABLE_NAME", "value": "llamastack_kvstore"})

    # Depending on parameter files_provider, configure files provider and obtain required env_vars
    files_provider = params.get("files_provider") or "local"
    env_vars_files = files_provider_config_factory(provider_name=files_provider)
    env_vars.extend(env_vars_files)

    # Depending on parameter vector_io_provider, deploy vector_io provider and obtain required env_vars
    vector_io_provider = params.get("vector_io_provider") or "milvus"
    env_vars_vector_io = vector_io_provider_deployment_config_factory(provider_name=vector_io_provider)
    env_vars.extend(env_vars_vector_io)

    if is_disconnected_cluster and HTTPS_PROXY:
        LOGGER.info("Setting proxy and tlsconfig configuration")
        env_vars.append({"name": "HTTPS_PROXY", "value": HTTPS_PROXY})

        # The operator sets SSL_CERT_FILE automatically when tls.trust is
        # configured, but the `requests` library (used by tiktoken to download
        # tokenizer data) ignores SSL_CERT_FILE and only checks REQUESTS_CA_BUNDLE.
        # Without this, tiktoken fails with SSL CERTIFICATE_VERIFY_FAILED when the
        # proxy uses a self-signed certificate (e.g. in disconnected clusters).
        env_vars.append({
            "name": "REQUESTS_CA_BUNDLE",
            "value": "/etc/ssl/certs/ca-bundle/ca-bundle.crt",
        })

        tls_config = {
            "trust": {
                "caCertificates": [
                    {"name": "odh-trusted-ca-bundle", "key": "ca-bundle.crt"},
                    {"name": "odh-trusted-ca-bundle", "key": "odh-ca-bundle.crt"},
                ],
            },
        }

    config: dict[str, Any] = {
        "distribution": {"name": "rh-dev"},
        "workload": {
            "resources": {
                "requests": {"cpu": cpu_requests, "memory": "3Gi"},
                "limits": {"cpu": cpu_limits, "memory": "6Gi"},
            },
            "overrides": {
                "env": env_vars,
            },
        },
    }

    if tls_config:
        config["tls"] = tls_config

    if params.get("ogx_storage_size"):
        if is_disconnected_cluster:
            LOGGER.warning("Skipping storage_size configuration on disconnected clusters due to known bug RHAIENG-1819")
        else:
            storage_size = params.get("ogx_storage_size")
            config["workload"]["storage"] = {"size": storage_size}

    return config
