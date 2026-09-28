from collections.abc import Callable
from typing import Any

import structlog

from tests.ogx.constants import (
    GEMINI_INFERENCE_MODEL,
    HTTPS_PROXY,
    OGX_CORE_EMBEDDING_MODEL,
    OGX_CORE_EMBEDDING_PROVIDER_MODEL_ID,
    OGX_CORE_INFERENCE_MODEL,
    OGX_CORE_VLLM_EMBEDDING_MAX_TOKENS,
    OGX_CORE_VLLM_EMBEDDING_TLS_VERIFY,
    OGX_CORE_VLLM_EMBEDDING_URL,
    OGX_CORE_VLLM_MAX_TOKENS,
    OGX_CORE_VLLM_TLS_VERIFY,
    OGX_CORE_VLLM_URL,
)

LOGGER = structlog.get_logger(name=__name__)


def build_ogx_server_config(
    vector_io_provider_deployment_config_factory: Callable[[str], list[dict[str, str]]],
    files_provider_config_factory: Callable[[str], list[dict[str, str]]],
    is_disconnected_cluster: bool,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Build OGXServer spec configuration matching the v1beta1 CRD.

    Assembles the ``distribution``, ``workload``, and optional ``tls`` sections.
    Core VLLM inference, vLLM embedding, and PostgreSQL kvstore environment
    variables are always set from ``tests.ogx.constants``; provider-specific
    variables come from the factory callables. The caller is responsible for
    adding network policy configuration.

    On disconnected clusters, when ``HTTPS_PROXY`` is set, proxy and trusted-CA
    TLS configuration is added. ``ogx_storage_size`` is ignored on disconnected
    clusters (RHAIENG-1819).

    Args:
        vector_io_provider_deployment_config_factory: Factory that deploys a
            vector I/O provider and returns its environment variable dicts.
        files_provider_config_factory: Factory that configures a files storage
            provider and returns its environment variable dicts.
        is_disconnected_cluster: Whether the target cluster is disconnected (air-gapped).
        params: Optional configuration overrides:
            - embedding_provider: Embedding backend; only ``"vllm-embedding"``
              is supported (default).
            - vector_io_provider: Vector I/O provider id passed to the factory
              (e.g. ``"milvus"``, ``"milvus-remote"``, ``"pgvector"``,
              ``"qdrant-remote"``, ``"faiss"``; default ``"milvus-remote"``).
            - files_provider: Files storage provider (``"local"`` or ``"s3"``;
              default ``"local"``).
            - ogx_storage_size: PVC size for workload storage (e.g. ``"2Gi"``).
            - enable_gemini: When truthy, inject ``GEMINI_API_KEY`` (from the
              ``ogx-distribution-secret``) so the distribution's conditional
              ``remote::gemini`` provider activates. Optionally also sets
              ``GEMINI_INFERENCE_MODEL`` when configured in constants.

    Returns:
        OGXServerSpec configuration dict with ``distribution``, ``workload``,
        and optional ``tls`` sections.
    """

    env_vars = []
    tls_config: dict[str, Any] | None = None
    cpu_requests = "1"
    cpu_limits = "2"

    env_vars.extend((
        {"name": "INFERENCE_MODEL", "value": OGX_CORE_INFERENCE_MODEL},
        {
            "name": "VLLM_API_TOKEN",
            "valueFrom": {"secretKeyRef": {"name": "ogx-distribution-secret", "key": "vllm-api-token"}},
        },
        {"name": "VLLM_URL", "value": OGX_CORE_VLLM_URL},
        {"name": "VLLM_TLS_VERIFY", "value": OGX_CORE_VLLM_TLS_VERIFY},
        {"name": "VLLM_MAX_TOKENS", "value": OGX_CORE_VLLM_MAX_TOKENS},
    ))

    # EMBEDDING_MODEL
    embedding_provider = params.get("embedding_provider") or "vllm-embedding"
    if embedding_provider == "vllm-embedding":
        env_vars.extend((
            {"name": "EMBEDDING_MODEL", "value": OGX_CORE_EMBEDDING_MODEL},
            {"name": "EMBEDDING_PROVIDER_MODEL_ID", "value": OGX_CORE_EMBEDDING_PROVIDER_MODEL_ID},
            {"name": "VLLM_EMBEDDING_URL", "value": OGX_CORE_VLLM_EMBEDDING_URL},
            {
                "name": "VLLM_EMBEDDING_API_TOKEN",
                "valueFrom": {"secretKeyRef": {"name": "ogx-distribution-secret", "key": "vllm-embedding-api-token"}},
            },
            {"name": "VLLM_EMBEDDING_MAX_TOKENS", "value": OGX_CORE_VLLM_EMBEDDING_MAX_TOKENS},
            {"name": "VLLM_EMBEDDING_TLS_VERIFY", "value": OGX_CORE_VLLM_EMBEDDING_TLS_VERIFY},
        ))
    else:
        raise ValueError(f"Unsupported embeddings provider: {embedding_provider}")

    # Remote Gemini inference provider (remote::gemini).
    # The distribution's config.yaml gates the provider on ENABLE_GEMINI:
    #   - provider_id: ${env.ENABLE_GEMINI:+gemini}
    #     provider_type: remote::gemini
    #     config:
    #       api_key: ${env.GEMINI_API_KEY:=}
    # so ENABLE_GEMINI is what makes the provider appear, and GEMINI_API_KEY
    # (sourced from the ogx-distribution-secret) is what authenticates it.
    if params.get("enable_gemini"):
        env_vars.extend((
            {"name": "ENABLE_GEMINI", "value": "1"},
            {
                "name": "GEMINI_API_KEY",
                "valueFrom": {"secretKeyRef": {"name": "ogx-distribution-secret", "key": "gemini-api-token"}},
            },
        ))
        if GEMINI_INFERENCE_MODEL:
            env_vars.append({"name": "GEMINI_INFERENCE_MODEL", "value": GEMINI_INFERENCE_MODEL})

    # POSTGRESQL environment variables for sql_default and kvstore_default
    env_vars.extend((
        {"name": "POSTGRES_HOST", "value": "vector-io-postgres-service"},
        {"name": "POSTGRES_PORT", "value": "5432"},
        {
            "name": "POSTGRES_USER",
            "valueFrom": {"secretKeyRef": {"name": "ogx-distribution-secret", "key": "postgres-user"}},
        },
        {
            "name": "POSTGRES_PASSWORD",
            "valueFrom": {"secretKeyRef": {"name": "ogx-distribution-secret", "key": "postgres-password"}},
        },
        {"name": "POSTGRES_DB", "value": "ps_db"},
        {"name": "POSTGRES_TABLE_NAME", "value": "ogx_kvstore"},
    ))

    # Depending on parameter files_provider, configure files provider and obtain required env_vars
    files_provider = params.get("files_provider") or "local"
    env_vars_files = files_provider_config_factory(provider_name=files_provider)
    env_vars.extend(env_vars_files)

    # Depending on parameter vector_io_provider, deploy vector_io provider and obtain required env_vars
    vector_io_provider = params.get("vector_io_provider") or "milvus-remote"
    env_vars_vector_io = vector_io_provider_deployment_config_factory(provider_name=vector_io_provider)
    env_vars.extend(env_vars_vector_io)

    if is_disconnected_cluster and HTTPS_PROXY:
        LOGGER.info("Setting proxy and tlsconfig configuration")
        env_vars.extend(({"name": "HTTPS_PROXY", "value": HTTPS_PROXY},))

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
        "distribution": {"name": "rh"},
        "network": {"externalAccess": {"enabled": True}},
        "workload": {
            "resources": {
                "requests": {"cpu": cpu_requests, "memory": "1Gi"},
                "limits": {"cpu": cpu_limits, "memory": "2Gi"},
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
