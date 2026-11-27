"""Helpers for the remote::gemini provider test suite.

These utilities wrap the small amount of shared logic used across the Gemini
provider test files: discovering whether the ``remote::gemini`` provider is
active, resolving Gemini model ids dynamically from the running distribution,
building the per-request provider-data override header, and inspecting the
OgxServer pod for injected environment variables and logs.
"""

import json
import re
from typing import Any

import structlog
from ogx_client import OgxClient

from tests.ogx.constants import (
    GEMINI_EMBEDDING_MODEL,
    GEMINI_INFERENCE_MODEL,
    GEMINI_PROVIDER_DATA_HEADER,
    GEMINI_PROVIDER_ID,
    GEMINI_PROVIDER_TYPE,
)

LOGGER = structlog.get_logger(name=__name__)


def list_provider_types(ogx_client: OgxClient) -> list[str]:
    """Return the ``provider_type`` of every provider reported by the distribution.

    Args:
        ogx_client: The configured OgxClient.

    Returns:
        A list of ``provider_type`` strings (e.g. ``"remote::gemini"``).
    """
    return [provider.provider_type for provider in ogx_client.providers.list()]


def is_gemini_provider_active(ogx_client: OgxClient) -> bool:
    """Whether the ``remote::gemini`` provider is present in ``/v1/providers``.

    Args:
        ogx_client: The configured OgxClient.

    Returns:
        True if a provider with ``provider_type == "remote::gemini"`` is listed.
    """
    return GEMINI_PROVIDER_TYPE in list_provider_types(ogx_client=ogx_client)


def resolve_gemini_model_id(ogx_client: OgxClient, model_type: str = "llm") -> str | None:
    """Resolve a Gemini model id served by the ``remote::gemini`` provider.

    Prefers an explicit override from constants (``GEMINI_INFERENCE_MODEL`` /
    ``GEMINI_EMBEDDING_MODEL``); otherwise selects the first model registered by
    the Gemini provider matching ``model_type`` from ``GET /v1/models``.

    Args:
        ogx_client: The configured OgxClient.
        model_type: The model type to select (``"llm"`` or ``"embedding"``).

    Returns:
        The model id, or ``None`` if no matching Gemini model is registered.

    Raises:
        ValueError: If ``model_type`` is not ``"llm"`` or ``"embedding"``.
    """
    if model_type not in ("llm", "embedding"):
        raise ValueError(f"Unsupported model_type {model_type!r}; expected 'llm' or 'embedding'")

    override = GEMINI_INFERENCE_MODEL if model_type == "llm" else GEMINI_EMBEDDING_MODEL
    if override:
        return override

    for model in ogx_client.models.list().data:
        metadata = getattr(model, "custom_metadata", None) or {}
        if metadata.get("model_type") == model_type and metadata.get("provider_id") == GEMINI_PROVIDER_ID:
            return model.id

    return None


def provider_data_headers(gemini_api_key: str) -> dict[str, str]:
    """Build the ``x-ogx-provider-data`` header for a per-request API key override.

    Args:
        gemini_api_key: The Gemini API key to send for this request only.

    Returns:
        A headers dict suitable for passing as ``extra_headers`` to an
        OpenAI-compatible OgxClient call.
    """
    return {GEMINI_PROVIDER_DATA_HEADER: json.dumps({"gemini_api_key": gemini_api_key})}


def pod_env_var_is_set(pod: Any, name: str) -> bool:
    """Return whether an environment variable is set (non-empty) in the pod.

    Checks the value inside the running container without exposing it, mirroring
    the ``test -n "${VAR}"`` idiom from the test plan so that secret values are
    never printed to logs.

    Args:
        pod: An ``ocp_resources`` Pod object for the OgxServer pod.
        name: The environment variable name to check.

    Returns:
        True if the variable is set and non-empty inside the container.

    Raises:
        ValueError: If ``name`` is not a valid shell environment-variable
            identifier. Guards against shell injection (CWE-78) since ``name``
            is interpolated into an ``sh -c`` command.
    """
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Invalid environment variable name {name!r}")

    output = pod.execute(command=["sh", "-c", f'test -n "${{{name}}}" && echo SET || echo UNSET'])
    return output.strip() == "SET"
