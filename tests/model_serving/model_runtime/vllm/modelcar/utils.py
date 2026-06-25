import base64
import binascii
import json
import re

import pytest

from tests.model_serving.model_runtime.vllm.modelcar.constant import MODELCAR_REGISTRIES


def normalize_registry_pull_auth(raw_value: str, expected_host: str | None = None) -> str:
    """Return base64 auth from a plain string or JSON registry credentials object.

    Accepts either a raw base64 auth string (legacy) or JSON with an ``auth`` field,
    e.g. ``{"host": "registry.redhat.io", "auth": "<base64>", "content": "..."}``.
    Docker config JSON with ``auths`` is also supported when ``expected_host`` is set.

    Args:
        raw_value: Plain base64 auth or JSON credentials string.
        expected_host: Optional registry host for resolving ``auths`` entries.

    Returns:
        Base64-encoded Docker registry auth string.

    Raises:
        ValueError: If JSON is invalid or does not contain an auth value.
    """
    stripped = raw_value.strip()
    if not stripped:
        return stripped

    if not stripped.startswith(("{", "[")):
        return stripped

    try:
        parsed: object = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError("Registry pull secret is not valid JSON") from exc

    if isinstance(parsed, list):
        if not parsed:
            raise ValueError("Registry pull secret JSON array must not be empty")
        parsed = parsed[0]

    if not isinstance(parsed, dict):
        raise TypeError("Registry pull secret JSON must be an object")

    top_level_auth = parsed.get("auth")
    if isinstance(top_level_auth, str) and top_level_auth.strip():
        return top_level_auth.strip()

    auths = parsed.get("auths")
    if isinstance(auths, dict):
        if expected_host and expected_host in auths:
            host_auth = auths[expected_host]
            if isinstance(host_auth, dict):
                nested_auth = host_auth.get("auth")
                if isinstance(nested_auth, str) and nested_auth.strip():
                    return nested_auth.strip()

        if len(auths) == 1:
            host_auth = next(iter(auths.values()))
            if isinstance(host_auth, dict):
                nested_auth = host_auth.get("auth")
                if isinstance(nested_auth, str) and nested_auth.strip():
                    return nested_auth.strip()

    raise ValueError("Registry pull secret JSON must include a string 'auth' field")


def collect_modelcar_registry_credentials(
    pytestconfig: pytest.Config,
    *,
    required: bool = False,
) -> tuple[list[str], list[str]]:
    """Collect modelcar OCI registry hosts and pull secrets from pytest CLI/env config.

    Args:
        pytestconfig: Pytest config with per-registry pull-secret options.
        required: When True, raise if no registry pull secret is configured.

    Returns:
        Tuple of (registry_hosts, pull_secrets) for configured registries.

    Raises:
        ValueError: If required is True and no pull secret is configured, or auth is invalid.
    """
    hosts: list[str] = []
    secrets: list[str] = []

    for registry in MODELCAR_REGISTRIES:
        raw_secret = getattr(pytestconfig.option, registry.option_dest, None)
        if not raw_secret:
            continue
        auth = normalize_registry_pull_auth(raw_value=raw_secret, expected_host=registry.host)
        validate_registry_pull_auth(auth=auth)
        hosts.append(registry.host)
        secrets.append(raw_secret)

    if not hosts and required:
        options_help = ", ".join(f"`{registry.cli_option}` or `{registry.env_var}`" for registry in MODELCAR_REGISTRIES)
        raise ValueError(f"No modelcar registry pull secret is configured. Set at least one of: {options_help}")

    return hosts, secrets


def validate_registry_pull_auth(auth: str) -> None:
    """Validate that a registry pull auth value is valid base64.

    Args:
        auth: Base64-encoded Docker registry auth string.

    Raises:
        ValueError: If auth is not valid base64.
    """
    try:
        base64.b64decode(s=auth, validate=True)
    except binascii.Error as exc:
        raise ValueError("Registry pull secret is not a valid base64 encoded string") from exc


def safe_k8s_name(model_name: str, max_length: int = 20) -> str:
    """
    Create a safe Kubernetes name from model_name by truncating to max_length characters
    and ensuring it follows Kubernetes naming conventions.

    Args:
        model_name: The original model name
        max_length: Maximum length for the name (default: 20)

    Returns:
        A valid Kubernetes name truncated to max_length characters
    """
    if not model_name:
        return "default-model"

    safe_name = re.sub(r"[^a-z0-9-]", "-", model_name.lower())
    safe_name = re.sub(r"-+", "-", safe_name)
    safe_name = safe_name.strip("-")

    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length]

    safe_name = safe_name.rstrip("-")

    if not safe_name:
        return "model"

    return safe_name
