"""Utility functions for Model Registry Python Client Signing Tests."""

from tests.model_registry.model_registry.python_client.signing.constants import (
    SECURESIGN_ORGANIZATION_EMAIL,
    SECURESIGN_ORGANIZATION_NAME,
)


def get_organization_config() -> dict[str, str]:
    """Get organization configuration for certificates."""
    return {
        "organizationName": SECURESIGN_ORGANIZATION_NAME,
        "organizationEmail": SECURESIGN_ORGANIZATION_EMAIL,
    }


def is_securesign_ready(securesign_instance: dict) -> bool:
    """Check if a Securesign instance is ready.

    Args:
        securesign_instance: Securesign instance dictionary from Kubernetes API

    Returns:
        bool: True if instance has Ready condition with status True
    """
    conditions = securesign_instance.get("status", {}).get("conditions", [])
    ready = [
        condition for condition in conditions if condition.get("type") == "Ready" and condition.get("status") == "True"
    ]
    return bool(ready)


def get_tas_service_urls(securesign_instance: dict) -> dict[str, str]:
    """Extract TAS service URLs from Securesign instance status.

    Args:
        securesign_instance: Securesign instance dictionary from Kubernetes API

    Returns:
        dict: Service URLs with keys 'fulcio', 'rekor', 'tsa', 'tuf'

    Raises:
        KeyError: If expected status fields are missing from Securesign instance
    """
    status = securesign_instance["status"]

    return {
        "fulcio": status["fulcio"]["url"],
        "rekor": status["rekor"]["url"],
        "tsa": status["tsa"]["url"],
        "tuf": status["tuf"]["url"],
    }


def create_connection_type_field(
    name: str, description: str, env_var: str, default_value: str, required: bool = True
) -> dict:
    """Create a Connection Type field dictionary for ODH dashboard.

    Args:
        name: Display name of the field shown in UI
        description: Help text describing the field's purpose
        env_var: Environment variable name for programmatic access
        default_value: Default value to populate (typically a service URL)
        required: Whether the field must be filled

    Returns:
        dict: Field dictionary conforming to ODH Connection Type schema
    """
    return {
        "type": "short-text",
        "name": name,
        "description": description,
        "envVar": env_var,
        "properties": {"defaultValue": default_value},
        "required": required,
    }
