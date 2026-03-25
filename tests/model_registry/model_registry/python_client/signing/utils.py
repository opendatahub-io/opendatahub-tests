"""Utility functions for Model Registry Python Client Signing Tests."""

import hashlib
import os

import requests
import structlog
from pyhelper_utils.shell import run_command

from tests.model_registry.model_registry.python_client.signing.constants import (
    SECURESIGN_NAMESPACE,
    SECURESIGN_ORGANIZATION_EMAIL,
    SECURESIGN_ORGANIZATION_NAME,
)

LOGGER = structlog.get_logger(name=__name__)


def get_organization_config() -> dict[str, str]:
    """Get organization configuration for certificates."""
    return {
        "organizationName": SECURESIGN_ORGANIZATION_NAME,
        "organizationEmail": SECURESIGN_ORGANIZATION_EMAIL,
    }


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


def generate_token(temp_base_folder) -> str:
    """
    Create a service account token and save it to a temporary directory.
    """
    filepath = os.path.join(temp_base_folder, "token")

    LOGGER.info(f"Creating service account token for namespace {SECURESIGN_NAMESPACE}...")
    _, out, _ = run_command(
        command=["oc", "create", "token", "default", "-n", SECURESIGN_NAMESPACE, "--duration=1h"], check=True
    )

    token = out.strip()
    with open(filepath, "w") as fd:
        fd.write(token)
    return filepath


def get_root_checksum(sigstore_tuf_url: str) -> str:
    """
    Download root.json from TUF URL and calculate SHA256 checksum.
    """
    if not sigstore_tuf_url:
        raise ValueError("sigstore_tuf_url cannot be empty or None")

    try:
        LOGGER.info(f"Downloading root.json from: {sigstore_tuf_url}/root.json")
        response = requests.get(f"{sigstore_tuf_url}/root.json", timeout=30)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Calculate SHA256 checksum
        checksum = hashlib.sha256(response.content).hexdigest()
        LOGGER.info(f"Calculated root.json checksum: {checksum}")

    except requests.RequestException as e:
        LOGGER.error(f"Failed to download root.json from {sigstore_tuf_url}: {e}")
        raise
    except (ValueError, OSError) as e:
        LOGGER.error(f"Failed to calculate checksum: {e}")
        raise RuntimeError(f"Checksum calculation failed: {e}")
    else:
        return checksum


def check_model_signature_file(model_dir: str) -> bool:
    """
    Check for the presence of model.sig file in the model directory.

    Args:
        model_dir: Path to the model directory

    Returns:
        bool: True if model.sig file exists, False otherwise
    """
    sig_file_path = os.path.join(model_dir, "model.sig")
    LOGGER.info(f"Checking for signature file: {sig_file_path}")

    if os.path.exists(sig_file_path):
        LOGGER.info(f"Signature file found: {sig_file_path}")
        return True
    else:
        LOGGER.info(f"Signature file not found: {sig_file_path}")
        return False
