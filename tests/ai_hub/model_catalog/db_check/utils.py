import base64
import binascii

from ocp_resources.secret import Secret
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)


def extract_secret_values(secret: Secret) -> dict[str, str]:
    """Extract and decode secret data values from a Secret object.

    Args:
        secret: The Secret object to extract values from

    Returns:
        Dict mapping secret keys to decoded string values
    """
    secret_values = {}
    if secret.instance.data:
        for key, encoded_value in secret.instance.data.items():
            try:
                decoded_value = base64.b64decode(s=encoded_value).decode(encoding="utf-8")
                secret_values[key] = decoded_value
            except (binascii.Error, UnicodeDecodeError) as e:
                LOGGER.warning(f"Failed to decode secret key '{key}': {e}")
                secret_values[key] = encoded_value  # Keep encoded if decode fails

    return secret_values
