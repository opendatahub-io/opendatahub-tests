import base64
import binascii
import json

import structlog
from ocp_resources.secret import Secret

LOGGER = structlog.get_logger(name=__name__)


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


def parse_language_properties_from_db(psql_output: str) -> dict[str, set[str]]:
    """Parse psql output of language properties into a dict of model_name -> language codes.

    Args:
        psql_output: Raw psql output with model_name | language columns

    Returns:
        Dict mapping model names to sets of language code strings
    """
    db_languages: dict[str, set[str]] = {}
    for line in psql_output.strip().splitlines():
        line = line.strip()
        if not line or line.startswith(("-", "(")) or "|" not in line:
            continue
        if "model_name" in line and "language" in line:
            continue
        parts = line.split("|")
        if len(parts) == 2:
            model_name = parts[0].strip()
            raw_value = parts[1].strip()
            try:
                langs = json.loads(raw_value)
                if isinstance(langs, str):
                    db_languages.setdefault(model_name, set()).add(langs)
                elif isinstance(langs, (list, tuple, set)):
                    db_languages.setdefault(model_name, set()).update(langs)
                else:
                    db_languages.setdefault(model_name, set()).add(str(langs))
            except json.JSONDecodeError:
                db_languages.setdefault(model_name, set()).add(raw_value)
    return db_languages


def find_language_mismatches_between_api_and_db(
    db_languages: dict[str, set[str]],
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> list[str]:
    """Compare language properties from the database against the catalog API.

    Args:
        db_languages: Dict of model_name -> set of language codes from the database
        model_catalog_rest_url: REST URL(s) for the model catalog API
        model_registry_rest_headers: Authentication headers for API calls

    Returns:
        List of mismatch descriptions; empty if all match
    """
    from tests.model_registry.utils import execute_get_command

    mismatches = []
    for model_name, db_langs in db_languages.items():
        api_model = execute_get_command(
            url=f"{model_catalog_rest_url[0]}models?pageSize=1&filterQuery=name='{model_name}'",
            headers=model_registry_rest_headers,
        )
        items = api_model.get("items", [])
        if not items:
            mismatches.append(f"{model_name}: not found in API")
            continue

        api_langs = set(items[0].get("language", []))
        if db_langs != api_langs:
            mismatches.append(f"{model_name}: DB={sorted(db_langs)}, API={sorted(api_langs)}")

    return mismatches
