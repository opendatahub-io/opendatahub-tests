from typing import Any

from tests.ai_hub.constants import MODEL_ARTIFACT, MODEL_REGISTRY_BASE_URI  # noqa: F401

MODEL_REGISTER: dict[str, Any] = {
    "name": "model-rest-api",
    "description": "Model created via rest call",
    "owner": "opendatahub-tests",
    "customProperties": {
        "test_rm_bool_property": {"bool_value": False, "metadataType": "MetadataBoolValue"},
        "test_rm_str_property": {"string_value": "my_value", "metadataType": "MetadataStringValue"},
    },
}
MODEL_VERSION: dict[str, Any] = {
    "name": "v0.0.1",
    "state": "LIVE",
    "author": "opendatahub-tests",
    "description": "Model version created via rest call",
    "customProperties": {
        "test_mv_bool_property": {"bool_value": True, "metadataType": "MetadataBoolValue"},
        "test_mv_str_property": {"string_value": "my_value", "metadataType": "MetadataStringValue"},
    },
}

MODEL_REGISTER_DATA: dict[str, Any] = {
    "register_model_data": MODEL_REGISTER,
    "model_version_data": MODEL_VERSION,
    "model_artifact_data": MODEL_ARTIFACT,
}
CUSTOM_PROPERTY = {
    "customProperties": {
        "my_bool_property": {"bool_value": True, "metadataType": "MetadataBoolValue"},
        "my_str_property": {"string_value": "my_value", "metadataType": "MetadataStringValue"},
        "my_double_property": {"double_value": 500.01, "metadataType": "MetadataDoubleValue"},
    }
}
MODEL_VERSION_DESCRIPTION = {"description": "updated model version description"}
STATE_ARCHIVED = {"state": "ARCHIVED"}
STATE_LIVE = {"state": "LIVE"}
REGISTERED_MODEL_DESCRIPTION = {"description": "updated registered model description"}
MODEL_FORMAT_VERSION = {"modelFormatVersion": "v2"}
MODEL_FORMAT_NAME = {"modelFormatName": "tensorflow"}
MODEL_ARTIFACT_DESCRIPTION = {"description": "updated artifact description"}
MARIADB_METADATA_DB = {"db_name": "mariadb"}
