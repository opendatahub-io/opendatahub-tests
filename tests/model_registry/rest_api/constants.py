from typing import Any

from utilities.constants import ModelFormat

MODEL_REGISTER: dict[str, str] = {
    "name": "model-rest-api",
    "description": "Model created via rest call",
    "owner": "opendatahub-tests",
}
MODEL_VERSION: dict[str, Any] = {
    "name": "v0.0.1",
    "state": "LIVE",
    "author": "opendatahub-tests",
    "description": "Model version created via rest call",
}

MODEL_ARTIFACT: dict[str, Any] = {
    "name": "model-artifact-rest-api",
    "description": "Model artifact created via rest call",
    "uri": "https://huggingface.co/openai-community/gpt2/resolve/main/onnx/decoder_model.onnx",
    "state": "UNKNOWN",
    "modelFormatName": ModelFormat.ONNX,
    "modelFormatVersion": "v1",
    "artifactType": "model-artifact",
}
MODEL_REGISTER_DATA = {
    "register_model_data": MODEL_REGISTER,
    "model_version_data": MODEL_VERSION,
    "model_artifact_data": MODEL_ARTIFACT,
}
MODEL_REGISTRY_BASE_URI = "/api/model_registry/v1alpha3/"
