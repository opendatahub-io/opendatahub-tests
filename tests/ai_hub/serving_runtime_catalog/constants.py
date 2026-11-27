"""Controlled discovery metadata for serving runtime catalog integration tests."""

import base64
import json
from typing import Any

from utilities.image_constants import SharedImages

RUNTIME_API_PATH: str = "/api/serving_runtime_catalog/v1/"
SOURCE_API_PATH: str = "/api/model_catalog/v1/"
USER_CONFIGMAP: str = "serving-runtime-catalog-sources"
DEFAULT_SOURCE_ID: str = "rh_serving_runtimes"
DEFAULT_LABEL: str = "Red Hat Serving Runtimes"
DEFAULT_CATALOG_PATH: str = "/shared-data/redhat-serving-runtimes-catalog.yaml"
USER_MOUNT_PATH: str = "/data/user-serving-runtime-sources"
SOURCE_ALPHA: str = "odh-test-runtime-alpha"
SOURCE_BETA: str = "odh-test-runtime-beta"
SOURCE_UNLABELED: str = "odh-test-runtime-unlabeled"
LABEL_ALPHA: str = "ODH Test Runtime Alpha"
LABEL_BETA: str = "ODH Test Runtime Beta"
ABSENT_RUNTIME_ID: str = "2147483647"
LOGO_BYTES: bytes = b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>'
LOGO_URL: str = "https://example.invalid/runtime-logo.png"
SUPPORT_LEVELS: tuple[str, ...] = ("supported", "techPreview", "developerPreview", "community")

RUNTIME_TEMPLATE: str = json.dumps({
    "apiVersion": "serving.kserve.io/v1alpha1",
    "kind": "ServingRuntime",
    "metadata": {"name": "catalog-test-runtime"},
    "spec": {
        "supportedModelFormats": [{"name": "onnx"}],
        "containers": [{"name": "kserve-container", "image": SharedImages.VLLM_CPU}],
    },
})

TEST_SOURCES: list[dict[str, Any]] = [
    {
        "id": source_id,
        "name": source_id,
        "type": "yaml",
        "enabled": True,
        "labels": labels,
        "properties": {"yamlCatalogPath": f"{source_id}.yaml"},
    }
    for source_id, labels in (
        (SOURCE_ALPHA, [LABEL_ALPHA]),
        (SOURCE_BETA, [LABEL_BETA]),
        (SOURCE_UNLABELED, []),
    )
]

TEST_CATALOGS: dict[str, dict[str, Any]] = {
    SOURCE_ALPHA: {
        "serving_runtimes": [
            {
                "name": "catalog-alpha",
                "displayName": "Quartz runtime",
                "provider": "Acme inference",
                "description": "Fast zephyr inference for integration testing.",
                "readme": "# Controlled runtime\nDiscovery metadata only.",
                "logo": f"data:image/svg+xml;base64,{base64.b64encode(LOGO_BYTES).decode()}",
                "tags": ["catalog-test", "generative"],
                "license": "Apache-2.0",
                "documentationUrl": "https://example.invalid/runtime-docs",
                "supportedModelFormats": [{"name": "onnx", "autoSelect": True}],
                "capabilities": {
                    "requiresGPU": True,
                    "supportedAccelerators": ["nvidia.com/gpu"],
                    "multiModel": False,
                },
                "customProperties": {"test_owner": {"metadataType": "MetadataStringValue", "string_value": "QE"}},
                "versions": [
                    {
                        "version": f"{index + 1}.0.0",
                        "image": SharedImages.VLLM_CPU,
                        "supportLevel": support_level,
                        "minimumRHOAIVersion": "3.6",
                        "deprecated": index == 3,
                        "supportedModelFormats": [{"name": "onnx", "autoSelect": True}],
                        "protocolVersions": ["v2", "grpc-v2"],
                        "recommendedResources": {"recommended": {"cpu": "2", "memory": "4Gi"}},
                        "defaultArgs": ["--port", "8080"],
                        "env": [
                            {"name": "LOG_LEVEL", "defaultValue": "INFO", "secret": False},
                            {"name": "ACCESS_TOKEN", "required": True, "secret": True},
                        ],
                        "servingRuntimeTemplate": RUNTIME_TEMPLATE,
                    }
                    for index, support_level in enumerate(SUPPORT_LEVELS)
                ],
            },
            {
                "name": "catalog-bravo",
                "displayName": "CPU runtime",
                "provider": "Beta inference",
                "tags": ["catalog-test", "predictive"],
                "logo": LOGO_URL,
                "supportedModelFormats": [{"name": "onnx"}],
                "capabilities": {"requiresGPU": False, "multiModel": True},
                "versions": [{"version": "1.0.0", "image": SharedImages.VLLM_CPU, "supportLevel": "community"}],
            },
            {
                "name": "catalog-charlie",
                "provider": "Acme inference",
                "tags": ["catalog-test"],
                "capabilities": {"requiresGPU": False, "multiModel": False},
                "versions": [{"version": "1.0.0", "image": SharedImages.VLLM_CPU, "supportLevel": "supported"}],
            },
        ],
    },
    SOURCE_BETA: {
        "serving_runtimes": [
            {
                "name": "catalog-alpha",
                "provider": "Separate source",
                "tags": ["catalog-test"],
                "versions": [{"version": "9.0.0", "image": SharedImages.VLLM_CPU, "supportLevel": "community"}],
            },
        ],
    },
    SOURCE_UNLABELED: {
        "serving_runtimes": [
            {
                "name": "catalog-unlabeled",
                "tags": ["catalog-test"],
                "versions": [{"version": "1.0.0", "image": SharedImages.VLLM_CPU, "supportLevel": "supported"}],
            },
        ],
    },
}
