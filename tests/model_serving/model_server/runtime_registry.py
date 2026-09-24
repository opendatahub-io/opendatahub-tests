"""Arch-aware runtime registry for model_server tests.

Maps cluster CPU architecture to the appropriate serving runtime template,
model format, and inference configuration. This enables the same test logic
to run on both x86_64 (OVMS) and ARM64 (MLServer) clusters transparently.

Usage:
    from tests.model_serving.model_server.runtime_registry import (
        get_runtime_profile,
        ARCH_RUNTIME_REGISTRY,
        ClusterArch,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from utilities.constants import ModelFormat, RuntimeTemplates


class ClusterArch:
    AMD64: str = "amd64"
    ARM64: str = "arm64"
    SUPPORTED: list[str] = [AMD64, ARM64]  # noqa: RUF012


@dataclass(frozen=True)
class RuntimeProfile:
    """Encapsulates all runtime-specific configuration for a given arch + model format."""

    template: str
    model_format: str
    runtime_name: str
    inference_config_key: str
    node_selector: dict[str, str] = field(default_factory=dict)
    supported_model_formats: list[dict[str, Any]] | None = None


ARCH_RUNTIME_REGISTRY: dict[str, dict[str, RuntimeProfile]] = {
    ClusterArch.AMD64: {
        "onnx": RuntimeProfile(
            template=RuntimeTemplates.OVMS_KSERVE,
            model_format=ModelFormat.ONNX,
            runtime_name="onnx-runtime",
            inference_config_key="ovms_onnx",
            node_selector={"kubernetes.io/arch": ClusterArch.AMD64},
        ),
        "openvino_ir": RuntimeProfile(
            template=RuntimeTemplates.OVMS_KSERVE,
            model_format=ModelFormat.OPENVINO,
            runtime_name="openvino-runtime",
            inference_config_key="ovms_openvino",
            node_selector={"kubernetes.io/arch": ClusterArch.AMD64},
        ),
    },
    ClusterArch.ARM64: {
        "onnx": RuntimeProfile(
            template=RuntimeTemplates.MLSERVER,
            model_format=ModelFormat.ONNX,
            runtime_name="mlserver-onnx-runtime",
            inference_config_key="mlserver_onnx",
            node_selector={"kubernetes.io/arch": ClusterArch.ARM64},
        ),
    },
}

INFERENCE_CONFIG_MAP: dict[str, str] = {
    "ovms_onnx": "utilities.manifests.onnx.ONNX_INFERENCE_CONFIG",
    "ovms_openvino": "utilities.manifests.openvino.OPENVINO_INFERENCE_CONFIG",
    "mlserver_onnx": "utilities.manifests.onnx.ONNX_INFERENCE_CONFIG",
}


def get_runtime_profile(arch: str, model_format: str = "onnx") -> RuntimeProfile | None:
    """Look up the RuntimeProfile for a given architecture and model format.

    Returns None if the combination is not supported (e.g., openvino_ir on ARM64).
    """
    return ARCH_RUNTIME_REGISTRY.get(arch, {}).get(model_format)


def get_supported_formats(arch: str) -> list[str]:
    """Return list of model formats supported on the given architecture."""
    return list(ARCH_RUNTIME_REGISTRY.get(arch, {}).keys())


def is_format_supported(arch: str, model_format: str) -> bool:
    """Check if a model format is supported on the given architecture."""
    return model_format in ARCH_RUNTIME_REGISTRY.get(arch, {})
