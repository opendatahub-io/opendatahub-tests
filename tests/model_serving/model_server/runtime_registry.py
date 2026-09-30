"""Runtime and model-car pairs used by architecture-aware feature tests."""

from dataclasses import dataclass

from utilities.constants import ModelFormat, RuntimeTemplates
from utilities.image_constants import SharedImages


@dataclass(frozen=True)
class RuntimeProfile:
    """Keep the runtime template and its compatible OCI model together."""

    template: str
    model_car_image: str
    model_car_format: str


ARCH_RUNTIME_REGISTRY: dict[str, RuntimeProfile] = {
    "amd64": RuntimeProfile(RuntimeTemplates.OVMS_KSERVE, SharedImages.MODELCAR_MNIST_8_1, ModelFormat.OPENVINO),
    "arm64": RuntimeProfile(RuntimeTemplates.MLSERVER, SharedImages.MLSERVER_ONNX, ModelFormat.ONNX),
}


def get_runtime_profile(arch: str) -> RuntimeProfile:
    """Select the existing runtime/model pair for a validated architecture."""
    return ARCH_RUNTIME_REGISTRY[arch]


def resolve_cluster_arch(configured: str, detected: str | None) -> str:
    """Resolve an explicit override or detected worker architecture without guessing."""
    arch = detected if configured == "auto" else configured
    if arch not in ARCH_RUNTIME_REGISTRY:
        raise ValueError(f"Unsupported or undetected cluster architecture: {arch!r}; use --cluster-arch=amd64|arm64")
    return arch
