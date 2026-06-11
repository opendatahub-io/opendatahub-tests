"""Constants for fast template overlay validation tests."""

import pytest

from utilities.constants import RuntimeTemplates

SUPPORT_STATUS_ANNOTATION = "opendatahub.io/support-status"
UNSUPPORTED_VALUE = "unsupported"

FAST_SUFFIXES = ("fast-1", "fast-2")

VLLM_FAST_OVERLAY_BASES: list[tuple[str, str]] = [
    (RuntimeTemplates.VLLM_CUDA, "vllm_cuda"),
    (RuntimeTemplates.VLLM_ROCM, "vllm_rocm"),
    (RuntimeTemplates.VLLM_GAUDI, "vllm_gaudi"),
    (RuntimeTemplates.VLLM_SPYRE, "vllm_spyre_x86"),
    (RuntimeTemplates.VLLM_SPYRE_S390X, "vllm_spyre_s390x"),
    (RuntimeTemplates.VLLM_SPYRE_PPC64LE, "vllm_spyre_ppc64le"),
]

FAST_TEMPLATE_PARAMS = [
    pytest.param(
        {"base_template": base, "suffix": suffix},
        id=f"test_{short_name}_{suffix.replace('-', '_')}",
    )
    for base, short_name in VLLM_FAST_OVERLAY_BASES
    for suffix in FAST_SUFFIXES
]
