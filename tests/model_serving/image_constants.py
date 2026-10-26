class ModelServingImages:
    """Container images used by model_serving tests."""

    OVMS_SERVING_RUNTIME: str = (
        "quay.io/modh/openvino_model_server"
        "@sha256:53b7fcf95de9b81e4c8652d0bf4e84e22d5b696827a5d951d863420c68b9cfe8"  # pragma: allowlist secret
    )
    # Updated to 25.02 - last Triton release with TensorFlow backend included by default
    # TensorFlow backend was deprecated in 25.03 and removed in 26.x+
    # See: https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/
    TRITON: str = (
        "nvcr.io/nvidia/tritonserver"
        "@sha256:cac5c60eb969f6881e3d2c473e331a5232e1fd510d3fca56cc96e1835af5519d"  # pragma: allowlist secret
    )

    # Multi-arch (amd64 + arm64) rebuild of the sentiment transformer.
    # The amd64 manifest is byte-identical to the original spolti image
    # (quay.io/spolti/kserve-sentiment-custom-transformer@sha256:cd8c290d...);
    # the arm64 manifest enables the transformer auth tests on arm clusters.
    TRANSFORMER_IMAGE: str = (
        "quay.io/model_server/kserve-sentiment-custom-transformer"
        "@sha256:62043a1d95dd084d9a8ddc3775c8bde218020a520889f1af85b24f430c53cf4e"  # pragma: allowlist secret
    )

    VLLM_OMNI_MODELCAR: str = (
        "quay.io/opendatahub/modelcar-vllm-omni"
        ":qwen3-tts-12hz-1.7b-customvoice-v1"
        "@sha256:aa3e5a9583e06f8952d5329195f84a13d0475dd9350aec3896423e1c9cd20706"  # pragma: allowlist secret
    )
