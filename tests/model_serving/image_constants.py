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

    TRANSFORMER_IMAGE: str = (
        "quay.io/spolti/kserve-sentiment-custom-transformer"
        "@sha256:6af753f5d13e07fd2d0d3da9e55ddbcd4d5cabcd9d5f4c1fbbdce06fb1e08c67"  # pragma: allowlist secret
    )
