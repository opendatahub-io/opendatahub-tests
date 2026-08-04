class SharedImages:
    """Shared container images used across multiple test components.

    Images used by only one component should go in that component's
    image_constants.py instead (e.g. tests/ai_safety/image_constants.py).
    """

    POSTGRESQL_15: str = (
        "registry.redhat.io/rhel9/postgresql-15"
        "@sha256:90ec347a35ab8a5d530c8d09f5347b13cc71df04f3b994bfa8b1a409b1171d59"  # pragma: allowlist secret
    )

    # MLServer model car images (shared across model_serving and ai_hub)
    MLSERVER_SKLEARN: str = "oci://quay.io/jooholee/mlserver-sklearn@sha256:ec9bc6b520909c52bd1d4accc2b2d28adb04981bd4c3ce94f17f23dd573e1f55"  # noqa: E501
    MLSERVER_XGBOOST: str = "oci://quay.io/jooholee/mlserver-xgboost@sha256:5b6982bdc939b53a7a1210f56aa52bf7de0f0cbc693668db3fd1f496571bff29"  # noqa: E501
    MLSERVER_LIGHTGBM: str = "oci://quay.io/jooholee/mlserver-lightgbm@sha256:77eb15a2eccefa3756faaf2ee4bc1e63990b746427d323957c461f33a4f1a6a3"  # noqa: E501
    MLSERVER_ONNX: str = (
        "oci://quay.io/syedali/mlserver-onnx@sha256:1724ae50e1178a11c3b8dd3c65c03e85d3f416e5994c80c63bcc556c71189e9d"  # noqa: E501
    )

    BUSYBOX: str = (
        "quay.io/quay/busybox"
        "@sha256:92f3298bf80a1ba949140d77987f5de081f010337880cd771f7e7fc928f8c74d"  # pragma: allowlist secret
    )
