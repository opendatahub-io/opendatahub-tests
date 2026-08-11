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

    MODELCAR_MNIST_8_1: str = (
        "oci://quay.io/mwaykole/test@sha256:cb7d25c43e52c755e85f5b59199346f30e03b7112ef38b74ed4597aec8748743"
    )
    MODELCAR_GRANITE_8B_CODE_INSTRUCT: str = "oci://registry.redhat.io/rhelai1/modelcar-granite-8b-code-instruct@sha256:e23eafe347ecdcaf219da6b573f3ef9f526f86543f7bad8e7d3329b36f0bc631"  # noqa: E501

    OCI_TINYLLAMA: str = (
        "oci://quay.io/mwaykole/test@sha256:8bfd02132b03977ebbca93789e81c4549d8f724ee78fa378616d9ae4387717c8"
    )

    ZOT_REGISTRY: str = (
        "ghcr.io/project-zot/zot@sha256:cd2aea942f428630bcb4190542be6abd35e14177aab84fc7ccad0dca8ecb363d"
    )

    MINIO_KSERVE: str = (
        "quay.io/jooholee/model-minio@sha256:b9554be19a223830cf792d5de984ccc57fc140b954949f5ffc6560fab977ca7a"
    )
    MINIO_QWEN: str = (
        "quay.io/trustyai_testing/hf-llm-minio@sha256:2404a37d578f2a9c7adb3971e26a7438fedbe7e2e59814f396bfa47cd5fe93bb"  # noqa: E501
    )
    MINIO_QWEN_HAP_BPIV2: str = "quay.io/trustyai_testing/qwen2.5-0.5b-instruct-hap-bpiv2-minio@sha256:eac1ca56f62606e887c80b4a358b3061c8d67f0b071c367c0aa12163967d5b2b"  # noqa: E501
    MINIO_MODEL_REGISTRY: str = (
        "quay.io/minio/minio@sha256:14cea493d9a34af32f524e538b8346cf79f3321eff8e708c1e2960462bd8936e"
    )

    MARIADB_1011: str = (
        "registry.redhat.io/rhel9/mariadb-1011@sha256:092407d87f8017bb444a462fb3d38ad5070429e94df7cf6b91d82697f36d0fa9"
    )

    VLLM_CPU: str = "quay.io/pierdipi/vllm-cpu@sha256:ce3a0c057394b2c332498f9742a17fd31b5cc2ef07db882d579fd157fe2c9a98"

    OPENVINO_MODEL_SERVER: str = "quay.io/opendatahub/openvino_model_server@sha256:564664371d3a21b9e732a5c1b4b40bacad714a5144c0a9aaf675baec4a04b148"  # noqa: E501
