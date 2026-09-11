class SharedImages:
    """Shared container images used across multiple test components.

    Images used by only one component should go in that component's
    image_constants.py instead (e.g. tests/ai_safety/image_constants.py).
    """

    MILVUS: str = "docker.io/milvusdb/milvus@sha256:3d772c3eae3a6107b778636cea5715b9353360b92e5dcfdcaf4ca7022f4f497c"
    ETCD: str = "quay.io/coreos/etcd@sha256:3397341272b9e0a6f44d7e3fc7c321c6efe6cbe82ce866b9b01d0c704bfc5bf3"

    POSTGRESQL_15: str = (
        "registry.redhat.io/rhel9/postgresql-15"
        "@sha256:90ec347a35ab8a5d530c8d09f5347b13cc71df04f3b994bfa8b1a409b1171d59"  # pragma: allowlist secret
    )

    # MLServer model car images (shared across model_serving and ai_hub)
    MLSERVER_SKLEARN: str = "oci://quay.io/opendatahub/modelcar-mlserver-sklearn@sha256:671379c7d10c5f7ea3e7ad493ec563733d615556496c0d350df0f5a87f562c61"  # noqa: E501
    MLSERVER_XGBOOST: str = "oci://quay.io/opendatahub/modelcar-mlserver-xgboost@sha256:b4de2418d3c843d486b977777346f1cf2518b56df0780f78e2b55c01e6274b02"  # noqa: E501
    MLSERVER_LIGHTGBM: str = "oci://quay.io/opendatahub/modelcar-mlserver-lightgbm@sha256:2e4c2aff76656b3547e8af21728818eb586080202ae23a8b5155ac59f57d8328"  # noqa: E501
    MLSERVER_ONNX: str = "oci://quay.io/opendatahub/modelcar-mlserver-onnx@sha256:d7747270ba666c0585dc20f38425811e3d901f150618237d4ab94781b3ab31b7"  # noqa: E501

    BUSYBOX: str = (
        "quay.io/quay/busybox"
        "@sha256:92f3298bf80a1ba949140d77987f5de081f010337880cd771f7e7fc928f8c74d"  # pragma: allowlist secret
    )

    MODELCAR_MNIST_8_1: str = "oci://quay.io/opendatahub/modelcar-openvino@sha256:c92d7d0cb4a1e798ab6a6c4370259a081c6f335bad26627a99046607873b0e42"  # noqa: E501
    MODELCAR_GRANITE_8B_CODE_INSTRUCT: str = "oci://registry.redhat.io/rhelai1/modelcar-granite-8b-code-instruct@sha256:e23eafe347ecdcaf219da6b573f3ef9f526f86543f7bad8e7d3329b36f0bc631"  # noqa: E501

    OCI_TINYLLAMA: str = "oci://quay.io/opendatahub/modelcar-vllm@sha256:45e325523fb05f122f6f27b29d0fe767bc9162a90563a15150c8d4773df4265d"  # noqa: E501

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

    # Helm OCI chart
    OPENSHELL_HELM_CHART: str = "oci://ghcr.io/nvidia/openshell/helm-chart@sha256:87d9c5fe300f400b6c01434b3dc8fabbcf2bcc65f6fbb76ae93d6615a40a5053"  # noqa: E501
