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
    MLSERVER_SKLEARN: str = "oci://quay.io/opendatahub/modelcar-mlserver-sklearn@sha256:671379c7d10c5f7ea3e7ad493ec563733d615556496c0d350df0f5a87f562c61"  # noqa: E501
    MLSERVER_XGBOOST: str = "oci://quay.io/opendatahub/modelcar-mlserver-xgboost@sha256:b4de2418d3c843d486b977777346f1cf2518b56df0780f78e2b55c01e6274b02"  # noqa: E501
    MLSERVER_LIGHTGBM: str = "oci://quay.io/opendatahub/modelcar-mlserver-lightgbm@sha256:2e4c2aff76656b3547e8af21728818eb586080202ae23a8b5155ac59f57d8328"  # noqa: E501
    MLSERVER_ONNX: str = "oci://quay.io/opendatahub/modelcar-mlserver-onnx@sha256:d7747270ba666c0585dc20f38425811e3d901f150618237d4ab94781b3ab31b7"  # noqa: E501

    SEAWEEDFS: str = (
        "ghcr.io/chrislusf/seaweedfs@sha256:6620371e8af8282056685c652d4637265698c9e2c2d59f9594e6ac333ad6c634"
    )
    SEAWEEDFS_PPC64LE: str = (
        "icr.io/ppc64le-oss/seaweedfs-ppc64le"
        "@sha256:f0671dcb703b1301b546e4ab016496d818bdb1ae596dea91d135d9ce236eb698"  # pragma: allowlist secret
    )
    DISTRIBUTION_REGISTRY_PPC64LE: str = (
        "public.ecr.aws/docker/library/registry"
        "@sha256:d4032d4cc323c1731c707931ecada07a53076eac38e3b61ae57533f170f2c6c3"  # pragma: allowlist secret
    )
