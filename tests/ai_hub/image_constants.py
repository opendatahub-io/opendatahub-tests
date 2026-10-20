class AiHubImages:
    """Container images used by ai_hub tests."""

    MYSQL: str = (
        "public.ecr.aws/docker/library/mysql"
        "@sha256:28540698ce89bd72f985044de942d65bd99c6fadb2db105327db57f3f70564f0"  # pragma: allowlist secret
    )
    MYSQL_S390X: str = (
        "registry.redhat.io/rhel9/mysql-84"
        "@sha256:c16d572a6ff2ba6029a261ea4ba6342a14743f1e2615b23a32964a201bda9566"  # pragma: allowlist secret
    )
    BUSYBOX: str = (
        "public.ecr.aws/docker/library/busybox"
        "@sha256:1487d0af5f52b4ba31c7e465126ee2123fe3f2305d638e7827681e7cf6c83d5e"  # pragma: allowlist secret
    )
    POSTGRES: str = (
        "public.ecr.aws/docker/library/postgres"
        "@sha256:6e9bbed548cc1ca776dd4685cfea9efe60d58df91186ec6bad7328fd03b388a5"  # pragma: allowlist secret
    )
