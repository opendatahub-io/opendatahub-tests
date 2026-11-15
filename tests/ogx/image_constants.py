class OgxImages:
    """Container images used by the OGX test suite."""

    # Standalone Praxis gateway, deployed by the OGX -> Praxis migration tests.
    # TODO: replace with a Red Hat-built image once one ships the OpenAI store filters.
    PRAXIS_GATEWAY: str = (
        "ghcr.io/praxis-proxy/ai"
        "@sha256:550661eea7e392a9f4415b836e0b2393299f26856bc02cbbc8d3301d49b759f3"  # pragma: allowlist secret
    )

    # In-cluster mock upstream servers (stdlib-only Python HTTP servers, run
    # via ConfigMap-mounted scripts). A plain UBI python image is sufficient;
    # digest-pinned per the image policy (mutable tags are blocking). The
    # digest is the current multi-arch manifest list (amd64/arm64/ppc64le/
    # s390x) of ubi9/python-312:latest.
    MOCK_SERVER: str = (
        "registry.access.redhat.com/ubi9/python-312"
        "@sha256:5b184c2acfe9e2eb9df1a243da8e4a82fedd96e5e7f6648ec5512656177fa326"  # pragma: allowlist secret
    )
