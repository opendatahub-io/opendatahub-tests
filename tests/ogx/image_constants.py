class OgxImages:
    """Container images used by the OGX test suite."""

    # Standalone Praxis gateway, deployed by the OGX -> Praxis migration tests.
    # TODO: replace with a Red Hat-built image once one ships the OpenAI store filters.
    PRAXIS_GATEWAY: str = "ghcr.io/praxis-proxy/ai@sha256:550661eea7e392a9f4415b836e0b2393299f26856bc02cbbc8d3301d49b759f3"  # pragma: allowlist secret
