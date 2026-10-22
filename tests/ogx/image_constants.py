class OgxImages:
    """Container images used by the OGX test suite."""

    # Standalone Praxis gateway, deployed by the OGX -> Praxis migration tests.
    # TODO: replace with a Red Hat-built image once one ships the OpenAI store filters.
    PRAXIS_GATEWAY: str = "ghcr.io/praxis-proxy/ai:0.7.0"
