class AiGatewayImages:
    """Container images used by ai_gateway tests."""

    # v0.9.0 (not v0.8.x) because --mode echo with a non-HuggingFace model id needs no tokenizer render sidecar.
    LLMD_INFERENCE_SIM: str = (
        "ghcr.io/llm-d/llm-d-inference-sim@sha256:be957b008416a645f206532aad408b5ff29dc81c628b8486479e79d0c2b3801b"
    )
