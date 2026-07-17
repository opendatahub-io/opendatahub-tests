class SharedImages:
    """Shared container images used across multiple test components.

    Images used by only one component should go in that component's
    image_constants.py instead (e.g. tests/ai_safety/image_constants.py).
    """

    # --- Canary entries for CI validation (remove after workflow is verified) ---

    # IMG002: tag-only image in constants (should error)
    CANARY_TAG_ONLY: str = "quay.io/canary/tag-only:v1"

    # IMG002 suppressed: tag-only image, check suppressed
    CANARY_TAG_SUPPRESSED: str = "quay.io/canary/tag-suppressed:v1"  # noqa: IMG002

    # IMG003: DockerHub image in constants (should warn)
    CANARY_DOCKERHUB: str = (
        "docker.io/canary/constants-dockerhub"  # noqa: E501
        "@sha256:0000000000000000000000000000000000000000000000000000000000000000"
    )

    # IMG002 + IMG003: DockerHub image with tag in constants (should error + warn)
    CANARY_DOCKERHUB_TAG: str = "docker.io/canary/constants-dockerhub-tag:v1"

    # IMG002 suppressed + IMG003 suppressed: fully suppressed
    CANARY_BOTH_SUPPRESSED: str = "docker.io/canary/both-suppressed:v1"  # noqa: IMG002, IMG003
