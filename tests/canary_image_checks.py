"""Canary file to verify PR Container Image Checks workflow.

Each variable below triggers (or suppresses) a specific image check rule.
Remove this file once the CI workflow is validated.
"""

# IMG001: stray image not in any constants class (should warn)
CANARY_STRAY: str = "quay.io/canary/stray-image:v1"

# IMG001 suppressed: stray image, check suppressed
CANARY_STRAY_SUPPRESSED: str = "quay.io/canary/stray-suppressed:v1"  # noqa: IMG001

# IMG003: DockerHub image in test code (should warn)
CANARY_DOCKERHUB: str = (
    "docker.io/canary/dockerhub-image@sha256:0000000000000000000000000000000000000000000000000000000000000000"
)

# IMG003 suppressed: DockerHub image, check suppressed
CANARY_DOCKERHUB_SUPPRESSED: str = (
    "docker.io/canary/dh-sup@sha256:0000000000000000000000000000000000000000000000000000000000000000"  # noqa: IMG003, E501
)

# IMG001 + IMG003: stray DockerHub image (should warn for both)
CANARY_STRAY_DOCKERHUB: str = "docker.io/canary/stray-dockerhub:v1"

# IMG001 suppressed + IMG003: stray suppressed but DockerHub still warns
CANARY_STRAY_SUPPRESSED_DOCKERHUB: str = "docker.io/canary/stray-suppressed-dockerhub:v1"  # noqa: IMG001

# IMG001 + IMG003 both suppressed: fully suppressed
CANARY_FULLY_SUPPRESSED: str = "docker.io/canary/fully-suppressed:v1"  # noqa: IMG001, IMG003
