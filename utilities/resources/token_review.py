# Generated manually for Kubernetes TokenReview authentication.k8s.io/v1.

from typing import Any

from ocp_resources.resource import Resource


class TokenReview(Resource):
    """TokenReview validates a bearer token and returns the authenticated user."""

    api_group: str = "authentication.k8s.io"
    api_version: str = "authentication.k8s.io/v1"
    kind: str = "TokenReview"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
