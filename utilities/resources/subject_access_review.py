"""SubjectAccessReview resource wrapper used by observability authorization checks."""

from typing import Any

from ocp_resources.resource import Resource


class SubjectAccessReview(Resource):
    """Create a Kubernetes authorization review through the wrapper resource API."""

    api_group: str = "authorization.k8s.io"
    api_version: str = "authorization.k8s.io/v1"

    def __init__(self, spec: dict[str, Any], name: str = "observability-subject-access-review", **kwargs: Any) -> None:
        super().__init__(
            name=name,
            kind_dict={
                "apiVersion": self.api_version,
                "kind": "SubjectAccessReview",
                "metadata": {"name": name},
                "spec": spec,
            },
            **kwargs,
        )
