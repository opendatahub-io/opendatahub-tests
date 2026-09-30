# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/class_generator/README.md


from typing import Any

from ocp_resources.resource import Resource


class SubjectAccessReview(Resource):
    """
    SubjectAccessReview checks whether or not a user or group can perform an action.
    """

    api_group: str = "authorization.k8s.io"
    api_version: str = "authorization.k8s.io/v1"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    # End of generated code
