"""LeaderWorkerSetOperator custom resource for OpenShift LWS operator."""

from typing import Any

from ocp_resources.resource import Resource


class LeaderWorkerSetOperator(Resource):
    """LeaderWorkerSetOperator is the Schema for the leaderworkersetoperators API."""

    api_group: str = "operator.openshift.io"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def to_dict(self) -> None:
        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            self.res["spec"] = {}
