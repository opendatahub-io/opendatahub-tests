from typing import Any

from ocp_resources.exceptions import MissingRequiredArgumentError
from ocp_resources.resource import Resource


class ClusterQueue(Resource):
    """
    ClusterQueue is the Schema for the clusterqueues API (kueue.x-k8s.io).
    """

    api_group: str = "kueue.x-k8s.io"

    def __init__(
        self,
        namespace_selector: dict[str, Any] | None = None,
        resource_groups: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            namespace_selector (dict[str, Any]): Selector for namespaces this queue serves.

            resource_groups (list[dict[str, Any]]): Required. Resource groups managed by this queue.
        """
        super().__init__(**kwargs)

        self.namespace_selector = namespace_selector
        self.resource_groups = resource_groups

    def to_dict(self) -> None:
        super().to_dict()
        if not self.kind_dict and not self.yaml_file:
            if not self.resource_groups:
                raise MissingRequiredArgumentError(argument="resource_groups")

            self.res["spec"] = {}
            _spec = self.res["spec"]

            if self.namespace_selector is not None:
                _spec["namespaceSelector"] = self.namespace_selector
            else:
                _spec["namespaceSelector"] = {}

            if self.resource_groups:
                _spec["resourceGroups"] = self.resource_groups
