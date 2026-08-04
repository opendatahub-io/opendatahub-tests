from typing import Any

from ocp_resources.exceptions import MissingRequiredArgumentError
from ocp_resources.resource import NamespacedResource


class LocalQueue(NamespacedResource):
    """
    LocalQueue is the Schema for the localqueues API (kueue.x-k8s.io).
    """

    api_group: str = "kueue.x-k8s.io"

    def __init__(
        self,
        cluster_queue: str,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            cluster_queue (str): Name of the ClusterQueue this LocalQueue points to.
        """
        super().__init__(**kwargs)

        self.cluster_queue = cluster_queue

    def to_dict(self) -> None:
        super().to_dict()
        if not self.kind_dict and not self.yaml_file:
            if not self.cluster_queue:
                raise MissingRequiredArgumentError(argument="cluster_queue")

            self.res["spec"] = {}
            _spec = self.res["spec"]
            _spec["clusterQueue"] = self.cluster_queue
