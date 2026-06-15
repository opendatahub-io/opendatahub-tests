from typing import Any

from ocp_resources.resource import NamespacedResource


class EnvoyFilter(NamespacedResource):
    """
    EnvoyFilter object.
    """

    api_group: str = NamespacedResource.ApiGroup.NETWORKING_ISTIO_IO

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def to_dict(self) -> None:
        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            self.res["spec"] = {}
