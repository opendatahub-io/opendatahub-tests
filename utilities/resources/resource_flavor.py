from typing import Any

from ocp_resources.resource import Resource


class ResourceFlavor(Resource):
    """
    ResourceFlavor is the Schema for the resourceflavors API (kueue.x-k8s.io).
    """

    api_group: str = "kueue.x-k8s.io"

    def __init__(self, **kwargs: Any) -> None:
        r"""
        Args:
            kwargs: Keyword arguments passed to the Resource constructor.
        """
        super().__init__(**kwargs)

    def to_dict(self) -> None:
        super().to_dict()
        if not self.kind_dict and not self.yaml_file:
            self.res["spec"] = {}
