"""Kuadrant custom resource for Kuadrant API management."""

from typing import Any

from ocp_resources.resource import NamespacedResource

from utilities.constants import ApiGroups


class Kuadrant(NamespacedResource):
    """Kuadrant is the Schema for the kuadrants API."""

    api_group: str = ApiGroups.KUADRANT_IO

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def to_dict(self) -> None:
        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            self.res["spec"] = {}
