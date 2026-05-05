from typing import Any

from ocp_resources.exceptions import MissingRequiredArgumentError
from ocp_resources.resource import NamespacedResource


class TempoStack(NamespacedResource):
    """
    TempoStack manages a Tempo deployment in microservices mode.
    """

    api_group: str = NamespacedResource.ApiGroup.TEMPO_GRAFANA_COM

    def __init__(
        self,
        management_state: str | None = None,
        resources: dict[str, dict[str, dict[str, str]]] | None = None,
        storage: dict[str, dict[str, str]] | None = None,
        storage_size: str | None = None,
        template: dict[str, dict[str, dict[str, bool]]] | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            management_state (str): ManagementState defines if the CR should be managed by the operator or
              not. Default is managed.

            resources (dict[str, dict[str, dict[str, str]]]): Resources defines resources configuration.

            storage (dict[str, dict[str, str]]): Storage defines the spec for the object storage endpoint to store
              traces. User is required to create secret and supply it.

            storage_size (str): StorageSize for PVCs used by ingester. Defaults to 10Gi.

            template (dict[str, dict[str, dict[str, bool]]]):
                Template defines requirements for a set of tempo components.
        """
        super().__init__(**kwargs)

        self.management_state = management_state
        self.resources = resources
        self.storage = storage
        self.storage_size = storage_size
        self.template = template

    def to_dict(self) -> None:
        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            if self.management_state is None:
                raise MissingRequiredArgumentError(argument="self.management_state")

            if self.storage is None:
                raise MissingRequiredArgumentError(argument="self.storage")

            self.res["spec"] = {}
            _spec = self.res["spec"]

            _spec["managementState"] = self.management_state
            _spec["storage"] = self.storage

            if self.resources is not None:
                _spec["resources"] = self.resources

            if self.storage_size is not None:
                _spec["storageSize"] = self.storage_size

            if self.template is not None:
                _spec["template"] = self.template
