# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md

from typing import Any

from ocp_resources.resource import NamespacedResource


class EnvoyFilter(NamespacedResource):
    """
    No field description from API
    """

    api_group: str = NamespacedResource.ApiGroup.NETWORKING_ISTIO_IO

    def __init__(
        self,
        config_patches: list[Any] | None = None,
        priority: int | None = None,
        workload_selector: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            config_patches (list[Any]): One or more patches with match conditions.

            priority (int): Priority defines the order in which patch sets are applied within a
              context.

            workload_selector (dict[str, Any]): No field description from API

        """
        super().__init__(**kwargs)

        self.config_patches = config_patches
        self.priority = priority
        self.workload_selector = workload_selector

    def to_dict(self) -> None:

        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            self.res["spec"] = {}
            _spec = self.res["spec"]

            if self.config_patches is not None:
                _spec["configPatches"] = self.config_patches

            if self.priority is not None:
                _spec["priority"] = self.priority

            if self.workload_selector is not None:
                _spec["workloadSelector"] = self.workload_selector

    # End of generated code
