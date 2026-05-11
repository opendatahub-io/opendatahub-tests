# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md


from typing import Any

from ocp_resources.exceptions import MissingRequiredArgumentError
from ocp_resources.resource import Resource

from utilities.constants import ApiGroups


class WorkloadPriorityClass(Resource):
    """
    WorkloadPriorityClass is the Schema for the workloadPriorityClass API
    """

    api_group: str = ApiGroups.KUEUE_X_K8S_IO

    def __init__(
        self,
        description: str | None = None,
        value: int | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            description (str): description is an arbitrary string that usually provides guidelines on
              when this workloadPriorityClass should be used.

            value (int): value represents the integer value of this workloadPriorityClass. This
              is the actual priority that workloads receive when jobs have the
              name of this class in their workloadPriorityClass label. Changing
              the value of workloadPriorityClass doesn't affect the priority of
              workloads that were already created.

        """
        super().__init__(**kwargs)

        self.description = description
        self.value = value

    def to_dict(self) -> None:

        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            if self.value is None:
                raise MissingRequiredArgumentError(argument="self.value")

            self.res["value"] = self.value

            if self.description is not None:
                self.res["description"] = self.description

    # End of generated code
