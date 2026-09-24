from typing import Any

from ocp_resources.resource import MissingRequiredArgumentError, Resource


class WorkloadPriorityClass(Resource):
    """Kueue WorkloadPriorityClass resource.

    A cluster-scoped priority value that can be attached to a Kueue-managed Job
    via the ``kueue.x-k8s.io/priority-class`` label. Higher ``value`` wins during
    preemption. EvalHub-submitted jobs carry no priority class (priority 0), so a
    WorkloadPriorityClass is the way to create a higher-priority competitor.
    """

    api_group: str = "kueue.x-k8s.io"
    api_version: str = "kueue.x-k8s.io/v1beta2"

    def __init__(
        self,
        value: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            value (int): Integer priority value; higher values preempt lower ones.
        """
        super().__init__(**kwargs)

        self.value = value

    def to_dict(self) -> None:
        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            if self.value is None:
                raise MissingRequiredArgumentError(argument="value")

            self.res["value"] = self.value
