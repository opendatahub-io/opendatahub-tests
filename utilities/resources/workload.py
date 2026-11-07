# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/class_generator/README.md


from typing import Any

from ocp_resources.resource import NamespacedResource


class Workload(NamespacedResource):
    """
    Workload is the Schema for the workloads API
    """

    api_group: str = "kueue.x-k8s.io"

    def __init__(
        self,
        active: bool | None = None,
        maximum_execution_time_seconds: int | None = None,
        pod_sets: list[Any] | None = None,
        preemption_gates: list[Any] | None = None,
        priority: int | None = None,
        priority_class_ref: dict[str, Any] | None = None,
        queue_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            active (bool): active determines if a workload can be admitted into a queue. Changing
              active from true to false will evict any running workloads.
              Possible values are:    - false: indicates that a workload should
              never be admitted and evicts running workloads   - true: indicates
              that a workload can be evaluated for admission into it's
              respective queue.  Defaults to true

            maximum_execution_time_seconds (int): maximumExecutionTimeSeconds if provided, determines the maximum time,
              in seconds, the workload can be admitted before it's automatically
              deactivated.  If unspecified, no execution time limit is enforced
              on the Workload.

            pod_sets (list[Any]): podSets is a list of sets of homogeneous pods, each described by a Pod
              spec and a count. There must be at least one element and at most
              10. podSets cannot be changed.

            preemption_gates (list[Any]): preemptionGates is a list of gates governing whether the workload can
              trigger preemptions. The gates are closed by default.

            priority (int): priority determines the order of access to the resources managed by
              the ClusterQueue where the workload is queued. The priority value
              is populated from the referenced PriorityClass (via
              priorityClassRef). The higher the value, the higher the priority.
              If priorityClassRef is specified, priority must not be null.

            priority_class_ref (dict[str, Any]): priorityClassRef references a PriorityClass object that defines the
              workload's priority.

            queue_name (str): queueName is the name of the LocalQueue the Workload is associated
              with. queueName cannot be changed while .status.admission is not
              null.

        """
        super().__init__(**kwargs)

        self.active = active
        self.maximum_execution_time_seconds = maximum_execution_time_seconds
        self.pod_sets = pod_sets
        self.preemption_gates = preemption_gates
        self.priority = priority
        self.priority_class_ref = priority_class_ref
        self.queue_name = queue_name

    def to_dict(self) -> None:

        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            self.res["spec"] = {}
            _spec = self.res["spec"]

            if self.active is not None:
                _spec["active"] = self.active

            if self.maximum_execution_time_seconds is not None:
                _spec["maximumExecutionTimeSeconds"] = self.maximum_execution_time_seconds

            if self.pod_sets is not None:
                _spec["podSets"] = self.pod_sets

            if self.preemption_gates is not None:
                _spec["preemptionGates"] = self.preemption_gates

            if self.priority is not None:
                _spec["priority"] = self.priority

            if self.priority_class_ref is not None:
                _spec["priorityClassRef"] = self.priority_class_ref

            if self.queue_name is not None:
                _spec["queueName"] = self.queue_name

    # End of generated code
