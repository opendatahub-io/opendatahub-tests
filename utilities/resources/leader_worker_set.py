# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/class_generator/README.md


from typing import Any

from ocp_resources.exceptions import MissingRequiredArgumentError
from ocp_resources.resource import NamespacedResource


class LeaderWorkerSet(NamespacedResource):
    """
    LeaderWorkerSet is the Schema for the leaderworkersets API
    """

    api_group: str = "leaderworkerset.x-k8s.io"

    def __init__(
        self,
        leader_worker_template: dict[str, Any] | None = None,
        network_config: dict[str, Any] | None = None,
        replicas: int | None = None,
        rollout_strategy: dict[str, Any] | None = None,
        startup_policy: str | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            leader_worker_template (dict[str, Any]): LeaderWorkerTemplate defines the template for leader/worker pods

            network_config (dict[str, Any]): NetworkConfig defines the network configuration of the group

            replicas (int): Number of leader-workers groups. A scale subresource is available to
              enable HPA. The selector for HPA will be that of the leader pod,
              and so practically HPA will be looking up the leader pod metrics.
              Note that the leader pod could aggregate metrics from the rest of
              the group and expose them as a summary custom metric representing
              the whole group. On scale down, the leader pod as well as the
              workers statefulset will be deleted. Default to 1.

            rollout_strategy (dict[str, Any]): RolloutStrategy defines the strategy that will be applied to update
              replicas when a revision is made to the leaderWorkerTemplate.

            startup_policy (str): StartupPolicy determines the startup policy for the worker
              statefulset.

        """
        super().__init__(**kwargs)

        self.leader_worker_template = leader_worker_template
        self.network_config = network_config
        self.replicas = replicas
        self.rollout_strategy = rollout_strategy
        self.startup_policy = startup_policy

    def to_dict(self) -> None:

        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            if self.leader_worker_template is None:
                raise MissingRequiredArgumentError(argument="self.leader_worker_template")

            self.res["spec"] = {}
            _spec = self.res["spec"]

            _spec["leaderWorkerTemplate"] = self.leader_worker_template

            if self.network_config is not None:
                _spec["networkConfig"] = self.network_config

            if self.replicas is not None:
                _spec["replicas"] = self.replicas

            if self.rollout_strategy is not None:
                _spec["rolloutStrategy"] = self.rollout_strategy

            if self.startup_policy is not None:
                _spec["startupPolicy"] = self.startup_policy

    # End of generated code
