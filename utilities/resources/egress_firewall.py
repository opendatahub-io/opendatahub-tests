"""OVN-Kubernetes EgressFirewall resource wrapper."""

from typing import Any

from ocp_resources.resource import NamespacedResource


class EgressFirewall(NamespacedResource):
    """EgressFirewall restricts egress traffic leaving a namespace.

    Unlike NetworkPolicy, EgressFirewall rules are enforced by OVN-Kubernetes
    independently of other policies selecting the same pods, so they cannot be
    overridden by allow-all NetworkPolicies (e.g. the per-InferenceService
    policy created by odh-model-controller, which allows all egress).
    """

    api_group: str = NamespacedResource.ApiGroup.K8S_OVN_ORG

    def __init__(self, egress: list[dict[str, Any]], **kwargs: Any) -> None:
        """
        Args:
            egress: Ordered egress rules; each rule is a dict with a "type"
                ("Allow" or "Deny") and a "to" destination, e.g.
                {"type": "Deny", "to": {"cidrSelector": "0.0.0.0/0"}}.
        """
        super().__init__(**kwargs)
        self.egress = egress

    def to_dict(self) -> None:
        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            self.res["spec"] = {}
            self.res["spec"]["egress"] = self.egress
