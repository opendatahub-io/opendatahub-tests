# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md

from typing import Any

from ocp_resources.resource import NamespacedResource


class AIGateway(NamespacedResource):
    """
    AIGateway is the infra-level bootstrap CR for multi-tenant MaaS environments.
    """

    api_group: str = NamespacedResource.ApiGroup.MAAS_OPENDATAHUB_IO

    def __init__(
        self,
        tenant_namespace: dict[str, Any] | None = None,
        gateway: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            tenant_namespace (dict[str, Any]): Tenant namespace provisioning policy.
            gateway (dict[str, Any]): Gateway template for the tenant Gateway API resource.
        """
        super().__init__(**kwargs)

        self.tenant_namespace = tenant_namespace
        self.gateway = gateway

    def to_dict(self) -> None:
        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            self.res["spec"] = {}
            _spec = self.res["spec"]

            if self.tenant_namespace is not None:
                _spec["tenantNamespace"] = self.tenant_namespace

            if self.gateway is not None:
                _spec["gateway"] = self.gateway
