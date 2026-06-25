# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md

from typing import Any

from ocp_resources.resource import NamespacedResource


class AITenant(NamespacedResource):
    """
        AITenant bootstraps one tenant slice: a tenant namespace, an existing
    network-admin-provisioned Gateway reference, the MaaS tenant config object,
    and tenant-admin RBAC.
    """

    api_group: str = NamespacedResource.ApiGroup.MAAS_OPENDATAHUB_IO

    def __init__(
        self,
        gateway: dict[str, Any] | None = None,
        oidc: dict[str, Any] | None = None,
        rbac: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            gateway (dict[str, Any]): Gateway references the network-admin-provisioned Gateway API Gateway
              for this tenant.

            oidc (dict[str, Any]): OIDC contains non-MaaS-specific OIDC settings for this AI Gateway
              tenant. The controller mirrors this into the temporary Tenant
              config object until the MaaS config CR rename lands.

            rbac (dict[str, Any]): RBAC configures tenant-admin access to the tenant namespace and this
              AITenant object.

        """
        super().__init__(**kwargs)

        self.gateway = gateway
        self.oidc = oidc
        self.rbac = rbac

    def to_dict(self) -> None:
        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            self.res["spec"] = {}
            _spec = self.res["spec"]

            if self.gateway is not None:
                _spec["gateway"] = self.gateway

            if self.oidc is not None:
                _spec["oidc"] = self.oidc

            if self.rbac is not None:
                _spec["rbac"] = self.rbac

    # End of generated code
