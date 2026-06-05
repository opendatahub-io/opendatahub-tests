# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md

from typing import Any

from ocp_resources.exceptions import MissingRequiredArgumentError
from ocp_resources.resource import NamespacedResource


class AIGateway(NamespacedResource):
    """
    AIGateway bootstraps one tenant slice: a dedicated Gateway, a tenant namespace,
    the MaaS tenant config object, and tenant-admin RBAC.
    """

    api_group: str = NamespacedResource.ApiGroup.MAAS_OPENDATAHUB_IO

    def __init__(
        self,
        domain: str | None = None,
        gateway: dict[str, Any] | None = None,
        oidc: dict[str, Any] | None = None,
        rbac: dict[str, Any] | None = None,
        tenant_namespace: dict[str, Any] | None = None,
        tls: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            domain (str): Domain is the tenant hostname used for data-plane routing. When set
              together with TLS, the controller creates an HTTPS listener on
              port 443. When set without TLS, the controller creates an HTTP
              listener on port 80. When omitted, the controller creates a
              default HTTP listener on port 80 without a hostname.

            gateway (dict[str, Any]): Gateway is the Gateway API template reconciled for this tenant.

            oidc (dict[str, Any]): OIDC contains non-MaaS-specific OIDC settings for this AI Gateway. The
              current controller mirrors this into the temporary Tenant config
              object until the MaaS config CR rename lands.

            rbac (dict[str, Any]): RBAC configures tenant-admin access to the tenant namespace and this
              AIGateway object.

            tenant_namespace (dict[str, Any]): TenantNamespace identifies the namespace where tenant administrators
              manage MaaS objects.

            tls (dict[str, Any]): TLS configures the TLS certificate for the tenant Gateway HTTPS
              listener. Only effective when Domain is also set.

        """
        super().__init__(**kwargs)

        self.domain = domain
        self.gateway = gateway
        self.oidc = oidc
        self.rbac = rbac
        self.tenant_namespace = tenant_namespace
        self.tls = tls

    def to_dict(self) -> None:
        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            if self.tenant_namespace is None:
                raise MissingRequiredArgumentError(argument="self.tenant_namespace")

            self.res["spec"] = {}
            _spec = self.res["spec"]

            _spec["tenantNamespace"] = self.tenant_namespace

            if self.domain is not None:
                _spec["domain"] = self.domain

            if self.gateway is not None:
                _spec["gateway"] = self.gateway

            if self.oidc is not None:
                _spec["oidc"] = self.oidc

            if self.rbac is not None:
                _spec["rbac"] = self.rbac

            if self.tls is not None:
                _spec["tls"] = self.tls
