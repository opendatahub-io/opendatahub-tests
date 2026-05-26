# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md


from typing import Any

from ocp_resources.resource import MissingRequiredArgumentError, NamespacedResource


class OgxServer(NamespacedResource):
    """
    OGXServer is the Schema for the ogxservers API.

    API reference: https://github.com/ogx-ai/ogx-k8s-operator/blob/main/api/v1beta1/ogxserver_types.go
    """

    api_group: str = "ogx.io"
    kind: str = "OGXServer"

    def __init__(
        self,
        distribution: dict[str, Any] | None = None,
        workload: dict[str, Any] | None = None,
        network: dict[str, Any] | None = None,
        tls: dict[str, Any] | None = None,
        override_config: dict[str, Any] | None = None,
        providers: dict[str, Any] | None = None,
        storage: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            distribution (dict[str, Any]): Identifies the OGX distribution to deploy.
                Must contain either ``name`` or ``image``.

            workload (dict[str, Any]): Kubernetes deployment settings including replicas,
                resources, storage, autoscaling, and pod-level overrides (env vars, volumes).

            network (dict[str, Any]): Network access controls including port, TLS termination,
                external access, and NetworkPolicy configuration.

            tls (dict[str, Any]): Outbound TLS trust anchors and client identity for
                connections to providers and backends.

            override_config (dict[str, Any]): References a ConfigMap key containing a full
                config.yaml override. Mutually exclusive with providers, resources, and storage.

            providers (dict[str, Any]): Provider configuration by API type.
                Mutually exclusive with override_config.

            storage (dict[str, Any]): State storage backends (KV and SQL).
                Mutually exclusive with override_config.

        """
        super().__init__(**kwargs)

        self.distribution = distribution
        self.workload = workload
        self.network = network
        self.tls = tls
        self.override_config = override_config
        self.providers = providers
        self.storage = storage

    def to_dict(self) -> None:
        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            if self.distribution is None:
                raise MissingRequiredArgumentError(argument="self.distribution")

            self.res["spec"] = {}
            _spec = self.res["spec"]

            _spec["distribution"] = self.distribution

            if self.workload is not None:
                _spec["workload"] = self.workload

            if self.network is not None:
                _spec["network"] = self.network

            if self.tls is not None:
                _spec["tls"] = self.tls

            if self.override_config is not None:
                _spec["overrideConfig"] = self.override_config

            if self.providers is not None:
                _spec["providers"] = self.providers

            if self.storage is not None:
                _spec["storage"] = self.storage

    # End of generated code
