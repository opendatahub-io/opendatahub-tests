from typing import Any

from ocp_resources.exceptions import MissingRequiredArgumentError
from ocp_resources.resource import NamespacedResource

from utilities.constants import ApiGroups


class LegacyExternalModel(NamespacedResource):
    """Legacy ExternalModel CR served by maas-controller under maas.opendatahub.io."""

    api_group: str = ApiGroups.MAAS_IO
    kind: str = "ExternalModel"

    def __init__(
        self,
        provider: str | None = None,
        target_model: str | None = None,
        endpoint: str | None = None,
        credential_ref: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a legacy maas.opendatahub.io ExternalModel resource wrapper.

        Args:
            provider: Provider identifier (for example ``openai``).
            target_model: Upstream model name at the external provider.
            endpoint: External provider FQDN without scheme or path.
            credential_ref: Secret reference containing the provider API key.
        """
        super().__init__(**kwargs)

        self.provider = provider
        self.target_model = target_model
        self.endpoint = endpoint
        self.credential_ref = credential_ref

    def to_dict(self) -> None:
        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            if self.provider is None:
                raise MissingRequiredArgumentError(argument="self.provider")
            if self.target_model is None:
                raise MissingRequiredArgumentError(argument="self.target_model")
            if self.endpoint is None:
                raise MissingRequiredArgumentError(argument="self.endpoint")
            if self.credential_ref is None:
                raise MissingRequiredArgumentError(argument="self.credential_ref")

            self.res["spec"] = {
                "provider": self.provider,
                "targetModel": self.target_model,
                "endpoint": self.endpoint,
                "credentialRef": self.credential_ref,
            }
