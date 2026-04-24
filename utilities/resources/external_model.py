# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md


from typing import Any

from ocp_resources.exceptions import MissingRequiredArgumentError
from ocp_resources.resource import NamespacedResource


class ExternalModel(NamespacedResource):
    """
        ExternalModel is the Schema for the externalmodels API.
    It defines an external LLM provider (e.g., OpenAI, Anthropic) that can be
    referenced by MaaSModelRef resources.
    """

    api_group: str = NamespacedResource.ApiGroup.MAAS_OPENDATAHUB_IO

    def __init__(
        self,
        credential_ref: dict[str, Any] | None = None,
        endpoint: str | None = None,
        provider: str | None = None,
        target_model: str | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            credential_ref (dict[str, Any]): CredentialRef references a Kubernetes Secret containing the provider
              API key. The Secret must contain a data key "api-key" with the
              credential value.

            endpoint (str): Endpoint is the FQDN of the external provider (no scheme or path).
              e.g. "api.openai.com". This field is metadata for downstream
              consumers (e.g. BBR provider-resolver plugin) and is not used by
              the controller for endpoint derivation.

            provider (str): Provider identifies the API format and auth type for the external
              model. The allowed values are: "openai", "anthropic", "azure-
              openai", "vertex" and "bedrock-openai".

            target_model (str): TargetModel is the upstream model name at the external provider. e.g.
              "gpt-4o", "claude-sonnet-4-5-20241022".

        """
        super().__init__(**kwargs)

        self.credential_ref = credential_ref
        self.endpoint = endpoint
        self.provider = provider
        self.target_model = target_model

    def to_dict(self) -> None:

        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            if self.credential_ref is None:
                raise MissingRequiredArgumentError(argument="self.credential_ref")

            if self.endpoint is None:
                raise MissingRequiredArgumentError(argument="self.endpoint")

            if self.provider is None:
                raise MissingRequiredArgumentError(argument="self.provider")

            if self.target_model is None:
                raise MissingRequiredArgumentError(argument="self.target_model")

            self.res["spec"] = {}
            _spec = self.res["spec"]

            _spec["credentialRef"] = self.credential_ref
            _spec["endpoint"] = self.endpoint
            _spec["provider"] = self.provider
            _spec["targetModel"] = self.target_model

    # End of generated code
