# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md


from typing import Any

from ocp_resources.resource import Resource


class AIGateway(Resource):
    """
    AIGateway is the Schema for the aigateways API.

    Cluster-scoped. Name must be default-aigateway.
    Source: opendatahub-io/ai-gateway-operator api/components/v1alpha1.
    """

    api_group: str = Resource.ApiGroup.COMPONENTS_PLATFORM_OPENDATAHUB_IO

    def __init__(
        self,
        batch_gateway: dict[str, Any] | None = None,
        models_as_a_service: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            batch_gateway (dict[str, Any]): BatchGateway controls the batch-gateway
              operator sub-component (e.g. {"managementState": "Managed"}).

            models_as_a_service (dict[str, Any]): ModelsAsAService controls the
              maas-controller sub-component (e.g. {"managementState": "Managed"}).

        """
        super().__init__(**kwargs)

        self.batch_gateway = batch_gateway
        self.models_as_a_service = models_as_a_service

    def to_dict(self) -> None:
        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            self.res["spec"] = {}
            _spec = self.res["spec"]

            if self.batch_gateway is not None:
                _spec["batchGateway"] = self.batch_gateway

            if self.models_as_a_service is not None:
                _spec["modelsAsAService"] = self.models_as_a_service

    # End of generated code
