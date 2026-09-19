# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md

from typing import Any

from ocp_resources.resource import Resource


class TrustyAI(Resource):
    """
    TrustyAI is the Schema for the trustyais API.

    Cluster-scoped. Name must be default-trustyai.
    Source: trustyai-explainability/trustyai-service-operator trustyai-operator-module.
    """

    api_group: str = Resource.ApiGroup.COMPONENTS_PLATFORM_OPENDATAHUB_IO

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

    # End of generated code
