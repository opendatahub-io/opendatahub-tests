from ocp_resources.resource import NamespacedResource

from tests.model_explainability.evalhub.constants import (
    EVALHUB_KIND,
    EVALHUB_PLURAL,
)


class EvalHub(NamespacedResource):
    """OCP resource wrapper for EvalHub CR."""

    api_group: str = NamespacedResource.ApiGroup.TRUSTYAI_OPENDATAHUB_IO
    kind = EVALHUB_KIND
    singular_name = "evalhub"
    plural_name = EVALHUB_PLURAL
