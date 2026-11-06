# Generated manually for ODH platform Monitoring CR (services.platform.opendatahub.io/v1alpha1).

from typing import Any

from ocp_resources.resource import NamespacedResource


class Monitoring(NamespacedResource):
    """Monitoring is the namespace-scoped ODH observability operand (default-monitoring)."""

    api_group: str = "services.platform.opendatahub.io"
    api_version: str = "v1alpha1"
    kind: str = "Monitoring"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
