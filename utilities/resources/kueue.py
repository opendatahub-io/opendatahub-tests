from typing import Any

from ocp_resources.resource import Resource


class Kueue(Resource):
    """
    Kueue is the Schema for the kueues API (kueue.openshift.io).

    This is the CR exposed by the Red Hat build of Kueue operator.
    """

    api_group: str = "kueue.openshift.io"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        management_state: str | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            config (dict[str, Any]): Kueue controller configuration (e.g. framework integrations).

            management_state (str): managementState for the Kueue controller.
        """
        super().__init__(**kwargs)

        self.config = config
        self.management_state = management_state

    def to_dict(self) -> None:
        super().to_dict()
        if not self.kind_dict and not self.yaml_file:
            self.res["spec"] = {}
            _spec = self.res["spec"]

            if self.config is not None:
                _spec["config"] = self.config

            if self.management_state is not None:
                _spec["managementState"] = self.management_state
