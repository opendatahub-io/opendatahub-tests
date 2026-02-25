# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md


from typing import Any

from ocp_resources.resource import NamespacedResource


class EvalHub(NamespacedResource):
    """
    EvalHub is the Schema for the evalhubs API
    """

    api_group: str = NamespacedResource.ApiGroup.TRUSTYAI_OPENDATAHUB_IO

    def __init__(
        self,
        env: list[Any] | None = None,
        replicas: int | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            env (list[Any]): Environment variables for the eval-hub container

            replicas (int): Number of replicas for the eval-hub deployment

        """
        super().__init__(**kwargs)

        self.env = env
        self.replicas = replicas

    def to_dict(self) -> None:

        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            self.res["spec"] = {}
            _spec = self.res["spec"]

            if self.env is not None:
                _spec["env"] = self.env

            if self.replicas is not None:
                _spec["replicas"] = self.replicas

    # End of generated code
