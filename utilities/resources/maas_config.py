# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md

from typing import Any

from ocp_resources.resource import Resource


class Config(Resource):
    """
    Config is a cluster-scoped anchor for MaaS platform resources. Namespaced and
    cluster-scoped operands created by maas-controller reference this object as their controller
    owner so Kubernetes garbage collection can tear down the full graph when the Config
    is deleted (subject to finalizers on dependents).
    """

    api_group: str = Resource.ApiGroup.MAAS_OPENDATAHUB_IO

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

    # End of generated code
