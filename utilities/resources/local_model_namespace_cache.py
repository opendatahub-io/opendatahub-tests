# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md


from typing import Any

from ocp_resources.exceptions import MissingRequiredArgumentError
from ocp_resources.resource import NamespacedResource


class LocalModelNamespaceCache(NamespacedResource):
    """Namespace-scoped ``LocalModelNamespaceCache`` CR (KServe ``serving.kserve.io/v1alpha1``)."""

    api_group: str = NamespacedResource.ApiGroup.SERVING_KSERVE_IO

    def __init__(
        self,
        image_pull_secrets: list[str] | None = None,
        model_size: Any | None = None,
        node_groups: list[Any] | None = None,
        pvc_ref: str | None = None,
        service_account_name: str | None = None,
        source_model_uri: str | None = None,
        storage: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            image_pull_secrets (list[str]): Registry pull secret names for OCI imports

            model_size (Any): No field description from API

            node_groups (list[Any]): No field description from API

            pvc_ref (str): Existing shared PVC used for namespace cache storage

            service_account_name (str): No field description from API

            source_model_uri (str): No field description from API

            storage (dict[str, Any]): No field description from API

        """
        super().__init__(**kwargs)

        self.image_pull_secrets = image_pull_secrets
        self.model_size = model_size
        self.node_groups = node_groups
        self.pvc_ref = pvc_ref
        self.service_account_name = service_account_name
        self.source_model_uri = source_model_uri
        self.storage = storage

    def to_dict(self) -> None:

        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            if self.model_size is None:
                raise MissingRequiredArgumentError(argument="self.model_size")

            if self.node_groups is None and self.pvc_ref is None:
                raise MissingRequiredArgumentError(argument="self.node_groups or self.pvc_ref")
            if self.node_groups is not None and self.pvc_ref is not None:
                raise ValueError("node_groups and pvc_ref are mutually exclusive")

            if self.source_model_uri is None:
                raise MissingRequiredArgumentError(argument="self.source_model_uri")

            self.res["spec"] = {}
            _spec = self.res["spec"]

            _spec["modelSize"] = self.model_size
            _spec["sourceModelUri"] = self.source_model_uri

            if self.image_pull_secrets is not None:
                _spec["imagePullSecrets"] = [{"name": name} for name in self.image_pull_secrets]

            if self.node_groups is not None:
                _spec["nodeGroups"] = self.node_groups

            if self.pvc_ref is not None:
                _spec["pvcRef"] = self.pvc_ref

            if self.service_account_name is not None:
                _spec["serviceAccountName"] = self.service_account_name

            if self.storage is not None:
                _spec["storage"] = self.storage

    # End of generated code
