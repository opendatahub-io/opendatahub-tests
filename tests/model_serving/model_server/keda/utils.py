from typing import Any
from kubernetes.dynamic import DynamicClient
from ocp_resources.resource import NamespacedResource
from ocp_resources.inference_service import InferenceService
from kubernetes.dynamic.exceptions import ResourceNotFoundError


class ScaledObject(NamespacedResource):
    """KEDA ScaledObject resource for openshift-python-wrapper."""

    api_group: str = "keda.sh"
    api_version: str = "v1alpha1"

    def __init__(self, **kwargs: Any):
        """
        Args:
            kwargs: Keyword arguments to pass to the ScaledObject constructor
        """
        super().__init__(**kwargs)

    def to_dict(self) -> None:
        super().to_dict()
        if not self.kind_dict and not self.yaml_file:
            self.res["spec"] = {}

    @property
    def spec(self):
        """
        Expose spec for backward compatibility with dynamic client usage.

        Returns:
            The spec section of the ScaledObject resource
        """
        return self.instance.spec if self.instance else None


def get_isvc_keda_scaledobject(client: DynamicClient, isvc: InferenceService) -> list[ScaledObject]:
    """
    Get KEDA ScaledObject resources associated with an InferenceService.

    Args:
        client (DynamicClient): OCP Client to use.
        isvc (InferenceService): InferenceService object.

    Returns:
        list[ScaledObject]: A list of all matching ScaledObjects

    Raises:
        ResourceNotFoundError: if no ScaledObjects are found.
    """
    namespace = isvc.namespace
    scaled_object_name = isvc.name + "-predictor"

    try:
        scaled_objects = list(ScaledObject.get(dyn_client=client, namespace=namespace, name=scaled_object_name))

        if scaled_objects:
            return scaled_objects
        else:
            raise ResourceNotFoundError(f"{isvc.name} has no KEDA ScaledObjects")
    except Exception as e:
        raise ResourceNotFoundError(f"{isvc.name} has no KEDA ScaledObjects: {e}")
