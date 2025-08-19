from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.scaled_object import ScaledObject
from kubernetes.dynamic.exceptions import ResourceNotFoundError


def get_isvc_keda_scaledobject(client: DynamicClient, isvc: InferenceService) -> ScaledObject:
    """
    Get KEDA ScaledObject resource associated with an InferenceService.

    Args:
        client (DynamicClient): OCP Client to use.
        isvc (InferenceService): InferenceService object.

    Returns:
        ScaledObject: The ScaledObject for the InferenceService

    Raises:
        ResourceNotFoundError: if the ScaledObject is not found.
    """
    namespace = isvc.namespace
    scaled_object_name = isvc.name + "-predictor"

    try:
        return ScaledObject(client=client, name=scaled_object_name, namespace=namespace, ensure_exists=True)
    except Exception as e:
        raise ResourceNotFoundError(f"{isvc.name} has no KEDA ScaledObjects: {e}")
