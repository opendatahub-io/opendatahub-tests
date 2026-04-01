from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.scaled_object import ScaledObject


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
    return ScaledObject(client=client, name=f"{isvc.name}-predictor", namespace=isvc.namespace, ensure_exists=True)
