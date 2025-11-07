from ocp_resources.config_map import ConfigMap
from ocp_resources.inference_service import InferenceService


def validate_metrics_configuration(model_car_inference_service: InferenceService) -> None:
    """
    Validate that the InferenceService has proper metrics configuration.

    Checks:
    - Metrics dashboard ConfigMap has supported=true

    Args:
        model_car_inference_service: InferenceService object

    Raises:
        AssertionError: If validation fails
    """
    metrics_cm_name = f"{model_car_inference_service.name}-metrics-dashboard"
    metrics_cm = ConfigMap(
        client=model_car_inference_service.client,
        name=metrics_cm_name,
        namespace=model_car_inference_service.namespace,
    )

    assert metrics_cm.exists, (
        f"Metrics dashboard ConfigMap '{metrics_cm_name}' not found in namespace "
        f"'{model_car_inference_service.namespace}'"
    )

    supported_value = metrics_cm.instance.data.get("supported")

    assert supported_value == "true", (
        f"Metrics dashboard ConfigMap '{metrics_cm_name}' has 'supported: {supported_value}'. "
        f"Expected 'supported: true' for metrics to be available. "
    )
