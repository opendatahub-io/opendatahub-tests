from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.inference_service import InferenceService

from utilities.constants import Timeout
from utilities.infra import get_model_route


def assert_ingress_status_changed(client: DynamicClient, inference_service: InferenceService) -> None:
    """
    Validates that the ingress status changes correctly after route deletion.

    Args:
        client (DynamicClient): The administrative client used to manage the model route.
        inference_service (InferenceService): The inference service whose route status is being checked.

    Raises:
        ResourceNotFoundError: If the route does not exist before or after deletion.
        AssertionError: If any of the validation checks fail.

    Returns:
        None
    """
    route = get_model_route(client=client, isvc=inference_service)
    if not route.exists:
        raise ResourceNotFoundError("Route before deletion not found: No active route is currently available.")

    initial_status = route.instance.status["ingress"][0]["conditions"][0]
    initial_host = route.host
    initial_transition_time = initial_status["lastTransitionTime"]
    initial_status_value = initial_status["status"]

    route.delete(wait=True, timeout=Timeout.TIMEOUT_1MIN)

    if not route.exists:
        raise ResourceNotFoundError("Route after deletion not found: No active route is currently available.")

    updated_status = route.instance.status["ingress"][0]["conditions"][0]
    updated_host = route.host
    updated_transition_time = updated_status["lastTransitionTime"]
    updated_status_value = updated_status["status"]

    # Collect failures instead of stopping at the first failed assertion
    failures = []

    if updated_host != initial_host:
        failures.append(f"Host mismatch: before={initial_host}, after={updated_host}")

    if updated_transition_time == initial_transition_time:
        failures.append(
            f"Transition time did not change: before={initial_transition_time}, after={updated_transition_time}"
        )

    if updated_status_value != "True":
        failures.append(f"Updated ingress status incorrect: expected=True, actual={updated_status_value}")

    if initial_status_value != "True":
        failures.append(f"Initial ingress status incorrect: expected=True, actual={initial_status_value}")

    # Assert all failures at once
    assert not failures, "Ingress status validation failed:\n" + "\n".join(failures)
