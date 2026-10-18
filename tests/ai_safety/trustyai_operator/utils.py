from typing import Any

from ocp_resources.deployment import Deployment
from ocp_resources.network_policy import NetworkPolicy

from utilities.general import validate_image_format


def validate_trustyai_operator_image(
    related_images_refs: set[str], tai_operator_configmap_data: dict[str, str], tai_operator_deployment: Deployment
) -> None:
    """Validates the TrustyAI operator image.
    Checks if:
        - container image matches that of the operator configmap.
        - image is present in relatedImages of CSV.
        - image complies with OpenShift AI requirements i.e. sourced from registry.redhat.io and pinned w/o tags.

        Args:
            related_images_refs (set[str]): set of related image refs from the RHOAI CSV
            tai_operator_configmap_data (dict[str, str]): TrustyAI configmap data
            tai_operator_deployment (Deployment): TrustyAI deployment object

        Returns:
            None

        Raises:
            AssertionError: If any of the related images references are not present or invalid.
    """
    tai_operator_image = tai_operator_deployment.instance.spec.template.spec.containers[0].image
    assert tai_operator_image == tai_operator_configmap_data["trustyaiOperatorImage"]
    assert tai_operator_image in related_images_refs
    image_valid, error_message = validate_image_format(image=tai_operator_image)
    assert image_valid, error_message


def get_network_policy_pod_selector_labels(network_policy: NetworkPolicy) -> dict[str, str]:
    """Get the matchLabels of the pod selector that picks which pods a NetworkPolicy applies to.

    Args:
        network_policy: NetworkPolicy to read.

    Returns:
        The pod selector's matchLabels, or an empty dict if it selects every pod in the namespace.
    """
    return network_policy.instance.to_dict()["spec"].get("podSelector", {}).get("matchLabels", {})


def get_network_policy_policy_types(network_policy: NetworkPolicy) -> list[str]:
    """Get the traffic directions a NetworkPolicy enforces.

    Args:
        network_policy: NetworkPolicy to read.

    Returns:
        The policyTypes list, e.g. ["Ingress"] or ["Egress"].
    """
    return network_policy.instance.to_dict()["spec"].get("policyTypes", [])


def get_network_policy_ingress_rules(network_policy: NetworkPolicy) -> list[dict[str, Any]]:
    """Get the ingress rules of a NetworkPolicy as plain dictionaries.

    Args:
        network_policy: NetworkPolicy to read.

    Returns:
        List of ingress rules, each with its "from" peers and "ports".
    """
    return network_policy.instance.to_dict()["spec"].get("ingress", [])


def get_network_policy_egress_rules(network_policy: NetworkPolicy) -> list[dict[str, Any]]:
    """Get the egress rules of a NetworkPolicy as plain dictionaries.

    Args:
        network_policy: NetworkPolicy to read.

    Returns:
        List of egress rules, each with its "to" peers and "ports".
    """
    return network_policy.instance.to_dict()["spec"].get("egress", [])
