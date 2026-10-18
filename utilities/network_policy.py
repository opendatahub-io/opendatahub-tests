"""Shared readers for asserting on-cluster NetworkPolicy objects."""

from typing import Any

from ocp_resources.network_policy import NetworkPolicy


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
