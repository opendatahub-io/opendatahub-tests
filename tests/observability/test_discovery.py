from unittest.mock import Mock

import pytest

from tests.observability.discovery import ServiceEndpoint, discover_operand_namespace

pytestmark = pytest.mark.tier1


def test_service_endpoint_uses_discovered_namespace_and_port() -> None:
    """Given a discovered service identity, build an in-cluster URL without a hard-coded namespace."""
    endpoint = ServiceEndpoint(namespace="monitoring-a", name="tempo", port=8443)

    assert endpoint.base_url == "https://tempo.monitoring-a.svc:8443"
    assert endpoint.to_dict() == {
        "namespace": "monitoring-a",
        "name": "tempo",
        "port": 8443,
        "scheme": "https",
    }


def test_operand_namespace_discovery_requires_one_matching_operand() -> None:
    """Given multiple operand namespaces, reject ambiguous service discovery."""
    client = Mock()
    resource_api = Mock()
    resource_api.get.return_value = Mock(
        items=[
            Mock(metadata=Mock(namespace="monitoring-a")),
            Mock(metadata=Mock(namespace="monitoring-b")),
        ]
    )
    client.resources.get.return_value = resource_api

    with pytest.raises(LookupError, match="expected one Monitoring"):
        discover_operand_namespace(
            admin_client=client,
            api_version="monitoring.example/v1",
            kind="Monitoring",
        )
