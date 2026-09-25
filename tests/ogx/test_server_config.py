from typing import Any

from tests.ogx.server_config import build_ogx_server_config
from tests.ogx.utils import dummy_files_factory, dummy_vector_io_factory


def test_build_ogx_server_config_default() -> None:
    """Verify that default OGX server configuration omits network spec and sets default workload resources."""
    config = build_ogx_server_config(
        vector_io_provider_deployment_config_factory=dummy_vector_io_factory,
        files_provider_config_factory=dummy_files_factory,
        is_disconnected_cluster=False,
        params={},
    )

    assert config["distribution"] == {"name": "rh"}
    assert "network" not in config
    assert config["workload"]["resources"]["requests"] == {"cpu": "1", "memory": "1Gi"}
    assert config["workload"]["resources"]["limits"] == {"cpu": "2", "memory": "2Gi"}


def test_build_ogx_server_config_custom_network() -> None:
    """Verify that custom network specification in params is included in the returned configuration."""
    network_spec: dict[str, Any] = {"policy": {"enabled": True}}
    config = build_ogx_server_config(
        vector_io_provider_deployment_config_factory=dummy_vector_io_factory,
        files_provider_config_factory=dummy_files_factory,
        is_disconnected_cluster=False,
        params={"network": network_spec},
    )

    assert config["network"] == network_spec
