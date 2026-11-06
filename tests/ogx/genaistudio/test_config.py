"""Unit tests for the Gen AI Studio Playground configuration builders (RHAIENG-7804).

Pure config-level tests — no cluster required. They lock down the properties the
Playground deployment depends on, most importantly the ``genai-bff-proxy``
``remote::passthrough`` provider whose upstream removal (OGX 1.3.0,
RHAIENG-7281) broke the Playground configuration without any downstream test
failing.
"""

import pytest

from tests.ogx.constants import (
    GENAI_PLAYGROUND_CONFIGMAP_KEY,
    GENAI_PLAYGROUND_CONFIGMAP_NAME,
    GENAI_PLAYGROUND_FORWARD_HEADERS,
    GENAI_PLAYGROUND_PASSTHROUGH_PROVIDER_ID,
    GENAI_PLAYGROUND_PASSTHROUGH_PROVIDER_TYPE,
)
from tests.ogx.genaistudio.config import (
    build_genai_playground_config,
    build_genai_playground_server_config,
    get_genai_playground_configmap_spec,
)


def _get_inference_provider(config: dict, provider_type: str) -> dict:
    return [provider for provider in config["providers"]["inference"] if provider["provider_type"] == provider_type]


@pytest.mark.ogx
def test_build_genai_playground_config_passthrough_provider() -> None:
    """Verify the dashboard's genai-bff-proxy remote::passthrough provider is present."""
    config = build_genai_playground_config()

    passthrough = _get_inference_provider(config=config, provider_type=GENAI_PLAYGROUND_PASSTHROUGH_PROVIDER_TYPE)
    assert len(passthrough) == 1, "Expected exactly one remote::passthrough inference provider"
    provider = passthrough[0]
    assert provider["provider_id"] == GENAI_PLAYGROUND_PASSTHROUGH_PROVIDER_ID
    # Credentials flow per-request (X-OGX-Provider-Data), not via a static key.
    assert provider["config"]["api_key"] == ""
    # The per-request credential flow the dashboard relies on.
    assert provider["config"]["forward_headers"] == GENAI_PLAYGROUND_FORWARD_HEADERS
    assert "maas_subscription" in provider["config"]["forward_headers"]


@pytest.mark.ogx
def test_build_genai_playground_config_base_url_contains_namespace() -> None:
    """Verify the passthrough base_url embeds the namespace like the dashboard BFF proxy URL."""
    config = build_genai_playground_config(namespace="test-ogx-genai-playground")

    provider = _get_inference_provider(config=config, provider_type=GENAI_PLAYGROUND_PASSTHROUGH_PROVIDER_TYPE)[0]
    assert provider["config"]["base_url"].endswith("/ns/test-ogx-genai-playground")

    custom_url = "http://mock-proxy.test-ns.svc.cluster.local:9999/genai-proxy/ns/test-ns"
    config = build_genai_playground_config(passthrough_base_url=custom_url)
    provider = _get_inference_provider(config=config, provider_type=GENAI_PLAYGROUND_PASSTHROUGH_PROVIDER_TYPE)[0]
    assert provider["config"]["base_url"] == custom_url


@pytest.mark.ogx
def test_build_genai_playground_config_dashboard_api_set() -> None:
    """Verify the v3.6.0 dashboard API set and provider coverage."""
    config = build_genai_playground_config()

    assert config["version"] == "2"
    assert config["distro_name"] == "rh"
    assert config["apis"] == [
        "responses",
        "file_processors",
        "files",
        "inference",
        "tool_runtime",
        "vector_io",
    ]
    provider_types = {provider["provider_type"] for provider in config["providers"]["inference"]}
    assert "inline::sentence-transformers" in provider_types
    for api, expected_providers in {
        "responses": "inline::builtin",
        "file_processors": "inline::auto",
        "files": "inline::localfs",
        "tool_runtime": "inline::file-search",
    }.items():
        assert expected_providers in {provider["provider_type"] for provider in config["providers"][api]}, (
            f"Missing {expected_providers} under {api}"
        )


@pytest.mark.ogx
def test_build_genai_playground_config_registered_resources() -> None:
    """Verify the default embedding model is registered (no LLM models, like the dashboard)."""
    config = build_genai_playground_config()

    models = config["registered_resources"]["models"]
    assert len(models) == 1
    embedding = models[0]
    assert embedding["model_type"] == "embedding"
    assert embedding["provider_id"] == "sentence-transformers"
    assert embedding["metadata"]["embedding_dimension"] == 768
    # VectorIO starts empty (pgvector is only added when PostgreSQL is provisioned).
    assert config["vector_stores"]["default_provider_id"] == ""
    assert config["registered_resources"]["vector_stores"] == []


@pytest.mark.ogx
def test_build_genai_playground_server_config_override() -> None:
    """Verify the OGXServer spec uses the dashboard-equivalent overrideConfig reference."""
    spec = build_genai_playground_server_config()

    assert spec["distribution"] == {"name": "rh"}
    assert spec["overrideConfig"] == {"name": GENAI_PLAYGROUND_CONFIGMAP_NAME, "key": GENAI_PLAYGROUND_CONFIGMAP_KEY}
    # overrideConfig is mutually exclusive with providers/resources/storage; the
    # spec must not set them.
    for exclusive_key in ("providers", "resources", "storage"):
        assert exclusive_key not in spec
    assert "policy" in spec["network"]
    assert spec["workload"]["resources"]["requests"]["cpu"] == "1"


@pytest.mark.ogx
def test_get_genai_playground_configmap_spec() -> None:
    """Verify the ConfigMap spec carries the config.yaml and the operator watch label."""
    configmap_spec = get_genai_playground_configmap_spec(namespace="test-ogx-genai-playground")

    assert configmap_spec["name"] == GENAI_PLAYGROUND_CONFIGMAP_NAME
    # Required by the OGX operator for ConfigMaps referenced via spec.overrideConfig.
    assert configmap_spec["labels"]["ogx.io/watch"] == "true"
    assert GENAI_PLAYGROUND_CONFIGMAP_KEY in configmap_spec["data"]
    config = configmap_spec["data"][GENAI_PLAYGROUND_CONFIGMAP_KEY]
    assert _get_inference_provider(config=config, provider_type=GENAI_PLAYGROUND_PASSTHROUGH_PROVIDER_TYPE)
