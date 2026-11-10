"""Gen AI Studio Playground configuration test.

Starts OGX with the configuration the odh-dashboard generates for a Gen AI
Studio Playground (full config.yaml override whose LLM inference path is the
``genai-bff-proxy`` ``remote::passthrough`` provider) and verifies the server
comes up with that provider registered.

Regression guard: upstream OGX 1.3.0 removed ``remote::passthrough``, which
broke the Playground configuration without any downstream test failing.
A config containing ``remote::passthrough`` fails to start on
such an image, so this test fails before release.

Marker requirements: ``tier1`` and
``install``, plus ``pre_upgrade`` and ``post_upgrade`` — the single-test /
all-lanes pattern used by ``tests/spark/test_spark.py``: the test runs once in
install runs and in both legs of upgrade runs.
"""

import pytest
from ogx_client import OgxClient

from tests.ogx.constants import (
    GENAI_PLAYGROUND_PASSTHROUGH_PROVIDER_ID,
    GENAI_PLAYGROUND_PASSTHROUGH_PROVIDER_TYPE,
)


def _assert_playground_passthrough_provider(ogx_client: OgxClient) -> None:
    """Assert the dashboard's genai-bff-proxy remote::passthrough provider is registered."""
    providers = ogx_client.providers.list()
    passthrough = [
        provider for provider in providers if provider.provider_id == GENAI_PLAYGROUND_PASSTHROUGH_PROVIDER_ID
    ]
    assert passthrough, (
        f"Provider {GENAI_PLAYGROUND_PASSTHROUGH_PROVIDER_ID!r} is not registered; "
        f"registered providers: {[provider.provider_id for provider in providers]}"
    )
    assert passthrough[0].provider_type == GENAI_PLAYGROUND_PASSTHROUGH_PROVIDER_TYPE, (
        f"Expected provider type {GENAI_PLAYGROUND_PASSTHROUGH_PROVIDER_TYPE!r} for "
        f"{GENAI_PLAYGROUND_PASSTHROUGH_PROVIDER_ID!r}, got {passthrough[0].provider_type!r}"
    )


@pytest.mark.ogx
@pytest.mark.tier1
@pytest.mark.install
@pytest.mark.pre_upgrade
@pytest.mark.post_upgrade
@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [
        pytest.param(
            {"name": "test-ogx-genai-playground"},
        ),
    ],
    indirect=True,
)
def test_ogx_genai_playground_config_ready(ogx_client: OgxClient) -> None:
    """Verify OGX starts with the Gen AI Studio Playground configuration.

    Given: An OGXServer deployed the way the odh-dashboard deploys it for a
        Playground (config ConfigMap + spec.overrideConfig, with the
        genai-bff-proxy remote::passthrough inference provider).
    When: The server is queried for health, version and registered providers.
    Then: The server responds and the genai-bff-proxy remote::passthrough
        provider is registered — i.e. the exact configuration class that broke
        in OGX 1.3.0 still starts.
    """
    # GET /v1/health — raises on an unhealthy/unresponsive server.
    ogx_client.inspect.health()

    version = ogx_client.inspect.version()
    assert version.version, f"OGX server reported an empty version: {version}"

    _assert_playground_passthrough_provider(ogx_client=ogx_client)
