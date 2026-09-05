import json
import pathlib
from collections.abc import Generator
from typing import Any

import pytest
import structlog
from ocp_resources.route import Route
from openshell._proto import datamodel_pb2, openshell_pb2, sandbox_pb2
from openshell.sandbox import InferenceRouteClient, SandboxClient, SandboxSession, TlsConfig

from tests.openshell.constants import (
    OPENSHELL_BEARER_TOKEN,
    OPENSHELL_GATEWAY_URL,
    OPENSHELL_SANDBOX_OPENCODE_IMAGE,
    OPENSHELL_TLS_CA_PATH,
    OPENSHELL_TLS_CERT_PATH,
    OPENSHELL_TLS_KEY_PATH,
    OPENSHELL_VLLM_ENDPOINT,
    OPENSHELL_VLLM_MODEL,
    OPENSHELL_VLLM_PROVIDER,
    OPENSHELL_VLLM_TOKEN,
)
from utilities.general import generate_random_name

LOGGER = structlog.get_logger(name=__name__)

_REQUIRED_ENV_VARS: dict[str, str] = {
    "OPENSHELL_VLLM_MODEL": OPENSHELL_VLLM_MODEL,
    "OPENSHELL_VLLM_ENDPOINT": OPENSHELL_VLLM_ENDPOINT,
}


@pytest.fixture(scope="session")
def skip_if_missing_open_shell_config() -> None:
    """Skip tests that need vLLM env vars. Not autouse — install tests run without it."""
    missing = [name for name, value in _REQUIRED_ENV_VARS.items() if not value]
    if missing:
        pytest.skip(reason=f"Missing required env vars for open_shell tests: {', '.join(missing)}")


def _build_tls_config() -> TlsConfig | None:
    """Build a TlsConfig from env vars for the explicit gateway URL path.

    Supports three profiles matching the SDK's own trust hierarchy:
    - Full mTLS: OPENSHELL_TLS_CA_PATH + OPENSHELL_TLS_CERT_PATH + OPENSHELL_TLS_KEY_PATH
    - CA-only:   OPENSHELL_TLS_CA_PATH only (custom CA, no client identity)
    - System roots: no vars set — TlsConfig() uses the OS trust store
    """
    ca = pathlib.Path(OPENSHELL_TLS_CA_PATH) if OPENSHELL_TLS_CA_PATH else None
    cert = pathlib.Path(OPENSHELL_TLS_CERT_PATH) if OPENSHELL_TLS_CERT_PATH else None
    key = pathlib.Path(OPENSHELL_TLS_KEY_PATH) if OPENSHELL_TLS_KEY_PATH else None
    return TlsConfig(ca_path=ca, cert_path=cert, key_path=key)


def _opencode_config(model: str) -> dict:
    """Provide model metadata that OpenCode normally fetches from models.opencode.ai.

    The sandbox cannot reach models.opencode.ai (403), so we supply the
    provider + model definition explicitly.
    """
    return {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "rhoai": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "RHOAI",
                "options": {"baseURL": "https://inference.local/v1"},
                "models": {
                    model: {"name": model},
                },
            },
        },
        "model": f"rhoai/{model}",
    }


def _write_opencode_config(session: SandboxSession, model: str) -> None:
    """Write opencode.json into the sandbox so OpenCode can resolve the model."""
    LOGGER.info("Writing OpenCode config into sandbox")
    session.exec(["mkdir", "-p", "/sandbox/.config/opencode"], timeout_seconds=5)
    session.exec(
        [
            "sh",
            "-c",
            (
                "cat > /sandbox/.config/opencode/opencode.json << 'EOFCONFIG'\n"
                f"{json.dumps(_opencode_config(model), indent=2)}\nEOFCONFIG"
            ),
        ],
        timeout_seconds=5,
    )


def _network_policy_rules() -> list[openshell_pb2.PolicyMergeOperation]:
    """Build merge operations to allow sandbox egress to inference.local."""
    return [
        openshell_pb2.PolicyMergeOperation(
            add_rule=openshell_pb2.AddNetworkRule(
                rule_name="inference",
                rule=sandbox_pb2.NetworkPolicyRule(
                    name="inference",
                    endpoints=[sandbox_pb2.NetworkEndpoint(host="inference.local", port=443)],
                ),
            ),
        ),
    ]


@pytest.fixture(scope="session")
def sandbox_client(
    skip_if_missing_open_shell_config: None,
    installed_openshell_release: str,
    openshell_gateway_route: Route,
    openshell_tls_dir: pathlib.Path,
) -> Generator[SandboxClient, Any, Any]:
    """SandboxClient connected to the OpenShell gateway.

    Two paths, evaluated in order:

    1. Explicit URL: OPENSHELL_GATEWAY_URL env var — TLS from env vars.
    2. Helm install: uses the route host from the install fixture + mTLS
       material extracted from the K8s Secrets (server CA + client cert/key).
    """
    if OPENSHELL_GATEWAY_URL:
        LOGGER.info("Connecting to OpenShell gateway via explicit URL", url=OPENSHELL_GATEWAY_URL)
        client = SandboxClient(
            endpoint=OPENSHELL_GATEWAY_URL,
            tls=_build_tls_config(),
            bearer_token=OPENSHELL_BEARER_TOKEN or None,
        )
    else:
        endpoint = f"{installed_openshell_release}:443"
        LOGGER.info("Connecting to Helm-installed OpenShell gateway", endpoint=endpoint)
        client = SandboxClient(
            endpoint=endpoint,
            tls=TlsConfig(
                ca_path=openshell_tls_dir / "ca.crt",
                cert_path=openshell_tls_dir / "tls.crt",
                key_path=openshell_tls_dir / "tls.key",
            ),
        )

    with client:
        yield client


@pytest.fixture(scope="session")
def vllm_provider(
    sandbox_client: SandboxClient,
    teardown_resources: bool,
) -> Generator[str, Any, Any]:
    """Register the vLLM inference provider with the OpenShell gateway.

    Creates an OpenAI-compatible provider pointing at the vLLM endpoint so
    that ``inference_route`` can route sandbox inference requests through it.

    Note: uses ``client._stub.CreateProvider`` / ``DeleteProvider`` directly
    because the openshell SDK (v0.0.85) does not yet expose a high-level
    provider CRUD API. Replace with the public wrapper when available.
    """
    provider_name = OPENSHELL_VLLM_PROVIDER

    provider = datamodel_pb2.Provider(
        metadata=datamodel_pb2.ObjectMeta(name=provider_name),
        type="openai",
        config={"OPENAI_BASE_URL": OPENSHELL_VLLM_ENDPOINT},
        credentials={"OPENAI_API_KEY": OPENSHELL_VLLM_TOKEN},
    )

    LOGGER.info("Creating vLLM inference provider", name=provider_name, endpoint=OPENSHELL_VLLM_ENDPOINT)
    sandbox_client._stub.CreateProvider(  # noqa: FCN001
        openshell_pb2.CreateProviderRequest(provider=provider),
        timeout=sandbox_client._timeout,
    )

    yield provider_name

    if teardown_resources:
        LOGGER.info("Deleting vLLM inference provider", name=provider_name)
        sandbox_client._stub.DeleteProvider(  # noqa: FCN001
            openshell_pb2.DeleteProviderRequest(name=provider_name),
            timeout=sandbox_client._timeout,
        )


@pytest.fixture(scope="class")
def inference_route(sandbox_client: SandboxClient, vllm_provider: str) -> None:
    """Configures the vLLM inference route at cluster level."""
    LOGGER.info(
        "Setting vLLM inference route (cluster-level)", provider=OPENSHELL_VLLM_PROVIDER, model=OPENSHELL_VLLM_MODEL
    )
    route_client = InferenceRouteClient.from_sandbox_client(client=sandbox_client)
    route_client.set_cluster(
        provider_name=OPENSHELL_VLLM_PROVIDER,
        model_id=OPENSHELL_VLLM_MODEL,
        no_verify=True,
    )


@pytest.fixture(scope="class")
def sandbox(
    sandbox_client: SandboxClient,
    inference_route: None,
    teardown_resources: bool,
) -> Generator[SandboxSession, Any, Any]:
    """An OpenShell sandbox routed through the privacy router.

    Sets up the sandbox with:
    - Network policy rules for inference.local
    - The vLLM inference provider attached for credential injection
    - OPENAI_BASE_URL pointing to the privacy router
    """

    template_kwargs: dict = {
        "environment": {
            "OPENAI_BASE_URL": "https://inference.local/v1",
            "OPENAI_MODEL": OPENSHELL_VLLM_MODEL,
            "OPENAI_API_KEY": "unused",  # pragma: allowlist secret
        }
    }
    if OPENSHELL_SANDBOX_OPENCODE_IMAGE:
        LOGGER.info("Using custom sandbox image", image=OPENSHELL_SANDBOX_OPENCODE_IMAGE)
        template_kwargs["image"] = OPENSHELL_SANDBOX_OPENCODE_IMAGE

    spec = openshell_pb2.SandboxSpec(template=openshell_pb2.SandboxTemplate(**template_kwargs))
    sandbox_name = generate_random_name(prefix="open-shell")

    LOGGER.info("Creating OpenShell sandbox", name=sandbox_name)
    sandbox_ref = sandbox_client.create(spec=spec, name=sandbox_name)  # noqa: FCN001

    try:
        sandbox_client.wait_ready(sandbox_ref.name)  # noqa: FCN001

        # TODO(openshell-sdk): replace _stub.AttachSandboxProvider with public API
        # when the SDK exposes a high-level wrapper for provider attachment.
        LOGGER.info(
            "Attaching inference provider to sandbox", sandbox=sandbox_ref.name, provider=OPENSHELL_VLLM_PROVIDER
        )
        sandbox_client._stub.AttachSandboxProvider(  # noqa: FCN001
            openshell_pb2.AttachSandboxProviderRequest(
                sandbox_name=sandbox_ref.name,
                provider_name=OPENSHELL_VLLM_PROVIDER,
            ),
            timeout=sandbox_client._timeout,
        )

        # TODO(openshell-sdk): replace _stub.UpdateConfig with public API
        # when the SDK exposes a high-level wrapper for sandbox config updates.
        LOGGER.info("Adding network policy rules to sandbox", sandbox=sandbox_ref.name)
        sandbox_client._stub.UpdateConfig(  # noqa: FCN001
            openshell_pb2.UpdateConfigRequest(
                name=sandbox_ref.name,
                merge_operations=_network_policy_rules(),
            ),
            timeout=sandbox_client._timeout,
        )

        session = SandboxSession(sandbox_client, sandbox_ref)  # noqa: FCN001
        _write_opencode_config(session=session, model=OPENSHELL_VLLM_MODEL)

        yield session
    finally:
        if teardown_resources:
            LOGGER.info("Deleting OpenShell sandbox", name=sandbox_ref.name)
            sandbox_client.delete(sandbox_ref.name)
            sandbox_client.wait_deleted(sandbox_ref.name)
