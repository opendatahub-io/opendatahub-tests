"""Pytest fixtures for NeMo Guardrails tests."""

from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.nemo_guardrails import NemoGuardrails
from ocp_resources.route import Route
from ocp_resources.secret import Secret

from tests.ai_safety.nemo_guardrails.constants import PresidioEntity
from tests.ai_safety.nemo_guardrails.utils import (
    create_llm_judge_config,
    create_presidio_config,
    wait_for_nemo_guardrails_health,
)
from utilities.constants import LLMdInferenceSimConfig


# ===========================
# Secret Fixtures
# ===========================
@pytest.fixture(scope="class")
def nemo_api_token_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[Secret, Any, Any]:
    """Create a secret containing API token for model access."""
    with Secret(
        client=admin_client,
        name="nemo-api-token",
        namespace=model_namespace.name,
        string_data={
            "token": "test-token-123",  # pragma: allowlist secret
        },
        type="Opaque",
    ) as secret:
        yield secret


# ===========================
# ConfigMap Fixtures
# ===========================
@pytest.fixture(scope="class")
def nemo_llm_judge_configmap(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    llm_d_inference_sim_isvc: InferenceService,
) -> Generator[ConfigMap, Any, Any]:
    """ConfigMap with LLM-as-a-judge configuration."""
    config_data = create_llm_judge_config(
        namespace=model_namespace.name,
        model_isvc_name=llm_d_inference_sim_isvc.name,
        model_name=LLMdInferenceSimConfig.model_name,
    )

    with ConfigMap(
        client=admin_client,
        name="nemo-llm-judge-config",
        namespace=model_namespace.name,
        data=config_data,
    ) as cm:
        yield cm


@pytest.fixture(scope="class")
def nemo_presidio_configmap(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    llm_d_inference_sim_isvc: InferenceService,
) -> Generator[ConfigMap, Any, Any]:
    """ConfigMap with Presidio PII detection configuration."""
    config_data = create_presidio_config(
        namespace=model_namespace.name,
        model_isvc_name=llm_d_inference_sim_isvc.name,
        model_name=LLMdInferenceSimConfig.model_name,
        input_entities=[
            PresidioEntity.EMAIL_ADDRESS,
            PresidioEntity.US_SSN,
            PresidioEntity.CREDIT_CARD,
        ],
        output_entities=[
            PresidioEntity.PERSON,
            PresidioEntity.EMAIL_ADDRESS,
        ],
    )

    with ConfigMap(
        client=admin_client,
        name="nemo-presidio-config",
        namespace=model_namespace.name,
        data=config_data,
    ) as cm:
        yield cm


@pytest.fixture(scope="class")
def nemo_multi_config_a(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    llm_d_inference_sim_isvc: InferenceService,
) -> Generator[ConfigMap, Any, Any]:
    """First ConfigMap for multi-configuration test."""
    config_data = create_llm_judge_config(
        namespace=model_namespace.name,
        model_isvc_name=llm_d_inference_sim_isvc.name,
        model_name=LLMdInferenceSimConfig.model_name,
    )

    with ConfigMap(
        client=admin_client,
        name="nemo-multi-config-a",
        namespace=model_namespace.name,
        data=config_data,
    ) as cm:
        yield cm


@pytest.fixture(scope="class")
def nemo_multi_config_b(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    llm_d_inference_sim_isvc: InferenceService,
) -> Generator[ConfigMap, Any, Any]:
    """Second ConfigMap for multi-configuration test."""
    config_data = create_presidio_config(
        namespace=model_namespace.name,
        model_isvc_name=llm_d_inference_sim_isvc.name,
        model_name=LLMdInferenceSimConfig.model_name,
        input_entities=[PresidioEntity.EMAIL_ADDRESS],
        output_entities=[PresidioEntity.PERSON],
    )

    with ConfigMap(
        client=admin_client,
        name="nemo-multi-config-b",
        namespace=model_namespace.name,
        data=config_data,
    ) as cm:
        yield cm


# ===========================
# NeMoGuardrails CR Fixtures
# ===========================
@pytest.fixture(scope="class")
def nemo_guardrails_llm_judge(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    nemo_llm_judge_configmap: ConfigMap,
    nemo_api_token_secret: Secret,
) -> Generator[NemoGuardrails, Any, Any]:
    """NeMo Guardrails CR with LLM-as-a-judge config and auth enabled."""
    with NemoGuardrails(
        client=admin_client,
        name="nemo-llm-judge",
        namespace=model_namespace.name,
        annotations={
            "security.opendatahub.io/enable-auth": "true",
        },
        nemo_configs=[
            {
                "name": "llm-judge",
                "configMaps": [nemo_llm_judge_configmap.name],
                "default": True,
            }
        ],
        replicas=1,
        env=[
            {
                "name": "OPENAI_API_KEY",
                "valueFrom": {
                    "secretKeyRef": {
                        "name": nemo_api_token_secret.name,
                        "key": "token",
                    }
                },
            }
        ],
    ) as nemo_cr:
        # Wait for the deployment to be ready
        deployment = Deployment(
            client=admin_client,
            name=nemo_cr.name,
            namespace=nemo_cr.namespace,
            wait_for_resource=True,
        )
        deployment.wait_for_replicas()
        yield nemo_cr


@pytest.fixture(scope="class")
def nemo_guardrails_presidio(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    nemo_presidio_configmap: ConfigMap,
    nemo_api_token_secret: Secret,
) -> Generator[NemoGuardrails, Any, Any]:
    """NeMo Guardrails CR with Presidio config and auth disabled."""
    with NemoGuardrails(
        client=admin_client,
        name="nemo-presidio",
        namespace=model_namespace.name,
        # Note: No security annotation means auth is disabled
        nemo_configs=[
            {
                "name": "presidio",
                "configMaps": [nemo_presidio_configmap.name],
                "default": True,
            }
        ],
        replicas=1,
        env=[
            {
                "name": "OPENAI_API_KEY",
                "valueFrom": {"secretKeyRef": {"name": nemo_api_token_secret.name, "key": "token"}},
            }
        ],
    ) as nemo_cr:
        # Wait for the deployment to be ready
        deployment = Deployment(
            client=admin_client,
            name=nemo_cr.name,
            namespace=nemo_cr.namespace,
            wait_for_resource=True,
        )
        deployment.wait_for_replicas()
        yield nemo_cr


@pytest.fixture(scope="class")
def nemo_guardrails_multi_config(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    nemo_multi_config_a: ConfigMap,
    nemo_multi_config_b: ConfigMap,
    nemo_api_token_secret: Secret,
) -> Generator[NemoGuardrails, Any, Any]:
    """NeMo Guardrails CR with multiple configurations."""
    with NemoGuardrails(
        client=admin_client,
        name="nemo-multi-config",
        namespace=model_namespace.name,
        nemo_configs=[
            {
                "name": "config-a",
                "configMaps": [nemo_multi_config_a.name],
                "default": True,
            },
            {
                "name": "config-b",
                "configMaps": [nemo_multi_config_b.name],
                "default": False,
            },
        ],
        replicas=1,
        env=[
            {
                "name": "OPENAI_API_KEY",
                "valueFrom": {"secretKeyRef": {"name": nemo_api_token_secret.name, "key": "token"}},
            }
        ],
    ) as nemo_cr:
        # Wait for the deployment to be ready
        deployment = Deployment(
            client=admin_client,
            name=nemo_cr.name,
            namespace=nemo_cr.namespace,
            wait_for_resource=True,
        )
        deployment.wait_for_replicas()
        yield nemo_cr


@pytest.fixture(scope="class")
def nemo_guardrails_second_server(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    nemo_presidio_configmap: ConfigMap,
    nemo_api_token_secret: Secret,
) -> Generator[NemoGuardrails, Any, Any]:
    """Second NeMo Guardrails server for multi-server test."""
    with NemoGuardrails(
        client=admin_client,
        name="nemo-second-server",
        namespace=model_namespace.name,
        nemo_configs=[
            {
                "name": "second-server-config",
                "configMaps": [nemo_presidio_configmap.name],
                "default": True,
            }
        ],
        replicas=1,
        env=[
            {
                "name": "OPENAI_API_KEY",
                "valueFrom": {"secretKeyRef": {"name": nemo_api_token_secret.name, "key": "token"}},
            }
        ],
    ) as nemo_cr:
        # Wait for the deployment to be ready
        deployment = Deployment(
            client=admin_client,
            name=nemo_cr.name,
            namespace=nemo_cr.namespace,
            wait_for_resource=True,
        )
        deployment.wait_for_replicas()
        yield nemo_cr


@pytest.fixture(scope="class")
def nemo_config_update_configmap(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    llm_d_inference_sim_isvc: InferenceService,
) -> Generator[ConfigMap, Any, Any]:
    """ConfigMap for config update test (will be modified during test)."""
    config_data = create_llm_judge_config(
        namespace=model_namespace.name,
        model_isvc_name=llm_d_inference_sim_isvc.name,
        model_name=LLMdInferenceSimConfig.model_name,
    )

    with ConfigMap(
        client=admin_client,
        name="nemo-config-update-test",
        namespace=model_namespace.name,
        data=config_data,
    ) as cm:
        yield cm


@pytest.fixture(scope="class")
def nemo_guardrails_config_update(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    nemo_config_update_configmap: ConfigMap,
    nemo_api_token_secret: Secret,
) -> Generator[NemoGuardrails, Any, Any]:
    """NeMo Guardrails CR for config update testing."""
    with NemoGuardrails(
        client=admin_client,
        name="nemo-config-update",
        namespace=model_namespace.name,
        nemo_configs=[
            {
                "name": "update-test",
                "configMaps": [nemo_config_update_configmap.name],
                "default": True,
            }
        ],
        replicas=1,
        env=[
            {
                "name": "OPENAI_API_KEY",
                "valueFrom": {"secretKeyRef": {"name": nemo_api_token_secret.name, "key": "token"}},
            }
        ],
    ) as nemo_cr:
        # Wait for the deployment to be ready
        deployment = Deployment(
            client=admin_client,
            name=nemo_cr.name,
            namespace=nemo_cr.namespace,
            wait_for_resource=True,
        )
        deployment.wait_for_replicas()
        yield nemo_cr


def create_nemo_guardrails_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    nemo_cr: NemoGuardrails,
) -> Route:
    return Route(
        client=admin_client,
        name=nemo_cr.name,
        namespace=model_namespace.name,
        wait_for_resource=True,
    )


@pytest.fixture(scope="class")
def nemo_guardrails_llm_judge_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    nemo_guardrails_llm_judge: NemoGuardrails,
) -> Generator[Route, Any, Any]:
    yield create_nemo_guardrails_route(
        admin_client=admin_client,
        model_namespace=model_namespace,
        nemo_cr=nemo_guardrails_llm_judge,
    )


@pytest.fixture(scope="class")
def nemo_guardrails_presidio_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    nemo_guardrails_presidio: NemoGuardrails,
) -> Generator[Route, Any, Any]:
    yield create_nemo_guardrails_route(
        admin_client=admin_client,
        model_namespace=model_namespace,
        nemo_cr=nemo_guardrails_presidio,
    )


@pytest.fixture(scope="class")
def nemo_guardrails_multi_config_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    nemo_guardrails_multi_config: NemoGuardrails,
) -> Generator[Route, Any, Any]:
    yield create_nemo_guardrails_route(
        admin_client=admin_client,
        model_namespace=model_namespace,
        nemo_cr=nemo_guardrails_multi_config,
    )


@pytest.fixture(scope="class")
def nemo_guardrails_second_server_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    nemo_guardrails_second_server: NemoGuardrails,
) -> Generator[Route, Any, Any]:
    yield create_nemo_guardrails_route(
        admin_client=admin_client,
        model_namespace=model_namespace,
        nemo_cr=nemo_guardrails_second_server,
    )


@pytest.fixture(scope="class")
def nemo_guardrails_config_update_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    nemo_guardrails_config_update: NemoGuardrails,
) -> Generator[Route, Any, Any]:
    yield create_nemo_guardrails_route(
        admin_client=admin_client,
        model_namespace=model_namespace,
        nemo_cr=nemo_guardrails_config_update,
    )


def verify_guardrails_healthcheck(
    route: Route,
    openshift_ca_bundle_file: str,
    token: str | None = None,
) -> None:
    wait_for_nemo_guardrails_health(
        host=route.host,
        token=token,
        ca_bundle_file=openshift_ca_bundle_file,
    )


@pytest.fixture(scope="class")
def nemo_guardrails_llm_judge_healthcheck(
    nemo_guardrails_llm_judge: NemoGuardrails,
    nemo_guardrails_llm_judge_route: Route,
    current_client_token: str,
    openshift_ca_bundle_file: str,
) -> None:
    verify_guardrails_healthcheck(
        route=nemo_guardrails_llm_judge_route,
        openshift_ca_bundle_file=openshift_ca_bundle_file,
        token=current_client_token,
    )


@pytest.fixture(scope="class")
def nemo_guardrails_presidio_healthcheck(
    nemo_guardrails_presidio: NemoGuardrails,
    nemo_guardrails_presidio_route: Route,
    openshift_ca_bundle_file: str,
) -> None:
    verify_guardrails_healthcheck(
        route=nemo_guardrails_presidio_route,
        openshift_ca_bundle_file=openshift_ca_bundle_file,
    )


@pytest.fixture(scope="class")
def nemo_guardrails_multi_config_healthcheck(
    nemo_guardrails_multi_config: NemoGuardrails,
    nemo_guardrails_multi_config_route: Route,
    openshift_ca_bundle_file: str,
) -> None:
    verify_guardrails_healthcheck(
        route=nemo_guardrails_multi_config_route,
        openshift_ca_bundle_file=openshift_ca_bundle_file,
    )


@pytest.fixture(scope="class")
def nemo_guardrails_second_server_healthcheck(
    nemo_guardrails_second_server: NemoGuardrails,
    nemo_guardrails_second_server_route: Route,
    openshift_ca_bundle_file: str,
) -> None:
    verify_guardrails_healthcheck(
        route=nemo_guardrails_second_server_route,
        openshift_ca_bundle_file=openshift_ca_bundle_file,
    )


@pytest.fixture(scope="class")
def nemo_guardrails_config_update_healthcheck(
    nemo_guardrails_config_update: NemoGuardrails,
    nemo_guardrails_config_update_route: Route,
    openshift_ca_bundle_file: str,
) -> None:
    verify_guardrails_healthcheck(
        route=nemo_guardrails_config_update_route,
        openshift_ca_bundle_file=openshift_ca_bundle_file,
    )
