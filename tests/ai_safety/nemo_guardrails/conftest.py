"""Pytest fixtures for NeMo Guardrails tests."""

from collections.abc import Generator
from typing import Any

import pytest
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.nemo_guardrails import NemoGuardrails
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount

from tests.ai_safety.nemo_guardrails.constants import (
    NEMO_DEFAULT_CONFIG_CM_PII,
    NEMO_DEFAULT_CONFIG_CM_PREFIX,
    PresidioEntity,
)
from tests.ai_safety.nemo_guardrails.utils import (
    create_llm_judge_config,
    create_presidio_config,
    wait_for_nemo_guardrails_health,
)
from utilities.constants import LLMdInferenceSimConfig
from utilities.resources.route import Route as UtilitiesRoute


def _resource_name(resource: Namespace | InferenceService) -> str:
    """Return a Kubernetes resource name, asserting the wrapper populated it."""
    name = resource.name
    assert name is not None, "Resource must have a name"
    return name


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
        namespace=_resource_name(model_namespace),
        model_isvc_name=_resource_name(llm_d_inference_sim_isvc),
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
        namespace=_resource_name(model_namespace),
        model_isvc_name=_resource_name(llm_d_inference_sim_isvc),
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
        namespace=_resource_name(model_namespace),
        model_isvc_name=_resource_name(llm_d_inference_sim_isvc),
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
        namespace=_resource_name(model_namespace),
        model_isvc_name=_resource_name(llm_d_inference_sim_isvc),
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
def nemo_guardrails_external_caller_namespace(admin_client: DynamicClient) -> Generator[Namespace, Any, Any]:
    """Namespace used as the source for cross-namespace NeMo Guardrails requests."""
    with Namespace(client=admin_client, name="external-namespace-caller") as namespace:
        namespace.wait_for_status(status=Namespace.Status.ACTIVE)
        yield namespace


@pytest.fixture(scope="class")
def nemo_guardrails_external_caller_service_account(
    admin_client: DynamicClient,
    nemo_guardrails_external_caller_namespace: Namespace,
) -> Generator[ServiceAccount, Any, Any]:
    """ServiceAccount used by the cross-namespace admin caller pod."""
    with ServiceAccount(
        client=admin_client,
        name="nemo-admin-caller",
        namespace=nemo_guardrails_external_caller_namespace.name,
        wait_for_resource=True,
    ) as service_account:
        yield service_account


@pytest.fixture(scope="class")
def nemo_guardrails_external_caller_admin_role(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[Role, Any, Any]:
    """Allow the external caller to create NemoGuardrails resources for admin-listener authorization."""
    with Role(
        client=admin_client,
        name="nemo-admin-caller-create-guardrails",
        namespace=model_namespace.name,
        rules=[
            {
                "apiGroups": ["trustyai.opendatahub.io"],
                "resources": ["nemoguardrails"],
                "resourceNames": ["nemo-presidio-auth-cross-namespace"],
                "verbs": ["create"],
            }
        ],
        wait_for_resource=True,
    ) as role:
        yield role


@pytest.fixture(scope="class")
def nemo_guardrails_external_caller_admin_role_binding(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    nemo_guardrails_external_caller_service_account: ServiceAccount,
    nemo_guardrails_external_caller_admin_role: Role,
) -> Generator[RoleBinding, Any, Any]:
    """Bind the external caller ServiceAccount to admin-listener create permission."""
    with RoleBinding(
        client=admin_client,
        name="nemo-admin-caller-create-guardrails",
        namespace=model_namespace.name,
        subjects_kind="ServiceAccount",
        subjects_name=nemo_guardrails_external_caller_service_account.name,
        subjects_namespace=nemo_guardrails_external_caller_service_account.namespace,
        role_ref_kind="Role",
        role_ref_name=nemo_guardrails_external_caller_admin_role.name,
        wait_for_resource=True,
    ) as role_binding:
        yield role_binding


@pytest.fixture(scope="class")
def nemo_guardrails_presidio_auth(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    nemo_presidio_configmap: ConfigMap,
    nemo_api_token_secret: Secret,
) -> Generator[NemoGuardrails, Any, Any]:
    """NeMo Guardrails CR with Presidio config and auth enabled."""
    with NemoGuardrails(
        client=admin_client,
        name="nemo-presidio-auth",
        namespace=model_namespace.name,
        annotations={
            "security.opendatahub.io/enable-auth": "true",
        },
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
        deployment = Deployment(
            client=admin_client,
            name=nemo_cr.name,
            namespace=nemo_cr.namespace,
            wait_for_resource=True,
        )
        deployment.wait_for_replicas()
        yield nemo_cr


@pytest.fixture(scope="class")
def nemo_guardrails_presidio_auth_cross_namespace(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    nemo_presidio_configmap: ConfigMap,
    nemo_api_token_secret: Secret,
    nemo_guardrails_external_caller_namespace: Namespace,
) -> Generator[NemoGuardrails, Any, Any]:
    """Auth-enabled Presidio CR allowing the external caller namespace to reach public and admin listeners."""
    cr_name = "nemo-presidio-auth-cross-namespace"
    resource_dict = {
        "apiVersion": f"{NemoGuardrails.api_group}/v1alpha1",
        "kind": "NemoGuardrails",
        "metadata": {
            "name": cr_name,
            "namespace": model_namespace.name,
            "annotations": {"security.opendatahub.io/enable-auth": "true"},
        },
        "spec": {
            "nemoConfigs": [
                {
                    "name": "presidio",
                    "configMaps": [nemo_presidio_configmap.name],
                    "default": True,
                }
            ],
            "replicas": 1,
            "env": [
                {
                    "name": "OPENAI_API_KEY",
                    "valueFrom": {"secretKeyRef": {"name": nemo_api_token_secret.name, "key": "token"}},
                }
            ],
            "networkPolicy": {
                "additionalIngress": [
                    {
                        "namespaceSelector": {
                            "matchLabels": {
                                "kubernetes.io/metadata.name": nemo_guardrails_external_caller_namespace.name
                            }
                        },
                        "podSelector": {"matchLabels": {"app": "external-namespace-caller-pod"}},
                        "listeners": ["Public", "Admin"],
                    }
                ]
            },
        },
    }
    with NemoGuardrails(
        client=admin_client,
        name=cr_name,
        namespace=model_namespace.name,
        kind_dict=resource_dict,
    ) as nemo_cr:
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
        namespace=_resource_name(model_namespace),
        model_isvc_name=_resource_name(llm_d_inference_sim_isvc),
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
def nemo_guardrails_presidio_auth_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    nemo_guardrails_presidio_auth: NemoGuardrails,
) -> Generator[Route, Any, Any]:
    """Route for the auth-enabled Presidio NeMo Guardrails CR."""
    yield create_nemo_guardrails_route(
        admin_client=admin_client,
        model_namespace=model_namespace,
        nemo_cr=nemo_guardrails_presidio_auth,
    )


@pytest.fixture(scope="class")
def nemo_guardrails_presidio_admin_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    nemo_guardrails_presidio_auth: NemoGuardrails,
) -> Generator[Route, Any, Any]:
    """Route targeting the admin proxy port (444) for the auth-enabled Presidio CR."""
    with UtilitiesRoute(
        client=admin_client,
        name=f"{nemo_guardrails_presidio_auth.name}-admin",
        namespace=model_namespace.name,
        to={
            "kind": "Service",
            "name": nemo_guardrails_presidio_auth.name,
            "weight": 100,
        },
        port={"targetPort": f"{nemo_guardrails_presidio_auth.name}-admin"},
        tls={"termination": "reencrypt"},
        wait_for_resource=True,
    ) as route:
        yield route


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


@pytest.fixture(scope="class")
def nemo_guardrails_default_config(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    nemo_api_token_secret: Secret,
) -> Generator[NemoGuardrails, Any, Any]:
    """NeMo Guardrails CR referencing the operator-shipped default PII configmap."""
    with NemoGuardrails(
        client=admin_client,
        name="nemo-default-config",
        namespace=model_namespace.name,
        nemo_configs=[
            {
                "name": "default-pii",
                "configMaps": [NEMO_DEFAULT_CONFIG_CM_PII],
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
        deployment = Deployment(
            client=admin_client,
            name=nemo_cr.name,
            namespace=nemo_cr.namespace,
            wait_for_resource=True,
        )
        deployment.wait_for_replicas()
        yield nemo_cr


@pytest.fixture(scope="class")
def nemo_default_fallback_configmap(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[ConfigMap, Any, Any]:
    """A default-prefixed configmap that exists only in the user namespace (not the operator namespace).

    This exercises the fallback path in mountNemoConfigs where the operator namespace lookup
    fails and the reconciler falls back to the CR namespace.
    """
    minimal_config = yaml.dump({
        "passthrough": True,
        "rails": {"input": {"flows": []}, "output": {"flows": []}},
    })
    # Name carries the default prefix so the controller attempts the operator-NS lookup first.
    cm_name = f"{NEMO_DEFAULT_CONFIG_CM_PREFIX}-custom-test"
    with ConfigMap(
        client=admin_client,
        name=cm_name,
        namespace=model_namespace.name,
        data={"config.yaml": minimal_config, "rails.co": ""},
    ) as cm:
        yield cm


@pytest.fixture(scope="class")
def nemo_guardrails_default_config_fallback(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    nemo_default_fallback_configmap: ConfigMap,
    nemo_api_token_secret: Secret,
) -> Generator[NemoGuardrails, Any, Any]:
    """NeMo Guardrails CR that uses a default-prefixed CM present only in the user namespace."""
    with NemoGuardrails(
        client=admin_client,
        name="nemo-default-fallback",
        namespace=model_namespace.name,
        nemo_configs=[
            {
                "name": "fallback-config",
                "configMaps": [nemo_default_fallback_configmap.name],
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
        deployment = Deployment(
            client=admin_client,
            name=nemo_cr.name,
            namespace=nemo_cr.namespace,
            wait_for_resource=True,
        )
        deployment.wait_for_replicas()
        yield nemo_cr


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
def nemo_guardrails_presidio_auth_healthcheck(
    nemo_guardrails_presidio_auth: NemoGuardrails,
    nemo_guardrails_presidio_auth_route: Route,
    current_client_token: str,
    openshift_ca_bundle_file: str,
) -> None:
    """Healthcheck for the auth-enabled Presidio NeMo Guardrails CR."""
    verify_guardrails_healthcheck(
        route=nemo_guardrails_presidio_auth_route,
        openshift_ca_bundle_file=openshift_ca_bundle_file,
        token=current_client_token,
    )


@pytest.fixture(scope="class")
def nemo_guardrails_presidio_admin_route_healthcheck(
    nemo_guardrails_presidio_auth: NemoGuardrails,
    nemo_guardrails_presidio_admin_route: Route,
    current_client_token: str,
    openshift_ca_bundle_file: str,
) -> None:
    """Healthcheck for the admin route of the auth-enabled Presidio NeMo Guardrails CR."""
    wait_for_nemo_guardrails_health(
        host=nemo_guardrails_presidio_admin_route.instance.spec.host,
        token=current_client_token,
        ca_bundle_file=openshift_ca_bundle_file,
        health_endpoint="/healthz",
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
