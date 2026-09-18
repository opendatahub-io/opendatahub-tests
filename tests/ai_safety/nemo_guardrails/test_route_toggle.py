"""Tests for NeMo Guardrails route exposure toggle (spec.exposeRoute).

Covers:
  - CRs created with exposeRoute=true create a Route.
  - CRs created with exposeRoute=false do not create a Route.
  - CRs created without exposeRoute (field omitted, i.e. pre-update CRs) default to true
    and therefore do have a Route — confirming the operator default is not changed by this feature.
  - Patching exposeRoute from true → false causes the operator to delete the existing Route.
  - Patching exposeRoute from false → true causes the operator to create a Route.
  - The RouteDisabled condition is set when exposeRoute=false.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.nemo_guardrails import NemoGuardrails
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from timeout_sampler import TimeoutSampler

from tests.ai_safety.nemo_guardrails.constants import NEMO_DEFAULT_CONFIG_CM_PII

_TIMEOUT = 120
_SLEEP = 5

_NEMO_CONFIGS = [{"name": "route-toggle-pii", "configMaps": [NEMO_DEFAULT_CONFIG_CM_PII], "default": True}]


def _api_key_env(secret_name: str) -> list[dict]:
    return [
        {
            "name": "OPENAI_API_KEY",
            "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "token"}},
        }
    ]


def _route_exists(client: DynamicClient, name: str, namespace: str) -> bool:
    return bool(Route(client=client, name=name, namespace=namespace).exists)


def _wait_for_route(client: DynamicClient, name: str, namespace: str, *, present: bool) -> None:
    for sample in TimeoutSampler(
        wait_timeout=_TIMEOUT,
        sleep=_SLEEP,
        func=lambda: _route_exists(client, name, namespace),
    ):
        if sample == present:
            break


def _condition_reason(nemo_cr: NemoGuardrails, condition_type: str) -> str | None:
    conditions = (nemo_cr.instance.status or {}).get("conditions", [])
    for cond in conditions:
        if cond.get("type") == condition_type:
            return cond.get("reason")
    return None


@pytest.mark.tier2
@pytest.mark.ai_safety
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-nemo-guardrails"})],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
class TestNemoGuardrailsRouteToggle:
    """Structural tests for the exposeRoute field."""

    def test_expose_route_true_creates_route(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_api_token_secret: Secret,
    ) -> None:
        """A CR created with exposeRoute=true gets a Route from the operator.

        Given: NemoGuardrails CR with spec.exposeRoute=true
        When: Reconciliation runs
        Then: A Route named after the CR exists in the CR namespace
        """
        with NemoGuardrails(
            client=admin_client,
            name="nemo-route-enabled",
            namespace=model_namespace.name,
            expose_route=True,
            nemo_configs=_NEMO_CONFIGS,
            replicas=1,
            env=_api_key_env(nemo_api_token_secret.name),
        ) as nemo_cr:
            Deployment(
                client=admin_client,
                name=nemo_cr.name,
                namespace=nemo_cr.namespace,
                wait_for_resource=True,
            ).wait_for_replicas()

            assert _route_exists(admin_client, nemo_cr.name, model_namespace.name), (
                f"Expected Route '{nemo_cr.name}' to exist in '{model_namespace.name}' when exposeRoute=true"
            )

    def test_expose_route_false_does_not_create_route(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_api_token_secret: Secret,
    ) -> None:
        """A CR created with exposeRoute=false does not get a Route from the operator.

        Given: NemoGuardrails CR with spec.exposeRoute=false
        When: Reconciliation runs
        Then: No Route named after the CR exists in the CR namespace
        """
        with NemoGuardrails(
            client=admin_client,
            name="nemo-route-disabled",
            namespace=model_namespace.name,
            expose_route=False,
            nemo_configs=_NEMO_CONFIGS,
            replicas=1,
            env=_api_key_env(nemo_api_token_secret.name),
        ) as nemo_cr:
            Deployment(
                client=admin_client,
                name=nemo_cr.name,
                namespace=nemo_cr.namespace,
                wait_for_resource=True,
            ).wait_for_replicas()

            assert not _route_exists(admin_client, nemo_cr.name, model_namespace.name), (
                f"Expected no Route '{nemo_cr.name}' in '{model_namespace.name}' when exposeRoute=false"
            )

    def test_omitted_expose_route_defaults_to_route_created(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_api_token_secret: Secret,
    ) -> None:
        """A CR that omits exposeRoute (pre-update behaviour) still gets a Route.

        The operator CRD sets +kubebuilder:default=true, so a CR that was created before
        this field existed — and therefore has no value stored — should have the field
        defaulted to true by the API server.

        Given: NemoGuardrails CR with no exposeRoute field set
        When: Reconciliation runs
        Then: A Route named after the CR exists in the CR namespace
        """
        with NemoGuardrails(
            client=admin_client,
            name="nemo-route-default",
            namespace=model_namespace.name,
            nemo_configs=_NEMO_CONFIGS,
            replicas=1,
            env=_api_key_env(nemo_api_token_secret.name),
        ) as nemo_cr:
            Deployment(
                client=admin_client,
                name=nemo_cr.name,
                namespace=nemo_cr.namespace,
                wait_for_resource=True,
            ).wait_for_replicas()

            assert _route_exists(admin_client, nemo_cr.name, model_namespace.name), (
                f"Expected Route '{nemo_cr.name}' to exist in '{model_namespace.name}' "
                "when exposeRoute is omitted (defaulted to true)"
            )

    def test_patch_expose_route_true_to_false_deletes_route(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_api_token_secret: Secret,
    ) -> None:
        """Patching exposeRoute from true to false causes the operator to delete the Route.

        Given: NemoGuardrails CR with spec.exposeRoute=true (Route is created)
        When: spec.exposeRoute is patched to false
        Then: The Route is deleted within the reconciliation timeout
        """
        with NemoGuardrails(
            client=admin_client,
            name="nemo-route-patch-disable",
            namespace=model_namespace.name,
            expose_route=True,
            nemo_configs=_NEMO_CONFIGS,
            replicas=1,
            env=_api_key_env(nemo_api_token_secret.name),
        ) as nemo_cr:
            Deployment(
                client=admin_client,
                name=nemo_cr.name,
                namespace=nemo_cr.namespace,
                wait_for_resource=True,
            ).wait_for_replicas()

            assert _route_exists(admin_client, nemo_cr.name, model_namespace.name), (
                f"Pre-condition failed: Route '{nemo_cr.name}' not found before patch"
            )

            nemo_cr.update(
                resource_dict={
                    "metadata": {"name": nemo_cr.name},
                    "spec": {"exposeRoute": False},
                }
            )

            _wait_for_route(client=admin_client, name=nemo_cr.name, namespace=model_namespace.name, present=False)

            assert not _route_exists(admin_client, nemo_cr.name, model_namespace.name), (
                f"Expected Route '{nemo_cr.name}' to be deleted after patching exposeRoute to false"
            )

    def test_patch_expose_route_false_to_true_creates_route(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_api_token_secret: Secret,
    ) -> None:
        """Patching exposeRoute from false to true causes the operator to create a Route.

        Given: NemoGuardrails CR with spec.exposeRoute=false (no Route)
        When: spec.exposeRoute is patched to true
        Then: A Route is created within the reconciliation timeout
        """
        with NemoGuardrails(
            client=admin_client,
            name="nemo-route-patch-enable",
            namespace=model_namespace.name,
            expose_route=False,
            nemo_configs=_NEMO_CONFIGS,
            replicas=1,
            env=_api_key_env(nemo_api_token_secret.name),
        ) as nemo_cr:
            Deployment(
                client=admin_client,
                name=nemo_cr.name,
                namespace=nemo_cr.namespace,
                wait_for_resource=True,
            ).wait_for_replicas()

            assert not _route_exists(admin_client, nemo_cr.name, model_namespace.name), (
                f"Pre-condition failed: Route '{nemo_cr.name}' unexpectedly exists before patch"
            )

            nemo_cr.update(
                resource_dict={
                    "metadata": {"name": nemo_cr.name},
                    "spec": {"exposeRoute": True},
                }
            )

            _wait_for_route(client=admin_client, name=nemo_cr.name, namespace=model_namespace.name, present=True)

            assert _route_exists(admin_client, nemo_cr.name, model_namespace.name), (
                f"Expected Route '{nemo_cr.name}' to be created after patching exposeRoute to true"
            )

    def test_route_disabled_condition_set_when_expose_route_false(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_api_token_secret: Secret,
    ) -> None:
        """The operator sets a RouteDisabled condition on the CR when exposeRoute=false.

        Given: NemoGuardrails CR with spec.exposeRoute=false
        When: Reconciliation runs to completion
        Then: The CR status has a Route condition with reason RouteDisabled
        """
        with NemoGuardrails(
            client=admin_client,
            name="nemo-route-condition",
            namespace=model_namespace.name,
            expose_route=False,
            nemo_configs=_NEMO_CONFIGS,
            replicas=1,
            env=_api_key_env(nemo_api_token_secret.name),
        ) as nemo_cr:
            Deployment(
                client=admin_client,
                name=nemo_cr.name,
                namespace=nemo_cr.namespace,
                wait_for_resource=True,
            ).wait_for_replicas()

            for sample in TimeoutSampler(
                wait_timeout=_TIMEOUT,
                sleep=_SLEEP,
                func=lambda: _condition_reason(nemo_cr=nemo_cr, condition_type="RouteReady"),
            ):
                if sample is not None:
                    break

            reason = _condition_reason(nemo_cr=nemo_cr, condition_type="RouteReady")
            assert reason == "RouteDisabled", f"Expected Route condition reason 'RouteDisabled', got: {reason!r}"
