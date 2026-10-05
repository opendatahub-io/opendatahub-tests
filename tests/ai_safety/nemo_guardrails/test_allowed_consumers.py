"""Tests for the NeMo Guardrails allowedConsumers field (spec.allowedConsumers).

The allowedConsumers field controls which K8s/Openshift namespaces are allowed to attach
an AIGuardrail consumer to a NeMo Guardrails server. The operator supports three modes:
  - Same: only the namespace that owns the server may attach (default when field is omitted)
  - All: any namespace may attach
  - Selector: only namespaces whose labels match the given selector may attach

Covers:
  - Omitting allowedConsumers deploys the server successfully (Same is the implicit default).
  - Setting from=Same, from=All, or from=Selector with a valid selector marks the server ready.
  - Setting from=Selector with a malformed label key stops reconciliation and puts the server in an error state.
"""

from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from timeout_sampler import TimeoutSampler

from tests.ai_safety.nemo_guardrails.constants import NEMO_DEFAULT_CONFIG_CM_PII
from tests.ai_safety.nemo_guardrails.utils import (
    build_api_key_env,
    condition_status,
    nemo_cr_phase,
)
from utilities.resources.nemo_guardrails import NemoGuardrails

_TIMEOUT = 120
_SLEEP = 5

_NEMO_CONFIGS = [{"name": "allowed-consumers-pii", "configMaps": [NEMO_DEFAULT_CONFIG_CM_PII], "default": True}]


@pytest.mark.tier2
@pytest.mark.ai_safety
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-nemo-guardrails"})],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
class TestNemoGuardrailsAllowedConsumers:
    """Tests that the operator correctly enforces the allowedConsumers access policy.

    Each test creates its own NemoGuardrails server, waits for the operator to reconcile,
    checks the resulting status, then deletes the server.
    """

    def test_allowed_consumers_omitted_reconciles_successfully(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_api_token_secret: Secret,
    ) -> None:
        """A server with no allowedConsumers field starts up successfully with same-namespace access by default.

        Given: A NemoGuardrails server created without the allowedConsumers field
        When: The operator reconciles it
        Then: The server deployment becomes ready with same-namespace access.
        """
        with NemoGuardrails(
            client=admin_client,
            name="nemo-consumers-omitted",
            namespace=model_namespace.name,
            nemo_configs=_NEMO_CONFIGS,
            replicas=1,
            env=build_api_key_env(nemo_api_token_secret.name),
        ) as nemo_cr:
            Deployment(
                client=admin_client,
                name=nemo_cr.name,
                namespace=nemo_cr.namespace,
                wait_for_resource=True,
            ).wait_for_replicas()

    @pytest.mark.parametrize(
        "cr_name, allowed_consumers",
        [
            pytest.param(
                "nemo-consumers-same",
                {"namespaces": {"from": "Same"}},
                id="test_from_same",
            ),
            pytest.param(
                "nemo-consumers-all",
                {"namespaces": {"from": "All"}},
                id="test_from_all",
            ),
            pytest.param(
                "nemo-consumers-selector",
                {
                    "namespaces": {
                        "from": "Selector",
                        "selector": {
                            "matchLabels": {
                                "kubernetes.io/metadata.name": "test-nemo-guardrails",
                            }
                        },
                    }
                },
                id="test_from_selector_valid",
            ),
        ],
    )
    def test_allowed_consumers_reconciles_successfully(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_api_token_secret: Secret,
        cr_name: str,
        allowed_consumers: dict[str, Any],
    ) -> None:
        """A server with a valid allowedConsumers setting reports itself as ready.

        Given: A NemoGuardrails server with allowedConsumers set to Same, All, or Selector
        When: The operator reconciles it
        Then: The AllowedConsumersReady condition is True and the deployment is ready
        """
        with NemoGuardrails(
            client=admin_client,
            name=cr_name,
            namespace=model_namespace.name,
            allowed_consumers=allowed_consumers,
            nemo_configs=_NEMO_CONFIGS,
            replicas=1,
            env=build_api_key_env(nemo_api_token_secret.name),
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
                func=lambda: condition_status(nemo_cr=nemo_cr, condition_type="AllowedConsumersReady"),
            ):
                if sample == "True":
                    break

            status = condition_status(nemo_cr=nemo_cr, condition_type="AllowedConsumersReady")
            assert status == "True", (
                f"Expected AllowedConsumersReady status 'True' for allowedConsumers={allowed_consumers}, "
                f"got: {status!r}"
            )

    def test_allowed_consumers_from_selector_invalid_label_errors(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_api_token_secret: Secret,
    ) -> None:
        """A server with a broken label selector stops reconciling and enters an error state.

        Given: A NemoGuardrails server with allowedConsumers set to Selector mode
              but a label key that is not valid Kubernetes syntax
        When: The operator tries to reconcile it
        Then: The AllowedConsumersReady condition is False and the server phase is Error
        """
        with NemoGuardrails(
            client=admin_client,
            name="nemo-consumers-bad-selector",
            namespace=model_namespace.name,
            allowed_consumers={
                "namespaces": {
                    "from": "Selector",
                    "selector": {
                        "matchLabels": {
                            "???": "invalid-label-key",
                        }
                    },
                }
            },
            nemo_configs=_NEMO_CONFIGS,
            replicas=1,
            env=build_api_key_env(nemo_api_token_secret.name),
        ) as nemo_cr:
            for sample in TimeoutSampler(
                wait_timeout=_TIMEOUT,
                sleep=_SLEEP,
                func=lambda: condition_status(nemo_cr=nemo_cr, condition_type="AllowedConsumersReady"),
            ):
                if sample == "False":
                    break

            status = condition_status(nemo_cr=nemo_cr, condition_type="AllowedConsumersReady")
            assert status == "False", (
                f"Expected AllowedConsumersReady status 'False' for invalid label key, got: {status!r}"
            )

            phase = nemo_cr_phase(nemo_cr=nemo_cr)
            assert phase == "Error", f"Expected CR phase 'Error' for invalid label key, got: {phase!r}"
