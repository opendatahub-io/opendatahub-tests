"""
Tests for AuthPolicy cleanup after HTTPRoute deletion.

Bug 7 (High): When an HTTPRoute is deleted, the associated AuthPolicy is not
cleaned up, leaving stale references. This confirms RHOAIENG-56131.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.resource import Resource

from tests.model_serving.model_server.llmd.utils import ns_from_file

pytestmark = [pytest.mark.tier2, pytest.mark.gpu]

NAMESPACE = ns_from_file(file=__file__)


def _get_auth_policies(client: DynamicClient, namespace: str) -> list:
    """Get AuthPolicies in the given namespace."""
    try:
        return list(
            Resource.get(
                client=client,
                api_version="kuadrant.io/v1",
                kind="AuthPolicy",
                namespace=namespace,
            )
        )
    except Exception:  # noqa: BLE001
        return []


def _get_httproutes(client: DynamicClient, namespace: str) -> list:
    """Get HTTPRoutes in the given namespace."""
    try:
        return list(
            Resource.get(
                client=client,
                api_version="gateway.networking.k8s.io/v1",
                kind="HTTPRoute",
                namespace=namespace,
            )
        )
    except Exception:  # noqa: BLE001
        return []


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [({"name": NAMESPACE}, "TinyLlamaOciConfig")],
    indirect=True,
)
@pytest.mark.usefixtures("valid_aws_config", "skip_if_no_gpu_available", "skip_if_disconnected")
class TestLlmdAuthPolicyCleanup:
    """Verify AuthPolicies are cleaned up when LLMISVC HTTPRoutes are deleted.

    Preconditions:
        - LLMInferenceService deployed with auth enabled
        - GPU available on cluster

    Test Steps:
        1. Deploy auth-enabled LLMISVC
        2. Verify AuthPolicies reference valid HTTPRoutes
        3. After LLMISVC deletion, verify no stale AuthPolicies remain

    Expected Results:
        - All AuthPolicies should reference existing HTTPRoutes
        - No orphaned AuthPolicies after LLMISVC deletion (RHOAIENG-56131)
    """

    def test_authpolicies_reference_valid_httproutes(
        self,
        admin_client: DynamicClient,
        unprivileged_model_namespace,
        llmisvc: LLMInferenceService,
    ):
        """Verify each AuthPolicy references an existing HTTPRoute."""
        ns = unprivileged_model_namespace.name
        auth_policies = _get_auth_policies(client=admin_client, namespace=ns)
        httproutes = _get_httproutes(client=admin_client, namespace=ns)
        httproute_names = {r.name for r in httproutes}

        for ap in auth_policies:
            target_ref = ap.instance.spec.get("targetRef", {})
            if target_ref.get("kind") == "HTTPRoute":
                assert target_ref.get("name") in httproute_names, (
                    f"AuthPolicy {ap.name} references HTTPRoute '{target_ref.get('name')}' "
                    f"which does not exist. Available: {httproute_names}. "
                    "This may indicate stale AuthPolicy after HTTPRoute deletion (Bug 7 / RHOAIENG-56131)."
                )
