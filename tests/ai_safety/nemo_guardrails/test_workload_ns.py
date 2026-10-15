"""Tests for the NeMo Guardrails workload namespace auto-selection.

The operator derives the workload namespace from its own controller namespace:
  - Controller in 'redhat-ods-applications' → 'redhat-ai-gateway-infra'
  - Controller in 'opendatahub' → 'odh-ai-gateway-infra'
  - Any other namespace → CR namespace (fallback)

When the derived infra namespace differs from the CR namespace the operator
validates that it exists and carries the label
'trustyai.opendatahub.io/nemo-guardrails-workload=true'.  If it does not,
the operator sets phase Error with reason WorkloadNamespaceNotReady and stops
reconciliation.

The Deployment, Service, and Route keep the CR name and are created in the
infra namespace. User ConfigMaps are copied there under their original name.
The original ConfigMap remains in the CR namespace.

Covers:
  - Infra namespace does not exist → phase Error, reason WorkloadNamespaceNotReady.
  - Infra namespace exists but is NOT labeled → the same error, and no Service
    is created in either namespace.
  - Infra namespace is labeled → the Deployment, Service, and Route land in the
    infra namespace and status.workloadNamespace reflects it.
  - nemoConfig ConfigMaps are copied into the infra namespace; the originals
    remain in the CR namespace.
"""

import contextlib
import hashlib
from collections.abc import Generator

import pytest
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.secret import Secret
from timeout_sampler import TimeoutSampler

from tests.ai_safety.nemo_guardrails.constants import NEMO_DEFAULT_CONFIG_CM_PII
from tests.ai_safety.nemo_guardrails.utils import (
    build_api_key_env,
    condition_reason,
    configmap_exists,
    nemo_cr_phase,
    nemo_cr_workload_namespace,
    route_exists,
    service_exists,
)
from utilities.infra import get_rhods_subscription
from utilities.resources.nemo_guardrails import NemoGuardrails

_TIMEOUT = 120
_SLEEP = 5

_NEMO_CONFIGS = [{"name": "workload-ns-pii", "configMaps": [NEMO_DEFAULT_CONFIG_CM_PII], "default": True}]

# The two known infra namespace names the operator looks for.
# The MaaS controller creates one of these depending on whether RHOAI or ODH is installed.
_RHOAI_AI_GATEWAY_INFRA_NAMESPACE = "redhat-ai-gateway-infra"
_ODH_AI_GATEWAY_INFRA_NAMESPACE = "odh-ai-gateway-infra"

# Label required on the infra namespace for the operator to deploy the workload there.
_WORKLOAD_NS_LABEL = "trustyai.opendatahub.io/nemo-guardrails-workload"

# Status reason set when the derived infra namespace is missing or unlabeled.
_REJECTION_REASON = "WorkloadNamespaceNotReady"


def _dns1123_name(name: str) -> str:
    """Truncate a name the same way the operator's dns1123Name does.

    Args:
        name: Candidate Kubernetes object name

    Returns:
        The lowercased name, shortened to 63 characters when necessary
    """
    lowered = name.lower()
    if len(lowered) <= 63:
        return lowered
    suffix = hashlib.sha256(lowered.encode()).hexdigest()[:8]
    trimmed = lowered[: 63 - len(suffix) - 1].rstrip("-")
    return f"{trimmed}-{suffix}"


def _workload_object_name(owner_namespace: str, owner_name: str, target_namespace: str) -> str:
    """Return the Deployment, Service, and Route name used in the workload namespace.

    Args:
        owner_namespace: Namespace of the NemoGuardrails CR
        owner_name: Name of the NemoGuardrails CR
        target_namespace: Namespace where the workload is created

    Returns:
        owner_name when the workload stays in the CR namespace, otherwise
        '<owner_namespace>-<owner_name>' truncated to a DNS-1123 label
    """
    if not target_namespace or target_namespace == owner_namespace:
        return owner_name
    return _dns1123_name(f"{owner_namespace}-{owner_name}")


def _wait_for_namespace_rejection(nemo_cr: NemoGuardrails) -> None:
    """Wait until reconciliation records a workload-namespace rejection.

    Args:
        nemo_cr: NemoGuardrails resource to poll
    """
    for phase, reason in TimeoutSampler(
        wait_timeout=_TIMEOUT,
        sleep=_SLEEP,
        func=lambda: (
            nemo_cr_phase(nemo_cr=nemo_cr),
            condition_reason(nemo_cr=nemo_cr, condition_type="ReconcileComplete"),
        ),
    ):
        if phase == "Error" and reason == _REJECTION_REASON:
            break

    phase = nemo_cr_phase(nemo_cr=nemo_cr)
    reason = condition_reason(nemo_cr=nemo_cr, condition_type="ReconcileComplete")
    assert (
        phase == "Error" and reason == _REJECTION_REASON
    ), f"Expected phase 'Error' and reason '{_REJECTION_REASON}', got phase={phase!r} reason={reason!r}"


def _assert_workload_absent(
    client: DynamicClient,
    cr_name: str,
    owner_namespace: str,
    infra_namespace: str,
) -> None:
    """Assert no Service was created for this CR in the infra or CR namespace.

    Args:
        client: DynamicClient used to query the cluster
        cr_name: Name of the NemoGuardrails CR
        owner_namespace: Namespace of the NemoGuardrails CR
        infra_namespace: Derived platform infra namespace
    """
    workload_name = _workload_object_name(
        owner_namespace=owner_namespace,
        owner_name=cr_name,
        target_namespace=infra_namespace,
    )
    for name in (cr_name, workload_name):
        assert not service_exists(
            client, name, infra_namespace
        ), f"Service '{name}' must not exist in infra namespace '{infra_namespace}'"
        assert not service_exists(
            client, name, owner_namespace
        ), f"Service '{name}' must not exist in CR namespace '{owner_namespace}'"


def _find_labeled_infra_namespace(client: DynamicClient) -> Namespace | None:
    """Return the platform-appropriate AI Gateway infra namespace if it carries the workload label.

    Uses the installed operator subscription to determine whether to check
    'redhat-ai-gateway-infra' (RHOAI) or 'odh-ai-gateway-infra' (ODH).

    Args:
        client: DynamicClient used to query the cluster

    Returns:
        The labeled Namespace object, or None if the infra namespace is missing or unlabeled
    """
    name = _platform_infra_namespace_name()
    ns = Namespace(client=client, name=name)
    if ns.exists and (ns.instance.metadata.labels or {}).get(_WORKLOAD_NS_LABEL) == "true":
        return ns
    return None


def _platform_infra_namespace_name() -> str:
    """Return the AI Gateway infra namespace name for the installed platform.

    ODH uses 'odh-ai-gateway-infra'; RHOAI uses 'redhat-ai-gateway-infra'.
    The correct name is determined by checking whether the RHOAI subscription
    exists in the operator namespace rather than relying on namespace existence,
    which can be misleading when leftover namespaces from other components are present.

    Returns:
        The platform-appropriate infra namespace name
    """
    if get_rhods_subscription() is not None:
        return _RHOAI_AI_GATEWAY_INFRA_NAMESPACE
    return _ODH_AI_GATEWAY_INFRA_NAMESPACE


@pytest.fixture
def labeled_infra_namespace(admin_client: DynamicClient) -> Generator[Namespace]:
    """Yield a labeled AI Gateway infra namespace.

    If the platform infra namespace is already labeled on the cluster it is
    yielded as-is (no changes on teardown).

    If the namespace exists but is unlabeled the workload label is added via
    ResourceEditor and removed on teardown — the namespace itself is left intact.

    If the namespace does not exist it is created with the label and deleted on
    teardown.
    """
    existing = _find_labeled_infra_namespace(client=admin_client)
    if existing is not None:
        yield existing
        return

    infra_ns_name = _platform_infra_namespace_name()
    ns = Namespace(client=admin_client, name=infra_ns_name)

    if ns.exists:
        patch = {ns: {"metadata": {"labels": {_WORKLOAD_NS_LABEL: "true"}}}}
        editor = ResourceEditor(patches=patch)
        editor.update()
        try:
            yield ns
        finally:
            ResourceEditor(patches={ns: {"metadata": {"labels": {_WORKLOAD_NS_LABEL: None}}}}).update()
    else:
        ns = Namespace(
            client=admin_client,
            name=infra_ns_name,
            label={_WORKLOAD_NS_LABEL: "true"},
        )
        ns.create()
        try:
            yield ns
        finally:
            ns.clean_up()


@pytest.mark.tier1
@pytest.mark.ai_safety
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-nemo-guardrails"})],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
class TestNemoGuardrailsWorkloadNamespace:
    """Tests that the operator deploys the workload to the right namespace."""

    def test_workload_rejects_missing_infra_namespace(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_api_token_secret: Secret,
    ) -> None:
        """The operator errors when the platform infra namespace does not exist.

        Given: A NemoGuardrails CR and no platform infra namespace on the cluster
        When: The operator tries to reconcile
        Then: The CR enters the Error phase with reason WorkloadNamespaceNotReady
              and no Service is created
        """
        existing = _find_labeled_infra_namespace(client=admin_client)
        if existing is not None:
            pytest.skip(
                f"Cluster already has labeled infra namespace '{existing.name}'. "
                "This test requires that no labeled infra namespace exists."
            )

        infra_ns_name = _platform_infra_namespace_name()
        infra_ns = Namespace(client=admin_client, name=infra_ns_name)
        if infra_ns.exists:
            pytest.skip(
                f"Infra namespace '{infra_ns_name}' already exists on the cluster. "
                "This test requires the infra namespace to be absent."
            )

        with NemoGuardrails(
            client=admin_client,
            name="nemo-workload-ns-no-infra",
            namespace=model_namespace.name,
            nemo_configs=_NEMO_CONFIGS,
            replicas=1,
            env=build_api_key_env(nemo_api_token_secret.name),
        ) as nemo_cr:
            _wait_for_namespace_rejection(nemo_cr=nemo_cr)
            _assert_workload_absent(
                client=admin_client,
                cr_name=nemo_cr.name,
                owner_namespace=model_namespace.name,
                infra_namespace=infra_ns_name,
            )

    def test_workload_rejects_unlabeled_infra_namespace(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_api_token_secret: Secret,
    ) -> None:
        """The operator errors when the infra namespace exists but is not labeled.

        Given: The platform infra namespace exists but does NOT carry the
               'trustyai.opendatahub.io/nemo-guardrails-workload=true' label
        When: The operator tries to reconcile
        Then: The CR enters the Error phase with reason WorkloadNamespaceNotReady
              and no Service is created in either namespace
        """
        existing = _find_labeled_infra_namespace(client=admin_client)
        if existing is not None:
            pytest.skip(
                f"Cluster already has labeled infra namespace '{existing.name}'. "
                "This test requires that no labeled infra namespace exists."
            )

        infra_ns_name = _platform_infra_namespace_name()
        infra_ns = Namespace(client=admin_client, name=infra_ns_name)
        ns_ctx = contextlib.nullcontext(infra_ns) if infra_ns.exists else infra_ns
        with (
            ns_ctx,
            NemoGuardrails(
                client=admin_client,
                name="nemo-workload-ns-unlabeled-infra",
                namespace=model_namespace.name,
                nemo_configs=_NEMO_CONFIGS,
                replicas=1,
                env=build_api_key_env(nemo_api_token_secret.name),
            ) as nemo_cr,
        ):
            _wait_for_namespace_rejection(nemo_cr=nemo_cr)
            _assert_workload_absent(
                client=admin_client,
                cr_name=nemo_cr.name,
                owner_namespace=model_namespace.name,
                infra_namespace=infra_ns_name,
            )

    def test_workload_deploys_to_labeled_infra_namespace(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_api_token_secret: Secret,
        labeled_infra_namespace: Namespace,
    ) -> None:
        """When an infra namespace is labeled the operator deploys the workload there.

        Given: An infra namespace ('redhat-ai-gateway-infra' or 'odh-ai-gateway-infra')
               that carries the label 'trustyai.opendatahub.io/nemo-guardrails-workload=true'
        When: The operator finishes reconciling
        Then: The Deployment, Service, and Route exist in the infra namespace
              under the CR name and not in the CR namespace, and
              status.workloadNamespace equals the infra namespace
        """
        infra_ns = labeled_infra_namespace
        with NemoGuardrails(
            client=admin_client,
            name="nemo-workload-ns-infra",
            namespace=model_namespace.name,
            nemo_configs=_NEMO_CONFIGS,
            replicas=1,
            env=build_api_key_env(nemo_api_token_secret.name),
        ) as nemo_cr:
            Deployment(
                client=admin_client,
                name=nemo_cr.name,
                namespace=infra_ns.name,
                wait_for_resource=True,
            ).wait_for_replicas()

            assert service_exists(
                admin_client, nemo_cr.name, infra_ns.name
            ), f"Expected Service '{nemo_cr.name}' in infra namespace '{infra_ns.name}'"
            assert route_exists(
                admin_client, nemo_cr.name, infra_ns.name
            ), f"Expected Route '{nemo_cr.name}' in infra namespace '{infra_ns.name}'"
            assert not service_exists(admin_client, nemo_cr.name, model_namespace.name), (
                f"Service '{nemo_cr.name}' must not exist in CR namespace '{model_namespace.name}' "
                f"when the workload is in infra namespace '{infra_ns.name}'"
            )

            for sample in TimeoutSampler(
                wait_timeout=_TIMEOUT,
                sleep=_SLEEP,
                func=lambda: nemo_cr_workload_namespace(nemo_cr=nemo_cr),
            ):
                if sample:
                    break

            assert nemo_cr_workload_namespace(nemo_cr=nemo_cr) == infra_ns.name, (
                f"Expected status.workloadNamespace='{infra_ns.name}', "
                f"got: {nemo_cr_workload_namespace(nemo_cr=nemo_cr)!r}"
            )

    def test_configmaps_copied_to_infra_namespace(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_api_token_secret: Secret,
        labeled_infra_namespace: Namespace,
    ) -> None:
        """nemoConfig ConfigMaps are copied into the infra namespace when it is the workload target.

        The operator always reads nemoConfig ConfigMaps from the CR namespace.  When
        the workload namespace differs (the labeled infra namespace), the operator
        copies each referenced ConfigMap there so the server pod can mount it.  The
        original ConfigMap is left untouched in the CR namespace.

        Given: A user ConfigMap in the CR namespace and a labeled infra namespace
        When: The operator finishes reconciling
        Then: A copy of the ConfigMap exists in the infra namespace under its
              original name, and the original ConfigMap still exists in the CR namespace
        """
        infra_ns = labeled_infra_namespace
        minimal_config = yaml.dump({
            "passthrough": True,
            "rails": {"input": {"flows": []}, "output": {"flows": []}},
        })
        with (
            ConfigMap(
                client=admin_client,
                name="nemo-workload-ns-user-config",
                namespace=model_namespace.name,
                data={"config.yaml": minimal_config, "rails.co": ""},
            ) as user_cm,
            NemoGuardrails(
                client=admin_client,
                name="nemo-workload-ns-cm-copy",
                namespace=model_namespace.name,
                nemo_configs=[{"name": "user-config", "configMaps": [user_cm.name], "default": True}],
                replicas=1,
                env=build_api_key_env(nemo_api_token_secret.name),
            ) as nemo_cr,
        ):
            Deployment(
                client=admin_client,
                name=nemo_cr.name,
                namespace=infra_ns.name,
                wait_for_resource=True,
            ).wait_for_replicas()

            assert configmap_exists(
                admin_client, user_cm.name, infra_ns.name
            ), f"Expected ConfigMap '{user_cm.name}' to be copied into infra namespace '{infra_ns.name}'"
            assert configmap_exists(admin_client, user_cm.name, model_namespace.name), (
                f"Original ConfigMap '{user_cm.name}' should still exist in "
                f"CR namespace '{model_namespace.name}' after copying"
            )
