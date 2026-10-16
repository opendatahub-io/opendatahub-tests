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
infra namespace. Their pod label and Service selector are
'<cr-namespace>-<cr-name>', so another project cannot select these pods by the
CR name alone. A second CR that reuses the same name is rejected and must not
replace the first project's objects.

User ConfigMaps and Secrets are copied there as
'<cr-namespace>-<cr-name>-<source-name>'. A default ConfigMap is copied as
'<cr-namespace>-<cr-name>-default-pii'. Each copy keeps the source data. Kubernetes
rejects an owner reference that points at another namespace, so the operator
records the owner as the label 'trustyai.opendatahub.io/owner-uid' and the
annotations 'trustyai.opendatahub.io/owner-name' and
'trustyai.opendatahub.io/owner-namespace'. The original ConfigMap remains in
the CR namespace.

Covers:
  - Infra namespace does not exist → phase Error, reason WorkloadNamespaceNotReady.
  - Infra namespace exists but is NOT labeled → the same error, and no Service
    is created in either namespace.
  - Infra namespace is labeled → the Deployment, Service, and Route land in the
    infra namespace, the selector is '<cr-namespace>-<cr-name>', and
    status.workloadNamespace reflects it.
  - nemoConfig ConfigMaps and the API key Secret are copied into the infra
    namespace with the source data and the owner label and annotations.
  - A second project that reuses the CR name does not take the first project's
    Deployment.
"""

import contextlib
from collections.abc import Generator

import pytest
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.nemo_guardrails.constants import NEMO_DEFAULT_CONFIG_CM_PII
from tests.ai_safety.nemo_guardrails.utils import (
    build_api_key_env,
    condition_reason,
    condition_status,
    configmap_exists,
    nemo_cr_phase,
    nemo_cr_workload_namespace,
    route_exists,
    secret_exists,
    service_exists,
)
from utilities.infra import get_dsci_applications_namespace, get_rhods_subscription
from utilities.resources.nemo_guardrails import NemoGuardrails

_TIMEOUT = 120
_SLEEP = 5
# How long to watch after the second project starts reconciling.
_COLLISION_WINDOW = 45

_NEMO_CONFIGS = [{"name": "workload-ns-pii", "configMaps": [NEMO_DEFAULT_CONFIG_CM_PII], "default": True}]

# The two known infra namespace names the operator looks for.
# The MaaS controller creates one of these depending on whether RHOAI or ODH is installed.
_RHOAI_AI_GATEWAY_INFRA_NAMESPACE = "redhat-ai-gateway-infra"
_ODH_AI_GATEWAY_INFRA_NAMESPACE = "odh-ai-gateway-infra"

# Label required on the infra namespace for the operator to deploy the workload there.
_WORKLOAD_NS_LABEL = "trustyai.opendatahub.io/nemo-guardrails-workload"

# Status reason set when the derived infra namespace is missing or unlabeled.
_REJECTION_REASON = "WorkloadNamespaceNotReady"

# Cross-namespace ownership. Kubernetes rejects an owner reference whose owner
# lives in another namespace, so the operator records the custom resource here.
_OWNER_UID_LABEL = "trustyai.opendatahub.io/owner-uid"
_OWNER_NAME_ANNOTATION = "trustyai.opendatahub.io/owner-name"
_OWNER_NAMESPACE_ANNOTATION = "trustyai.opendatahub.io/owner-namespace"


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
    assert phase == "Error" and reason == _REJECTION_REASON, (
        f"Expected phase 'Error' and reason '{_REJECTION_REASON}', got phase={phase!r} reason={reason!r}"
    )


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
    previous_workload_name = f"{owner_namespace}-{cr_name}"
    for name in (cr_name, previous_workload_name):
        assert not service_exists(client, name, infra_namespace), (
            f"Service '{name}' must not exist in infra namespace '{infra_namespace}'"
        )
        assert not service_exists(client, name, owner_namespace), (
            f"Service '{name}' must not exist in CR namespace '{owner_namespace}'"
        )


def _label_value(labels: object, key: str) -> str:
    """Read one label from a Kubernetes label map.

    Args:
        labels: Label mapping from a Deployment or Service spec. None when the field is unset.
        key: Label key to read

    Returns:
        The label value, or an empty string when the key is absent
    """
    if not labels:
        return ""
    if isinstance(labels, dict):
        return str(labels.get(key, ""))
    getter = getattr(labels, "get", None)
    if callable(getter):
        return str(getter(key, "") or "")
    return str(getattr(labels, key, "") or "")


def _assert_workload_selector(deployment: Deployment, service: Service, selector: str) -> None:
    """Assert the Deployment and Service select pods by the infra-namespace selector.

    Args:
        deployment: Workload Deployment in the infra namespace
        service: Workload Service in the infra namespace
        selector: Expected app label, '<cr-namespace>-<cr-name>'
    """
    match_labels = deployment.instance.spec.selector.matchLabels
    template_labels = deployment.instance.spec.template.metadata.labels
    service_selector = service.instance.spec.selector
    assert _label_value(labels=match_labels, key="app") == selector, (
        f"Deployment '{deployment.name}' selector app is {_label_value(labels=match_labels, key='app')!r}, "
        f"expected {selector!r}"
    )
    assert _label_value(labels=template_labels, key="app") == selector, (
        f"Deployment '{deployment.name}' pod label app is {_label_value(labels=template_labels, key='app')!r}, "
        f"expected {selector!r}"
    )
    assert _label_value(labels=service_selector, key="app") == selector, (
        f"Service '{service.name}' selector app is {_label_value(labels=service_selector, key='app')!r}, "
        f"expected {selector!r}"
    )


def _assert_main_port_name(service: Service, cr_name: str) -> None:
    """Assert the Service port keeps the CR name.

    Auth is disabled in these tests, so the admin port is not created. The CR
    name used here is within the 15-character Service port-name limit.

    Args:
        service: Workload Service in the infra namespace
        cr_name: NemoGuardrails name, which is also the expected port name
    """
    names: list[str] = []
    for port in service.instance.spec.ports or []:
        if isinstance(port, dict):
            names.append(str(port.get("name", "")))
        else:
            names.append(str(getattr(port, "name", "")))
    assert names == [cr_name], (
        f"Service '{service.name}' ports are {names}. "
        f"'{cr_name}' is {len(cr_name)} characters, within the 15-character port-name limit, "
        "so the port keeps the custom resource name. Auth is disabled, so no admin port is created."
    )


def _assert_owned_by(resource: ConfigMap | Deployment | Secret | Service, owner: NemoGuardrails) -> None:
    """Assert a cross-namespace object records this NemoGuardrails CR as its owner.

    The workload namespace is not the custom resource namespace, so the operator
    does not set an owner reference. It stores the owner UID in a label and the
    owner name and namespace in annotations.

    Args:
        resource: Object in the infra namespace
        owner: NemoGuardrails custom resource that should own it
    """
    labels = resource.instance.metadata.labels or {}
    annotations = resource.instance.metadata.annotations or {}
    owner_uid = _label_value(labels=labels, key=_OWNER_UID_LABEL)
    assert owner_uid == str(owner.instance.metadata.uid), (
        f"{type(resource).__name__} '{resource.namespace}/{resource.name}' label {_OWNER_UID_LABEL} is {owner_uid!r}"
    )
    assert _label_value(labels=annotations, key=_OWNER_NAME_ANNOTATION) == owner.name, (
        f"{type(resource).__name__} '{resource.namespace}/{resource.name}' "
        f"annotation {_OWNER_NAME_ANNOTATION} is "
        f"{_label_value(labels=annotations, key=_OWNER_NAME_ANNOTATION)!r}"
    )
    assert _label_value(labels=annotations, key=_OWNER_NAMESPACE_ANNOTATION) == owner.namespace, (
        f"{type(resource).__name__} '{resource.namespace}/{resource.name}' "
        f"annotation {_OWNER_NAMESPACE_ANNOTATION} is "
        f"{_label_value(labels=annotations, key=_OWNER_NAMESPACE_ANNOTATION)!r}"
    )


def _resource_data(resource: ConfigMap | Secret) -> dict[str, str]:
    """Return the data map of a ConfigMap or Secret.

    Args:
        resource: ConfigMap or Secret to read

    Returns:
        Data keys and values. Secret values are not logged by callers.
    """
    raw = resource.instance.data or {}
    return {str(key): str(value) for key, value in dict(raw).items()}


def _assert_data_matches(copied: ConfigMap | Secret, source: ConfigMap | Secret) -> None:
    """Assert a copied object has the same data as its source.

    Args:
        copied: Object created in the infra namespace
        source: Object the operator copied
    """
    copied_data = _resource_data(resource=copied)
    source_data = _resource_data(resource=source)
    if isinstance(copied, Secret):
        message = f"Secret '{copied.namespace}/{copied.name}' data does not match the source Secret"
    else:
        message = (
            f"ConfigMap '{copied.namespace}/{copied.name}' data does not match "
            f"'{source.namespace}/{source.name}'. "
            f"Copied keys: {sorted(copied_data)}. Source keys: {sorted(source_data)}."
        )
    assert copied_data == source_data, message


def _second_project_took_workload(
    client: DynamicClient,
    deployment: Deployment,
    second: NemoGuardrails,
    expected_selector: str,
    copied_secret_name: str,
    infra_namespace: str,
) -> str:
    """Return why the second project replaced the first, or an empty string.

    Args:
        client: DynamicClient used to query the cluster
        deployment: First project's Deployment in the infra namespace
        second: NemoGuardrails CR in the other namespace
        expected_selector: Selector the first project deployed
        copied_secret_name: Secret name the second project would copy after a missed collision check
        infra_namespace: Namespace where the workload lives

    Returns:
        A failure reason, or an empty string when the first project's workload is unchanged
    """
    if nemo_cr_phase(nemo_cr=second) == "Ready":
        return "the second custom resource became Ready"
    if nemo_cr_workload_namespace(nemo_cr=second):
        return "the second custom resource wrote status.workloadNamespace"
    selector = _label_value(labels=deployment.instance.spec.selector.matchLabels, key="app")
    if selector != expected_selector:
        return f"the Deployment selector changed to {selector!r}"
    if secret_exists(client, copied_secret_name, infra_namespace):
        return f"Secret '{copied_secret_name}' was copied for the second custom resource"
    return ""


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
              under the CR name and not in the CR namespace. Pods and the Service
              select app=<cr-namespace>-<cr-name>. The Service port is the CR name.
              The default ConfigMap and API key Secret match their sources and carry
              this CR's owner label and annotations.
        """
        infra_ns = labeled_infra_namespace
        with NemoGuardrails(
            client=admin_client,
            name="nemo-infra",
            namespace=model_namespace.name,
            nemo_configs=_NEMO_CONFIGS,
            replicas=1,
            env=build_api_key_env(nemo_api_token_secret.name),
        ) as nemo_cr:
            deployment = Deployment(
                client=admin_client,
                name=nemo_cr.name,
                namespace=infra_ns.name,
                wait_for_resource=True,
            )
            deployment.wait_for_replicas()
            service = Service(
                client=admin_client,
                name=nemo_cr.name,
                namespace=infra_ns.name,
                ensure_exists=True,
            )
            selector = f"{model_namespace.name}-{nemo_cr.name}"
            _assert_workload_selector(deployment=deployment, service=service, selector=selector)
            _assert_main_port_name(service=service, cr_name=nemo_cr.name)
            _assert_owned_by(resource=deployment, owner=nemo_cr)
            _assert_owned_by(resource=service, owner=nemo_cr)

            assert route_exists(admin_client, nemo_cr.name, infra_ns.name), (
                f"Expected Route '{nemo_cr.name}' in infra namespace '{infra_ns.name}'"
            )
            assert not service_exists(admin_client, nemo_cr.name, model_namespace.name), (
                f"Service '{nemo_cr.name}' must not exist in CR namespace '{model_namespace.name}' "
                f"when the workload is in infra namespace '{infra_ns.name}'"
            )
            assert not service_exists(admin_client, selector, infra_ns.name), (
                f"This reconcile must not create Service '{selector}' in infra namespace '{infra_ns.name}'"
            )
            default_copy = ConfigMap(
                client=admin_client,
                name=f"{selector}-default-pii",
                namespace=infra_ns.name,
                ensure_exists=True,
            )
            default_source = ConfigMap(
                client=admin_client,
                name=NEMO_DEFAULT_CONFIG_CM_PII,
                namespace=get_dsci_applications_namespace(client=admin_client),
                ensure_exists=True,
            )
            _assert_data_matches(copied=default_copy, source=default_source)
            _assert_owned_by(resource=default_copy, owner=nemo_cr)
            copied_secret = Secret(
                client=admin_client,
                name=f"{selector}-{nemo_api_token_secret.name}",
                namespace=infra_ns.name,
                ensure_exists=True,
            )
            _assert_data_matches(copied=copied_secret, source=nemo_api_token_secret)
            _assert_owned_by(resource=copied_secret, owner=nemo_cr)

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
        Then: The infra ConfigMap '<cr-namespace>-<cr-name>-<configmap-name>'
              has the same data as the source and this CR's owner label and annotations.
              The original ConfigMap still exists in the CR namespace
        """
        infra_ns = labeled_infra_namespace
        minimal_config = yaml.dump({
            "passthrough": True,
            "rails": {"input": {"flows": []}, "output": {"flows": []}},
        })
        with (
            ConfigMap(
                client=admin_client,
                name="nemo-user-config",
                namespace=model_namespace.name,
                data={"config.yaml": minimal_config, "rails.co": ""},
            ) as user_cm,
            NemoGuardrails(
                client=admin_client,
                name="nemo-cm-copy",
                namespace=model_namespace.name,
                nemo_configs=[{"name": "user-config", "configMaps": [user_cm.name], "default": True}],
                replicas=1,
                env=build_api_key_env(nemo_api_token_secret.name),
            ) as nemo_cr,
        ):
            copied_name = f"{model_namespace.name}-{nemo_cr.name}-{user_cm.name}"
            Deployment(
                client=admin_client,
                name=nemo_cr.name,
                namespace=infra_ns.name,
                wait_for_resource=True,
            ).wait_for_replicas()

            copied = ConfigMap(
                client=admin_client,
                name=copied_name,
                namespace=infra_ns.name,
                ensure_exists=True,
            )
            _assert_data_matches(copied=copied, source=user_cm)
            _assert_owned_by(resource=copied, owner=nemo_cr)
            assert configmap_exists(admin_client, user_cm.name, model_namespace.name), (
                f"Original ConfigMap '{user_cm.name}' should still exist in "
                f"CR namespace '{model_namespace.name}' after copying"
            )

    def test_second_project_does_not_take_shared_cr_name(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_api_token_secret: Secret,
        labeled_infra_namespace: Namespace,
    ) -> None:
        """A second project that reuses the custom resource name does not take the workload.

        The Deployment, Service, and Route in the infra namespace are named the
        custom resource name. The operator rejects a second NemoGuardrails with
        that name before it creates or updates anything there.

        Given: A NemoGuardrails named 'nemo-shared' is running in the labeled infra namespace
        When: Another namespace creates a NemoGuardrails with the same name
        Then: The infra Deployment selector stays '<first-namespace>-nemo-shared',
              the Deployment stays owned by the first custom resource, and the
              second custom resource does not become Ready
        """
        infra_ns = labeled_infra_namespace
        with NemoGuardrails(
            client=admin_client,
            name="nemo-shared",
            namespace=model_namespace.name,
            nemo_configs=_NEMO_CONFIGS,
            replicas=1,
            env=build_api_key_env(nemo_api_token_secret.name),
        ) as first:
            deployment = Deployment(
                client=admin_client,
                name=first.name,
                namespace=infra_ns.name,
                wait_for_resource=True,
            )
            deployment.wait_for_replicas()
            expected_selector = f"{model_namespace.name}-{first.name}"
            assert _label_value(labels=deployment.instance.spec.selector.matchLabels, key="app") == expected_selector, (
                f"Deployment selector app is "
                f"{_label_value(labels=deployment.instance.spec.selector.matchLabels, key='app')!r}, "
                f"expected {expected_selector!r}"
            )
            _assert_owned_by(resource=deployment, owner=first)

            with (
                Namespace(client=admin_client, name="test-nemo-other") as other_ns,
                Secret(
                    client=admin_client,
                    name="nemo-api-token",
                    namespace=other_ns.name,
                    string_data={"token": "test-token-123"},  # pragma: allowlist secret
                    type="Opaque",
                ) as other_secret,
                NemoGuardrails(
                    client=admin_client,
                    name=first.name,
                    namespace=other_ns.name,
                    nemo_configs=_NEMO_CONFIGS,
                    replicas=1,
                    env=build_api_key_env(other_secret.name),
                ) as second,
            ):
                copied_secret_name = f"{other_ns.name}-{second.name}-{other_secret.name}"
                for status in TimeoutSampler(
                    wait_timeout=_TIMEOUT,
                    sleep=_SLEEP,
                    func=lambda: condition_status(nemo_cr=second, condition_type="AllowedConsumersReady"),
                ):
                    if status == "True":
                        break
                assert condition_status(nemo_cr=second, condition_type="AllowedConsumersReady") == "True", (
                    "The second custom resource did not start reconciling"
                )
                try:
                    for reason in TimeoutSampler(
                        wait_timeout=_COLLISION_WINDOW,
                        sleep=_SLEEP,
                        func=lambda: (
                            _second_project_took_workload(
                                client=admin_client,
                                deployment=deployment,
                                second=second,
                                expected_selector=expected_selector,
                                copied_secret_name=copied_secret_name,
                                infra_namespace=infra_ns.name,
                            )
                            or None
                        ),
                    ):
                        if reason:
                            raise AssertionError(f"Second project took the workload: {reason}")
                except TimeoutExpiredError:
                    pass
                assert (
                    _label_value(labels=deployment.instance.spec.selector.matchLabels, key="app") == expected_selector
                )
                _assert_owned_by(resource=deployment, owner=first)
                assert nemo_cr_phase(nemo_cr=second) != "Ready"
                assert nemo_cr_workload_namespace(nemo_cr=second) == ""
                assert not secret_exists(admin_client, copied_secret_name, infra_ns.name), (
                    f"Secret '{copied_secret_name}' must not be created for the rejected custom resource"
                )
