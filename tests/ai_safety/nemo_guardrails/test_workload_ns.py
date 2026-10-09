"""Tests for the NeMo Guardrails workload namespace auto-selection.

The operator checks for a known AI Gateway infra namespace, either
'redhat-ai-gateway-infra' (RHOAI) or 'odh-ai-gateway-infra' (ODH), and deploys
the workload (Deployment, Service, Route) there when that namespace carries the
label 'trustyai.opendatahub.io/nemo-guardrails-workload=true'.

When neither infra namespace is labeled the workload lands in the same namespace
as the custom resource.  After each successful reconciliation the operator records
status.workloadNamespace with the namespace it actually deployed into.

ConfigMaps referenced in nemoConfigs are always read from the CR namespace.  When
the workload namespace differs the operator copies them there so the server pod can
mount them.

Covers:
  - Neither infra namespace exists → workload lands in the CR namespace.
  - Infra namespace exists but is NOT labeled → workload stays in the CR namespace.
  - Infra namespace is labeled → workload lands in the infra namespace and
    status.workloadNamespace reflects it.
  - nemoConfig ConfigMaps are copied into the infra namespace; the originals
    remain in the CR namespace.
"""

import pytest
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from timeout_sampler import TimeoutSampler

from tests.ai_safety.nemo_guardrails.constants import NEMO_DEFAULT_CONFIG_CM_PII
from tests.ai_safety.nemo_guardrails.utils import (
    build_api_key_env,
    configmap_exists,
    nemo_cr_workload_namespace,
    route_exists,
    service_exists,
)
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


@pytest.mark.tier2
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

    def test_workload_stays_in_cr_namespace_without_infra_ns(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_api_token_secret: Secret,
    ) -> None:
        """When neither infra namespace exists the workload lands in the CR namespace.

        Given: A NemoGuardrails CR and no 'redhat-ai-gateway-infra' or
               'odh-ai-gateway-infra' namespace on the cluster
        When: The operator finishes reconciling
        Then: The Deployment, Service, and Route all exist in the CR namespace
              and status.workloadNamespace equals the CR namespace
        """
        with NemoGuardrails(
            client=admin_client,
            name="nemo-workload-ns-no-infra",
            namespace=model_namespace.name,
            nemo_configs=_NEMO_CONFIGS,
            replicas=1,
            env=build_api_key_env(nemo_api_token_secret.name),
        ) as nemo_cr:
            Deployment(
                client=admin_client,
                name=nemo_cr.name,
                namespace=model_namespace.name,
                wait_for_resource=True,
            ).wait_for_replicas()

            assert service_exists(admin_client, nemo_cr.name, model_namespace.name), (
                f"Expected Service '{nemo_cr.name}' in CR namespace '{model_namespace.name}' "
                "when no infra namespace exists"
            )
            assert route_exists(admin_client, nemo_cr.name, model_namespace.name), (
                f"Expected Route '{nemo_cr.name}' in CR namespace '{model_namespace.name}' "
                "when no infra namespace exists"
            )

            for sample in TimeoutSampler(
                wait_timeout=_TIMEOUT,
                sleep=_SLEEP,
                func=lambda: nemo_cr_workload_namespace(nemo_cr=nemo_cr),
            ):
                if sample:
                    break

            assert nemo_cr_workload_namespace(nemo_cr=nemo_cr) == model_namespace.name, (
                f"Expected status.workloadNamespace='{model_namespace.name}', "
                f"got: {nemo_cr_workload_namespace(nemo_cr=nemo_cr)!r}"
            )

    @pytest.mark.parametrize(
        "infra_ns_name",
        [
            pytest.param(_RHOAI_AI_GATEWAY_INFRA_NAMESPACE, id="test-rhoai-infra-namespace"),
            pytest.param(_ODH_AI_GATEWAY_INFRA_NAMESPACE, id="test-odh-infra-namespace"),
        ],
    )
    def test_workload_stays_in_cr_namespace_when_infra_ns_not_labeled(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_api_token_secret: Secret,
        infra_ns_name: str,
    ) -> None:
        """The infra namespace must be labeled or the workload stays in the CR namespace.

        Given: An infra namespace that exists but does NOT carry the
               'trustyai.opendatahub.io/nemo-guardrails-workload=true' label
        When: The operator finishes reconciling
        Then: The Deployment, Service, and Route land in the CR namespace (not the
              infra namespace) and status.workloadNamespace equals the CR namespace
        """
        with (
            Namespace(
                client=admin_client,
                name=infra_ns_name,
            ),
            NemoGuardrails(
                client=admin_client,
                name="nemo-workload-ns-unlabeled-infra",
                namespace=model_namespace.name,
                nemo_configs=_NEMO_CONFIGS,
                replicas=1,
                env=build_api_key_env(nemo_api_token_secret.name),
            ) as nemo_cr,
        ):
            Deployment(
                client=admin_client,
                name=nemo_cr.name,
                namespace=model_namespace.name,
                wait_for_resource=True,
            ).wait_for_replicas()

            assert service_exists(admin_client, nemo_cr.name, model_namespace.name), (
                f"Expected Service '{nemo_cr.name}' in CR namespace '{model_namespace.name}' "
                f"when infra namespace '{infra_ns_name}' has no label"
            )
            assert route_exists(admin_client, nemo_cr.name, model_namespace.name), (
                f"Expected Route '{nemo_cr.name}' in CR namespace '{model_namespace.name}' "
                f"when infra namespace '{infra_ns_name}' has no label"
            )
            assert not service_exists(admin_client, nemo_cr.name, infra_ns_name), (
                f"Service '{nemo_cr.name}' must not exist in unlabeled infra namespace '{infra_ns_name}'"
            )

            for sample in TimeoutSampler(
                wait_timeout=_TIMEOUT,
                sleep=_SLEEP,
                func=lambda: nemo_cr_workload_namespace(nemo_cr=nemo_cr),
            ):
                if sample:
                    break

            assert nemo_cr_workload_namespace(nemo_cr=nemo_cr) == model_namespace.name, (
                f"Expected status.workloadNamespace='{model_namespace.name}', "
                f"got: {nemo_cr_workload_namespace(nemo_cr=nemo_cr)!r}"
            )

    @pytest.mark.parametrize(
        "infra_ns_name",
        [
            pytest.param(_RHOAI_AI_GATEWAY_INFRA_NAMESPACE, id="test-rhoai-infra-namespace"),
            pytest.param(_ODH_AI_GATEWAY_INFRA_NAMESPACE, id="test-odh-infra-namespace"),
        ],
    )
    def test_workload_deploys_to_labeled_infra_namespace(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_api_token_secret: Secret,
        infra_ns_name: str,
    ) -> None:
        """When an infra namespace is labeled the operator deploys the workload there.

        Given: An infra namespace ('redhat-ai-gateway-infra' or 'odh-ai-gateway-infra')
               that carries the label 'trustyai.opendatahub.io/nemo-guardrails-workload=true'
        When: The operator finishes reconciling
        Then: The Deployment, Service, and Route all exist in the infra namespace
              (not in the CR namespace) and status.workloadNamespace equals the infra namespace
        """
        with (
            Namespace(
                client=admin_client,
                name=infra_ns_name,
                labels={_WORKLOAD_NS_LABEL: "true"},
            ) as infra_ns,
            NemoGuardrails(
                client=admin_client,
                name="nemo-workload-ns-infra",
                namespace=model_namespace.name,
                nemo_configs=_NEMO_CONFIGS,
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

            assert service_exists(admin_client, nemo_cr.name, infra_ns.name), (
                f"Expected Service '{nemo_cr.name}' in infra namespace '{infra_ns.name}'"
            )
            assert route_exists(admin_client, nemo_cr.name, infra_ns.name), (
                f"Expected Route '{nemo_cr.name}' in infra namespace '{infra_ns.name}'"
            )
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
    ) -> None:
        """nemoConfig ConfigMaps are copied into the infra namespace when it is the workload target.

        The operator always reads nemoConfig ConfigMaps from the CR namespace.  When
        the workload namespace differs (the labeled infra namespace), the operator
        copies each referenced ConfigMap there so the server pod can mount it.  The
        original ConfigMap is left untouched in the CR namespace.

        Given: A user ConfigMap in the CR namespace and a labeled 'redhat-ai-gateway-infra'
               namespace
        When: The operator finishes reconciling
        Then: A copy of the ConfigMap exists in the infra namespace
              The original ConfigMap still exists in the CR namespace
        """
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
            Namespace(
                client=admin_client,
                name=_RHOAI_AI_GATEWAY_INFRA_NAMESPACE,
                labels={_WORKLOAD_NS_LABEL: "true"},
            ) as infra_ns,
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

            assert configmap_exists(admin_client, user_cm.name, infra_ns.name), (
                f"Expected ConfigMap '{user_cm.name}' to be copied into infra namespace '{infra_ns.name}'"
            )
            assert configmap_exists(admin_client, user_cm.name, model_namespace.name), (
                f"Original ConfigMap '{user_cm.name}' should still exist in "
                f"CR namespace '{model_namespace.name}' after copying"
            )
