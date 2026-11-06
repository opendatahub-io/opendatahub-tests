"""Tests for NeMo Guardrails internal service endpoint (status.endpoint).

The endpoint status field reports the in-cluster Service URL of the NeMo Guardrails server.
It uses https when authentication is enabled and http when authentication is disabled.
On ODH/RHOAI the operator deploys the server into the AI Gateway infra namespace as
"<server namespace>-<server name>", and the endpoint points there.

Covers:
  - A server without auth reports an http endpoint; a server with auth reports an https endpoint.
  - Enabling auth on a running server switches the reported endpoint from http to https.
  - A CA bundle ConfigMap in a namespace other than the server namespace or workload namespace
    is rejected: status.ca.userCAError explains why, a Warning Event is recorded, and the server
    still deploys without the CA bundle.
  - Infra namespace only:
    - The workload, endpoint, and status.workloadNamespace all point at the infra namespace.
    - A CA bundle ConfigMap in the infra namespace is mounted in place instead of copied.
    - An optional Secret reference that does not exist does not block the copy into the infra namespace.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from timeout_sampler import TimeoutSampler

from tests.ai_safety.nemo_guardrails.constants import NEMO_DEFAULT_CONFIG_CM_PII
from tests.ai_safety.nemo_guardrails.utils import (
    build_api_key_env,
    list_nemo_warning_events,
    nemo_cr_ca_status,
    nemo_cr_endpoint,
    nemo_cr_phase,
    nemo_cr_user_ca_error,
    nemo_cr_workload_namespace,
    nemo_internal_endpoint,
    wait_for_nemo_endpoint,
    wait_for_nemo_workload_ready,
)
from utilities.resources.nemo_guardrails import NemoGuardrails

_TIMEOUT = 120
_SLEEP = 5

_AUTH_ANNOTATION = "security.opendatahub.io/enable-auth"
_CA_BUNDLE_NAMESPACE_REJECTED_REASON = "CABundleNamespaceRejected"

_NEMO_CONFIGS = [{"name": "internal-endpoint-pii", "configMaps": [NEMO_DEFAULT_CONFIG_CM_PII], "default": True}]


@pytest.mark.tier2
@pytest.mark.ai_safety
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-nemo-guardrails"})],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
class TestNemoGuardrailsInternalServiceEndpoint:
    """Tests that the operator reports the internal service endpoint wherever the workload is deployed."""

    @pytest.mark.parametrize(
        "cr_name, annotations, expected_scheme",
        [
            pytest.param("nemo-endpoint-no-auth", None, "http", id="test_no_auth_reports_http"),
            pytest.param("nemo-endpoint-auth", {_AUTH_ANNOTATION: "true"}, "https", id="test_auth_reports_https"),
        ],
    )
    def test_internal_service_endpoint_scheme_matches_auth(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_api_token_secret: Secret,
        nemo_workload_namespace: str,
        cr_name: str,
        annotations: dict[str, str] | None,
        expected_scheme: str,
    ) -> None:
        """A ready server reports its in-cluster Service URL with a scheme that matches its auth setting.

        Given: A NemoGuardrails server created with or without the enable-auth annotation
        When: The operator reconciles it and the deployment becomes ready
        Then: status.endpoint is http://<service>.<workload namespace>.svc.cluster.local without auth,
              or https://<service>.<workload namespace>.svc.cluster.local with auth
        """
        with NemoGuardrails(
            client=admin_client,
            name=cr_name,
            namespace=model_namespace.name,
            annotations=annotations,
            nemo_configs=_NEMO_CONFIGS,
            replicas=1,
            env=build_api_key_env(nemo_api_token_secret.name),
        ) as nemo_cr:
            wait_for_nemo_workload_ready(
                client=admin_client, nemo_cr=nemo_cr, workload_namespace=nemo_workload_namespace
            )

            expected_endpoint = nemo_internal_endpoint(
                nemo_cr=nemo_cr, workload_namespace=nemo_workload_namespace, scheme=expected_scheme
            )
            wait_for_nemo_endpoint(nemo_cr=nemo_cr, expected_endpoint=expected_endpoint)

            endpoint = nemo_cr_endpoint(nemo_cr=nemo_cr)
            assert endpoint == expected_endpoint, f"Expected status.endpoint {expected_endpoint!r}, got: {endpoint!r}"

    def test_internal_service_endpoint_switches_to_https_when_auth_enabled(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_api_token_secret: Secret,
        nemo_workload_namespace: str,
    ) -> None:
        """Turning on auth for a running server switches its reported endpoint from http to https.

        Given: A ready NemoGuardrails server without auth reporting an http endpoint
        When: The enable-auth annotation is set to "true"
        Then: status.endpoint switches to https and the redeployed server becomes ready
        """
        with NemoGuardrails(
            client=admin_client,
            name="nemo-endpoint-auth-toggle",
            namespace=model_namespace.name,
            nemo_configs=_NEMO_CONFIGS,
            replicas=1,
            env=build_api_key_env(nemo_api_token_secret.name),
        ) as nemo_cr:
            deployment = wait_for_nemo_workload_ready(
                client=admin_client, nemo_cr=nemo_cr, workload_namespace=nemo_workload_namespace
            )
            wait_for_nemo_endpoint(
                nemo_cr=nemo_cr,
                expected_endpoint=nemo_internal_endpoint(
                    nemo_cr=nemo_cr, workload_namespace=nemo_workload_namespace, scheme="http"
                ),
            )

            nemo_cr.update(
                resource_dict={
                    "metadata": {"name": nemo_cr.name, "annotations": {_AUTH_ANNOTATION: "true"}},
                }
            )

            https_endpoint = nemo_internal_endpoint(
                nemo_cr=nemo_cr, workload_namespace=nemo_workload_namespace, scheme="https"
            )
            wait_for_nemo_endpoint(nemo_cr=nemo_cr, expected_endpoint=https_endpoint)
            deployment.wait_for_replicas()

            endpoint = nemo_cr_endpoint(nemo_cr=nemo_cr)
            assert endpoint == https_endpoint, (
                f"Expected status.endpoint {https_endpoint!r} after enabling auth, got: {endpoint!r}"
            )

    def test_ca_bundle_from_unrelated_namespace_is_rejected(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_api_token_secret: Secret,
        nemo_workload_namespace: str,
        nemo_unrelated_namespace: Namespace,
        nemo_unrelated_namespace_ca_bundle: ConfigMap,
    ) -> None:
        """A CA bundle ConfigMap from an unrelated namespace is not mounted and the rejection is reported.

        Given: A NemoGuardrails server whose caBundleConfig points at a ConfigMap in a namespace
               that is neither the server namespace nor its workload namespace
        When: The operator reconciles it
        Then: status.ca.userCAError names the rejected, server, and workload namespaces,
              a CABundleNamespaceRejected Warning Event is recorded,
              and the server still deploys and reports its endpoint
        """
        with NemoGuardrails(
            client=admin_client,
            name="nemo-endpoint-unrelated-ca",
            namespace=model_namespace.name,
            ca_bundle_config={
                "configMapName": nemo_unrelated_namespace_ca_bundle.name,
                "configMapNamespace": nemo_unrelated_namespace.name,
                "configMapKeys": ["ca.crt"],
            },
            nemo_configs=_NEMO_CONFIGS,
            replicas=1,
            env=build_api_key_env(nemo_api_token_secret.name),
        ) as nemo_cr:
            wait_for_nemo_workload_ready(
                client=admin_client, nemo_cr=nemo_cr, workload_namespace=nemo_workload_namespace
            )

            for sample in TimeoutSampler(
                wait_timeout=_TIMEOUT,
                sleep=_SLEEP,
                func=lambda: nemo_cr_user_ca_error(nemo_cr=nemo_cr),
            ):
                if sample:
                    break

            user_ca_error = nemo_cr_user_ca_error(nemo_cr=nemo_cr) or ""
            for namespace_name in (nemo_unrelated_namespace.name, model_namespace.name, nemo_workload_namespace):
                assert f'"{namespace_name}"' in user_ca_error, (
                    f"Expected status.ca.userCAError to name namespace {namespace_name!r}, got: {user_ca_error!r}"
                )

            for events in TimeoutSampler(
                wait_timeout=_TIMEOUT,
                sleep=_SLEEP,
                func=lambda: list_nemo_warning_events(
                    client=admin_client,
                    nemo_cr=nemo_cr,
                    reason=_CA_BUNDLE_NAMESPACE_REJECTED_REASON,
                ),
            ):
                if events:
                    break

            assert any(nemo_unrelated_namespace.name in event.get("message", "") for event in events), (
                f"Expected a {_CA_BUNDLE_NAMESPACE_REJECTED_REASON} Warning Event naming "
                f"{nemo_unrelated_namespace.name!r}, got: {events}"
            )

            wait_for_nemo_endpoint(
                nemo_cr=nemo_cr,
                expected_endpoint=nemo_internal_endpoint(
                    nemo_cr=nemo_cr, workload_namespace=nemo_workload_namespace, scheme="http"
                ),
            )


@pytest.mark.tier2
@pytest.mark.ai_safety
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-nemo-guardrails"})],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
class TestNemoGuardrailsInfraNamespaceWorkload:
    """Tests for servers whose workload the operator places in the AI Gateway infra namespace.

    Skipped when the operator deploys workloads into the server's own namespace.
    """

    def test_workload_and_endpoint_live_in_infra_namespace(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_api_token_secret: Secret,
        nemo_infra_namespace: str,
    ) -> None:
        """A server's workload runs in the infra namespace and its status points there.

        Given: A NemoGuardrails server created in a user namespace
        When: The operator reconciles it on an ODH/RHOAI install
        Then: The Deployment "<server namespace>-<server name>" is ready in the infra namespace,
              no Deployment is created in the server namespace, status.workloadNamespace is the
              infra namespace, and status.endpoint is the infra namespace Service URL
        """
        with NemoGuardrails(
            client=admin_client,
            name="nemo-infra-endpoint",
            namespace=model_namespace.name,
            nemo_configs=_NEMO_CONFIGS,
            replicas=1,
            env=build_api_key_env(nemo_api_token_secret.name),
        ) as nemo_cr:
            wait_for_nemo_workload_ready(client=admin_client, nemo_cr=nemo_cr, workload_namespace=nemo_infra_namespace)

            expected_endpoint = nemo_internal_endpoint(
                nemo_cr=nemo_cr, workload_namespace=nemo_infra_namespace, scheme="http"
            )
            wait_for_nemo_endpoint(nemo_cr=nemo_cr, expected_endpoint=expected_endpoint)

            workload_namespace = nemo_cr_workload_namespace(nemo_cr=nemo_cr)
            assert workload_namespace == nemo_infra_namespace, (
                f"Expected status.workloadNamespace {nemo_infra_namespace!r}, got: {workload_namespace!r}"
            )
            assert not Deployment(client=admin_client, name=nemo_cr.name, namespace=model_namespace.name).exists, (
                f"Expected no Deployment {nemo_cr.name!r} in the server namespace {model_namespace.name!r}"
            )

    def test_ca_bundle_from_infra_namespace_is_mounted_in_place(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_api_token_secret: Secret,
        nemo_infra_namespace: str,
        nemo_infra_namespace_ca_bundle: ConfigMap,
    ) -> None:
        """A CA bundle ConfigMap in the infra namespace is accepted and mounted without being copied.

        Given: A NemoGuardrails server whose caBundleConfig points at a ConfigMap in the infra namespace
        When: The operator reconciles it
        Then: status.ca.userCAFound is true with no userCAError, no CABundleNamespaceRejected Event is
              recorded, the Deployment mounts the original ConfigMap, and no copy is created
        """
        with NemoGuardrails(
            client=admin_client,
            name="nemo-infra-ca",
            namespace=model_namespace.name,
            ca_bundle_config={
                "configMapName": nemo_infra_namespace_ca_bundle.name,
                "configMapNamespace": nemo_infra_namespace,
                "configMapKeys": ["ca.crt"],
            },
            nemo_configs=_NEMO_CONFIGS,
            replicas=1,
            env=build_api_key_env(nemo_api_token_secret.name),
        ) as nemo_cr:
            deployment = wait_for_nemo_workload_ready(
                client=admin_client, nemo_cr=nemo_cr, workload_namespace=nemo_infra_namespace
            )

            for ca_status in TimeoutSampler(
                wait_timeout=_TIMEOUT,
                sleep=_SLEEP,
                func=lambda: nemo_cr_ca_status(nemo_cr=nemo_cr),
            ):
                if ca_status.get("userCAFound"):
                    break

            assert not ca_status.get("userCAError"), (
                f"Expected no status.ca.userCAError for an infra namespace CA bundle, got: {ca_status}"
            )
            events = list_nemo_warning_events(
                client=admin_client, nemo_cr=nemo_cr, reason=_CA_BUNDLE_NAMESPACE_REJECTED_REASON
            )
            assert not events, f"Expected no {_CA_BUNDLE_NAMESPACE_REJECTED_REASON} Events, got: {events}"

            mounted_config_maps = {
                volume.configMap.name
                for volume in deployment.instance.spec.template.spec.volumes or []
                if volume.configMap
            }
            assert nemo_infra_namespace_ca_bundle.name in mounted_config_maps, (
                f"Expected Deployment to mount ConfigMap {nemo_infra_namespace_ca_bundle.name!r} in place, "
                f"mounted ConfigMaps: {sorted(mounted_config_maps)}"
            )
            copied_name = f"{model_namespace.name}-{nemo_cr.name}-{nemo_infra_namespace_ca_bundle.name}"
            assert not ConfigMap(client=admin_client, name=copied_name, namespace=nemo_infra_namespace).exists, (
                f"Expected no copied CA bundle ConfigMap {copied_name!r} in {nemo_infra_namespace!r}"
            )

    def test_missing_optional_secret_does_not_block_infra_deployment(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_api_token_secret: Secret,
        nemo_infra_namespace: str,
    ) -> None:
        """A missing Secret marked optional is skipped when Secrets are copied into the infra namespace.

        Given: A NemoGuardrails server whose env references an existing Secret and a missing optional Secret
        When: The operator copies referenced Secrets into the infra namespace
        Then: The existing Secret is copied, the missing one is skipped, and the server becomes Ready
        """
        missing_secret_name = "nemo-missing-optional-secret"  # pragma: allowlist secret
        with NemoGuardrails(
            client=admin_client,
            name="nemo-infra-optional-secret",
            namespace=model_namespace.name,
            nemo_configs=_NEMO_CONFIGS,
            replicas=1,
            env=[
                *build_api_key_env(nemo_api_token_secret.name),
                {
                    "name": "OPTIONAL_TOKEN",
                    "valueFrom": {"secretKeyRef": {"name": missing_secret_name, "key": "token", "optional": True}},
                },
            ],
        ) as nemo_cr:
            wait_for_nemo_workload_ready(client=admin_client, nemo_cr=nemo_cr, workload_namespace=nemo_infra_namespace)

            for phase in TimeoutSampler(
                wait_timeout=_TIMEOUT,
                sleep=_SLEEP,
                func=lambda: nemo_cr_phase(nemo_cr=nemo_cr),
            ):
                if phase == "Ready":
                    break

            copy_prefix = f"{model_namespace.name}-{nemo_cr.name}"
            copied_secret_name = f"{copy_prefix}-{nemo_api_token_secret.name}"
            assert Secret(client=admin_client, name=copied_secret_name, namespace=nemo_infra_namespace).exists, (
                f"Expected Secret {copied_secret_name!r} to be copied into {nemo_infra_namespace!r}"
            )
            skipped_secret_name = f"{copy_prefix}-{missing_secret_name}"
            assert not Secret(client=admin_client, name=skipped_secret_name, namespace=nemo_infra_namespace).exists, (
                f"Expected no Secret {skipped_secret_name!r} in {nemo_infra_namespace!r} for a missing optional Secret"
            )
