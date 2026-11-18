"""RHOAIENG-99402: live usage-log identity capture on a dedicated test cluster."""

import ssl
from collections.abc import Generator
from contextlib import ExitStack
from time import sleep, time_ns

import portforward
import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.dsc_initialization import DSCInitialization
from ocp_resources.maas_auth_policy import MaaSAuthPolicy
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.maas_subscription import MaaSSubscription
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.service import Service

from tests.ai_gateway.models_as_a_service.maas_subscription.utils import chat_payload_for_url, poll_expected_status
from tests.ai_gateway.models_as_a_service.observability.usage_log_capture import (
    LokiUsageLogClient,
    wait_for_usage_log,
)
from tests.ai_gateway.models_as_a_service.observability.utils import resolve_maas_monitoring_namespace
from tests.ai_gateway.models_as_a_service.utils import (
    build_maas_headers,
    create_api_key,
    patch_llmisvc_with_maas_router,
    revoke_api_key,
)
from utilities.constants import ModelStorage
from utilities.general import generate_random_name
from utilities.image_constants import SharedImages
from utilities.infra import get_openshift_token
from utilities.llmd_utils import create_llmisvc
from utilities.resources.maastenantconfig import MaasTenantConfig

LOKISTACK_CRD = "lokistacks.loki.grafana.com"
LOKI_GATEWAY_SERVICE = "data-science-usage-logs-lokistack-gateway-http"
LOKI_CA_BUNDLE_CONFIGMAP = "data-science-usage-logs-lokistack-gateway-ca-bundle"
CAPTURE_STATES = (
    (False, False),
    (True, True),
    (True, False),
    (False, True),
)


@pytest.fixture
def loki_usage_client(
    admin_client: DynamicClient,
    dsci_resource: DSCInitialization,
    kubconfig_filepath: str,
    usage_logs_monitoring_ready: None,
) -> Generator[LokiUsageLogClient]:
    """Connect to the DSCI-managed Loki gateway only when its log pipeline is ready."""
    if not CustomResourceDefinition(client=admin_client, name=LOKISTACK_CRD).exists:
        pytest.fail("[blocked] RHOAIENG-99402 requires the ODH LokiStack CRD")
    monitoring_namespace = resolve_maas_monitoring_namespace(admin_client=admin_client)
    assert monitoring_namespace, "MaaS monitoring namespace is not configured"
    monitoring = admin_client.resources.get(
        api_version="services.platform.opendatahub.io/v1alpha1", kind="Monitoring"
    ).get(name="default-monitoring", namespace=monitoring_namespace)
    monitoring_conditions = {
        condition.get("type"): condition.get("status")
        for condition in monitoring.to_dict().get("status", {}).get("conditions", [])
    }
    dsci_conditions = {
        condition.get("type"): condition.get("status")
        for condition in dsci_resource.instance.to_dict().get("status", {}).get("conditions", [])
    }
    if (
        monitoring_conditions.get("UsageLogsLokiStackAvailable") != "True"
        or dsci_conditions.get("UsageLogsCollectorAvailable") != "True"
    ):
        pytest.fail("[blocked] RHOAIENG-99402 requires a ready LokiStack and usage-log collector")
    gateway_service = Service(client=admin_client, name=LOKI_GATEWAY_SERVICE, namespace=monitoring_namespace)
    assert gateway_service.exists, "DSCI-managed Loki gateway Service is missing"

    ca_bundle = ConfigMap(
        client=admin_client,
        name=LOKI_CA_BUNDLE_CONFIGMAP,
        namespace=monitoring_namespace,
        ensure_exists=True,
    ).instance.data["service-ca.crt"]
    tls_context = ssl.create_default_context(cadata=ca_bundle)
    # A Kubernetes port-forward connects to a known Service, but its TLS certificate
    # names the Service rather than 127.0.0.1. Keep CA verification enabled.
    tls_context.check_hostname = False
    with portforward.forward(
        namespace=monitoring_namespace,
        pod_or_service=gateway_service.name,
        from_port=0,
        to_port=8080,
        config_path=kubconfig_filepath,
        bind_ip="127.0.0.1",
        waiting=2,
    ) as forwarder:
        yield LokiUsageLogClient(
            port=forwarder.from_port,
            tls_context=tls_context,
            bearer_token=get_openshift_token(client=admin_client),
        )


def wait_for_tenant_generation(tenant_config: MaasTenantConfig) -> None:
    """Wait until the tenant controller reports the current telemetry generation.

    Args:
        tenant_config: Patched tenant configuration.

    Raises:
        AssertionError: If the generation does not reconcile within five minutes.
    """
    for _attempt in range(60):
        resource = tenant_config.instance.to_dict()
        generation = resource["metadata"]["generation"]
        ready = next(
            (
                condition
                for condition in resource.get("status", {}).get("conditions", [])
                if condition["type"] == "Ready"
            ),
            None,
        )
        if ready and ready.get("status") == "True" and ready.get("observedGeneration", 0) >= generation:
            return
        sleep(5)
    raise AssertionError("MaasTenantConfig did not reconcile the telemetry update")


@pytest.mark.tier2
@pytest.mark.slow
def test_usage_log_identity_capture_follows_tenant_settings(
    loki_usage_client: LokiUsageLogClient,
    admin_client: DynamicClient,
    maas_subscription_namespace: Namespace,
    maas_unprivileged_model_namespace: Namespace,
    maas_host: str,
    base_url: str,
    request_session_http: requests.Session,
) -> None:
    """Given tenant capture toggles, when governed inference runs, then Loki logs carry only enabled identity fields."""
    gateway_probe = request_session_http.get(f"{base_url}/v1/models", timeout=30)
    assert gateway_probe.status_code in {200, 401}, "MaaS API is not reachable through the gateway"
    suffix = generate_random_name()
    organization_id = f"e2e-org-{suffix}"
    model_name = f"usage-capture-{suffix}"
    subscription_name = f"usage-capture-{suffix}"
    token = get_openshift_token(client=admin_client)
    tenant_config = MaasTenantConfig(
        client=admin_client,
        name="default-tenant",
        namespace=maas_subscription_namespace.name,
        ensure_exists=True,
    )
    assert tenant_config.exists, "MaaS default tenant configuration is missing"

    with ExitStack() as stack:
        namespace = maas_unprivileged_model_namespace
        model = stack.enter_context(
            cm=create_llmisvc(
                client=admin_client,
                name=model_name,
                namespace=namespace.name,
                storage_uri=ModelStorage.OCI.TINYLLAMA,
                container_image=SharedImages.VLLM_CPU,
                container_resources={
                    "limits": {"cpu": "2", "memory": "12Gi"},
                    "requests": {"cpu": "1", "memory": "8Gi"},
                },
                wait=False,
                timeout=900,
                teardown=True,
            )
        )
        stack.enter_context(cm=patch_llmisvc_with_maas_router(llm_service=model))
        model.wait_for_condition(condition="Ready", status="True", timeout=900)
        stack.enter_context(
            cm=MaaSModelRef(
                client=admin_client,
                name=model_name,
                namespace=namespace.name,
                model_ref={"name": model_name, "namespace": namespace.name, "kind": "LLMInferenceService"},
                wait_for_resource=True,
                teardown=True,
            )
        )
        auth_policy = stack.enter_context(
            cm=MaaSAuthPolicy(
                client=admin_client,
                name=f"usage-capture-{suffix}",
                namespace=maas_subscription_namespace.name,
                model_refs=[{"name": model_name, "namespace": namespace.name}],
                subjects={"groups": [{"name": "system:authenticated"}]},
                wait_for_resource=True,
                teardown=True,
            )
        )
        subscription = stack.enter_context(
            cm=MaaSSubscription(
                client=admin_client,
                name=subscription_name,
                namespace=maas_subscription_namespace.name,
                owner={"groups": [{"name": "system:authenticated"}]},
                model_refs=[
                    {
                        "name": model_name,
                        "namespace": namespace.name,
                        "tokenRateLimits": [{"limit": 10000, "window": "1m"}],
                    }
                ],
                token_metadata={"organizationId": organization_id},
                wait_for_resource=True,
                teardown=True,
            )
        )
        auth_policy.wait_for_condition(condition="Ready", status="True", timeout=300)
        subscription.wait_for_condition(condition="Ready", status="True", timeout=300)
        model_url = f"https://{maas_host}/{namespace.name}/{model_name}/v1/chat/completions"

        for capture_user, capture_organization in CAPTURE_STATES:
            with ResourceEditor(
                patches={
                    tenant_config: {
                        "spec": {
                            "telemetry": {
                                "enabled": True,
                                "logs": {
                                    "captureUser": capture_user,
                                    "captureOrganization": capture_organization,
                                },
                            }
                        }
                    }
                }
            ):
                wait_for_tenant_generation(tenant_config=tenant_config)
                key_name = generate_random_name(prefix="usage-capture-key")
                _, key = create_api_key(
                    base_url=base_url,
                    ocp_user_token=token,
                    request_session_http=request_session_http,
                    api_key_name=key_name,
                    subscription=subscription_name,
                )
                try:
                    start_ns = time_ns() - 5_000_000_000
                    response = poll_expected_status(
                        request_session_http=request_session_http,
                        model_url=model_url,
                        headers=build_maas_headers(token=key["key"]),
                        payload=chat_payload_for_url(model_url=model_url),
                        expected_statuses={200},
                    )
                    assert response.status_code == 200, "Governed inference did not succeed"
                    records = wait_for_usage_log(
                        client=loki_usage_client,
                        subscription=subscription_name,
                        key_name=key_name,
                        start_ns=start_ns,
                    )
                finally:
                    revoke_api_key(
                        request_session_http=request_session_http,
                        base_url=base_url,
                        key_id=key["id"],
                        ocp_user_token=token,
                    )

                for record in records:
                    user_id = record.get("user_id")
                    actual_organization = record.get("organization_id")
                    if capture_user:
                        assert user_id and user_id != "-", "Enabled logs.captureUser did not populate user_id"
                    else:
                        assert "user_id" not in record, "Disabled logs.captureUser still emitted user_id"
                    if capture_organization:
                        assert actual_organization == organization_id, (
                            "Enabled logs.captureOrganization did not populate the subscription organization_id"
                        )
                    else:
                        assert "organization_id" not in record, (
                            "Disabled logs.captureOrganization still emitted organization_id"
                        )
