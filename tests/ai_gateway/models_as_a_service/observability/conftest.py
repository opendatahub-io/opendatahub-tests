from collections.abc import Generator
from typing import Any

import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.dsc_initialization import DSCInitialization
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.service_monitor import ServiceMonitor
from ocp_utilities.monitoring import Prometheus

from tests.ai_gateway.models_as_a_service.observability.constants import (
    LIMITADOR_SCRAPE_INTERVAL_TEST_VALUE,
    LIMITADOR_SERVICE_MONITOR_NAME,
    LOKI_S3_BUCKET_NAME,
    LOKI_S3_REGION,
    LOKI_S3_SECRET_NAME,
    LOKI_STACK_CRD_NAME,
    OTEL_COLLECTOR_CRD_NAME,
    RHOAI_THANOS_QUERIER_ROUTE_NAME,
    SERVICE_MONITOR_CRD_NAME,
    USAGE_LOGS_SEAWEEDFS_CREDENTIALS_SECRET_NAME,
    USAGE_LOGS_SEAWEEDFS_FILER_CONFIG_NAME,
    USAGE_LOGS_SEAWEEDFS_NAME,
)
from tests.ai_gateway.models_as_a_service.observability.utils import (
    get_maas_config_default,
    limitador_is_deployed,
    maas_config_usage_logging_disabled,
    maas_config_usage_logging_enabled,
    monitoring_namespace_exists,
    opentelemetry_collector_crd_installed,
    patch_maas_config_limitador_scrape_interval,
    resolve_maas_monitoring_namespace,
    usage_logs_monitoring_spec,
    wait_for_dsci_usage_logs_conditions,
    wait_for_limitador_service_monitor,
)
from tests.ai_gateway.models_as_a_service.utils import (
    build_maas_headers,
    create_api_key,
    get_maas_models_response,
    revoke_api_key,
)
from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import Protocols, SeaweedFs
from utilities.general import generate_random_name
from utilities.image_constants import SharedImages
from utilities.infra import (
    create_ns,
    get_default_storage_class_name,
    get_dsci_monitoring_namespace,
    get_openshift_token,
    update_dsci_monitoring_spec,
)
from utilities.plugins.constant import OpenAIEnpoints
from utilities.resources.llm_inference_service import LLMInferenceService
from utilities.resources.maas_config import Config as MaaSConfig


@pytest.fixture(scope="session")
def maas_monitoring_namespace(
    admin_client: DynamicClient,
    maas_subscription_controller_enabled_latest: None,
) -> str:
    """Return the monitoring namespace configured for maas-controller observability."""
    monitoring_namespace = resolve_maas_monitoring_namespace(admin_client=admin_client)
    if not monitoring_namespace:
        pytest.fail("maas-controller MONITORING_NAMESPACE is not configured; observability reconcile cannot run")
    if not monitoring_namespace_exists(admin_client=admin_client, namespace_name=monitoring_namespace):
        pytest.fail(
            f"Monitoring namespace '{monitoring_namespace}' does not exist; "
            "maas-controller observability setup requires it"
        )
    return monitoring_namespace


@pytest.fixture(scope="session")
def servicemonitor_crd_available(admin_client: DynamicClient) -> None:
    """Fail when the Prometheus Operator ServiceMonitor CRD is not installed."""
    service_monitor_crd = CustomResourceDefinition(
        client=admin_client,
        name=SERVICE_MONITOR_CRD_NAME,
    )
    if not service_monitor_crd.exists:
        pytest.fail(
            f"ServiceMonitor CRD '{SERVICE_MONITOR_CRD_NAME}' not installed; Limitador metrics scrape requires it"
        )


@pytest.fixture(scope="session")
def opentelemetry_collector_crd_available(admin_client: DynamicClient) -> None:
    """Fail when the OpenTelemetryCollector CRD is not installed."""
    if not opentelemetry_collector_crd_installed(admin_client=admin_client):
        pytest.fail(f"{OTEL_COLLECTOR_CRD_NAME} not installed; usage-logs collector bundle requires it")


@pytest.fixture
def maas_config_with_usage_logging_enabled(
    maas_config_default: MaaSConfig,
) -> Generator[MaaSConfig]:
    """Enable usageLogging on Config/default for the test and restore on teardown."""
    with maas_config_usage_logging_enabled(maas_config=maas_config_default):
        yield maas_config_default


@pytest.fixture
def maas_config_with_limitador_scrape_interval_patched(
    maas_config_default: MaaSConfig,
) -> Generator[MaaSConfig]:
    """Patch Config/default limitadorScrapeInterval for the test and restore on teardown."""
    with patch_maas_config_limitador_scrape_interval(
        maas_config=maas_config_default,
        scrape_interval=LIMITADOR_SCRAPE_INTERVAL_TEST_VALUE,
    ):
        yield maas_config_default


@pytest.fixture(scope="session")
def limitador_service_monitor(
    admin_client: DynamicClient,
    maas_monitoring_namespace: str,
    servicemonitor_crd_available: None,
) -> ServiceMonitor:
    """Wait for maas-controller to create the Limitador ServiceMonitor."""
    limitador_service_monitor = wait_for_limitador_service_monitor(
        admin_client=admin_client,
        monitoring_namespace=maas_monitoring_namespace,
    )
    if limitador_service_monitor is None:
        pytest.fail(
            f"ServiceMonitor '{LIMITADOR_SERVICE_MONITOR_NAME}' not found in "
            f"'{maas_monitoring_namespace}' after maas-controller reconcile"
        )
    return limitador_service_monitor


@pytest.fixture(scope="session")
def maas_observability_prometheus(
    admin_client: DynamicClient,
    maas_monitoring_namespace: str,
) -> Prometheus:
    """Return a Prometheus client for the RHOAI observability Thanos querier route."""
    thanos_route = Route(
        client=admin_client,
        name=RHOAI_THANOS_QUERIER_ROUTE_NAME,
        namespace=maas_monitoring_namespace,
    )
    if not thanos_route.exists:
        pytest.fail(
            f"Route '{RHOAI_THANOS_QUERIER_ROUTE_NAME}' not found in '{maas_monitoring_namespace}'; "
            "RHOAI observability stack is not deployed"
        )
    return Prometheus(
        client=admin_client,
        namespace=maas_monitoring_namespace,
        resource_name=RHOAI_THANOS_QUERIER_ROUTE_NAME,
        verify_ssl=create_ca_bundle_file(client=admin_client),
        bearer_token=get_openshift_token(),
    )


@pytest.fixture(scope="session")
def limitador_deployed(
    admin_client: DynamicClient,
) -> None:
    """Fail when Limitador is not running in a known policy-engine namespace."""
    if not limitador_is_deployed(admin_client=admin_client):
        pytest.fail(
            "Limitador is not deployed in kuadrant-system or rh-connectivity-link; "
            "Limitador Prometheus metrics smoke test requires it"
        )


@pytest.fixture(scope="class")
def usage_logs_crds_available(admin_client: DynamicClient) -> None:
    """Skip when the CRDs required by the usage-logs LokiStack are not installed."""
    for crd_name in (LOKI_STACK_CRD_NAME, OTEL_COLLECTOR_CRD_NAME):
        if not CustomResourceDefinition(client=admin_client, name=crd_name).exists:
            pytest.skip(f"CRD '{crd_name}' not installed; usage-logs LokiStack requires it")


@pytest.fixture(scope="class")
def dsci_monitoring_namespace(
    admin_client: DynamicClient,
    dsci_resource: DSCInitialization,
    usage_logs_crds_available: None,
) -> Generator[str, Any, Any]:
    """Monitoring namespace for usage-logs resources; created when missing and left in place."""
    monitoring_namespace = get_dsci_monitoring_namespace(client=admin_client)
    if Namespace(client=admin_client, name=monitoring_namespace).exists:
        yield monitoring_namespace
        return

    # The namespace is shared with the async LokiStack teardown, so it is never deleted here.
    with create_ns(name=monitoring_namespace, admin_client=admin_client, teardown=False) as namespace:
        yield namespace.name


@pytest.fixture(scope="class")
def usage_logs_seaweedfs_filer_config(
    admin_client: DynamicClient,
    dsci_monitoring_namespace: str,
) -> Generator[ConfigMap, Any, Any]:
    """Configure SeaweedFS filer metadata storage on the writable temporary filesystem."""
    with ConfigMap(
        client=admin_client,
        name=USAGE_LOGS_SEAWEEDFS_FILER_CONFIG_NAME,
        namespace=dsci_monitoring_namespace,
        data={"filer.toml": '[leveldb2]\nenabled = true\ndir = "/tmp/filerldb2"\n'},
    ) as config_map:
        yield config_map


@pytest.fixture(scope="class")
def usage_logs_seaweedfs_credential_values() -> tuple[str, str]:
    """Generate a unique (access key, secret key) pair for the test class' SeaweedFS instance."""
    return (
        generate_random_name(prefix="usage-logs-access-key"),
        generate_random_name(prefix="usage-logs-secret-key"),
    )


@pytest.fixture(scope="class")
def usage_logs_seaweedfs_credentials(
    admin_client: DynamicClient,
    dsci_monitoring_namespace: str,
    usage_logs_seaweedfs_credential_values: tuple[str, str],
) -> Generator[Secret, Any, Any]:
    """SeaweedFS server credentials in the monitoring namespace."""
    access_key, secret_key = usage_logs_seaweedfs_credential_values
    with Secret(
        client=admin_client,
        name=USAGE_LOGS_SEAWEEDFS_CREDENTIALS_SECRET_NAME,
        namespace=dsci_monitoring_namespace,
        string_data={
            "accesskey": access_key,
            "secretkey": secret_key,
        },
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def usage_logs_seaweedfs_pod(
    admin_client: DynamicClient,
    dsci_monitoring_namespace: str,
    usage_logs_seaweedfs_filer_config: ConfigMap,
    usage_logs_seaweedfs_credentials: Secret,
) -> Generator[Pod, Any, Any]:
    """Run an authenticated SeaweedFS instance exposing its S3 API, with the usage-logs bucket created."""
    initialization_command = (
        "for attempt in $(seq 1 60); do "
        "wget -q --spider http://127.0.0.1:8333/status && break; "
        '[ "$attempt" -eq 60 ] && exit 1; sleep 2; '
        "done; "
        'echo "s3.configure -user admin -access_key $accesskey -secret_key $secretkey -actions Admin -apply" '
        "| /usr/bin/weed shell && "
        f"echo 's3.bucket.create -name {LOKI_S3_BUCKET_NAME}' | /usr/bin/weed shell"
    )
    with Pod(
        client=admin_client,
        name=USAGE_LOGS_SEAWEEDFS_NAME,
        namespace=dsci_monitoring_namespace,
        containers=[
            {
                "name": USAGE_LOGS_SEAWEEDFS_NAME,
                "image": SharedImages.SEAWEEDFS,
                "args": list(SeaweedFs.PodConfig.ARGS),
                "env": [
                    {
                        "name": "accesskey",
                        "valueFrom": {
                            "secretKeyRef": {
                                "name": usage_logs_seaweedfs_credentials.name,
                                "key": "accesskey",
                            },
                        },
                    },
                    {
                        "name": "secretkey",
                        "valueFrom": {
                            "secretKeyRef": {
                                "name": usage_logs_seaweedfs_credentials.name,
                                "key": "secretkey",
                            },
                        },
                    },
                ],
                "ports": [
                    {"containerPort": SeaweedFs.Metadata.DEFAULT_PORT, "name": "s3"},
                    {"containerPort": SeaweedFs.Metadata.FILER_PORT, "name": "filer"},
                    {"containerPort": SeaweedFs.Metadata.FILER_GRPC_PORT, "name": "filer-grpc"},
                ],
                "volumeMounts": [
                    {
                        "name": "filer-config",
                        "mountPath": "/etc/seaweedfs/filer.toml",
                        "subPath": "filer.toml",
                        "readOnly": True,
                    }
                ],
                "lifecycle": {"postStart": {"exec": {"command": ["/bin/sh", "-c", initialization_command]}}},
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                    "runAsNonRoot": True,
                    "seccompProfile": {"type": "RuntimeDefault"},
                },
            }
        ],
        volumes=[
            {"name": "filer-config", "configMap": {"name": usage_logs_seaweedfs_filer_config.name}},
        ],
        label={"app": USAGE_LOGS_SEAWEEDFS_NAME},
    ) as pod:
        # The pod only becomes Ready after the postStart bucket-creation command completes.
        pod.wait_for_condition(condition="Ready", status="True", timeout=180)
        yield pod


@pytest.fixture(scope="class")
def usage_logs_seaweedfs_service(
    admin_client: DynamicClient,
    dsci_monitoring_namespace: str,
    usage_logs_seaweedfs_pod: Pod,
) -> Generator[Service, Any, Any]:
    """Expose the SeaweedFS S3 API inside the monitoring namespace."""
    with Service(
        client=admin_client,
        name=USAGE_LOGS_SEAWEEDFS_NAME,
        namespace=dsci_monitoring_namespace,
        ports=[
            {
                "name": "s3",
                "port": SeaweedFs.Metadata.DEFAULT_PORT,
                "protocol": Protocols.TCP,
                "targetPort": SeaweedFs.Metadata.DEFAULT_PORT,
            },
        ],
        selector={"app": USAGE_LOGS_SEAWEEDFS_NAME},
    ) as service:
        yield service


@pytest.fixture(scope="class")
def usage_logs_loki_s3_secret(
    admin_client: DynamicClient,
    dsci_monitoring_namespace: str,
    usage_logs_seaweedfs_service: Service,
    usage_logs_seaweedfs_credential_values: tuple[str, str],
) -> Generator[Secret, Any, Any]:
    """LokiStack S3 storage secret in Loki format, pointing at the in-cluster SeaweedFS backend."""
    access_key, secret_key = usage_logs_seaweedfs_credential_values
    with Secret(
        client=admin_client,
        name=LOKI_S3_SECRET_NAME,
        namespace=dsci_monitoring_namespace,
        string_data={
            "access_key_id": access_key,
            "access_key_secret": secret_key,
            "bucketnames": LOKI_S3_BUCKET_NAME,
            "endpoint": (
                f"http://{usage_logs_seaweedfs_service.name}.{dsci_monitoring_namespace}.svc.cluster.local"
                f":{SeaweedFs.Metadata.DEFAULT_PORT}"
            ),
            "region": LOKI_S3_REGION,
            "insecure": "true",
            "s3ForcePathStyle": "true",
        },
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def dsci_usage_logs_monitoring(
    admin_client: DynamicClient,
    dsci_resource: DSCInitialization,
    usage_logs_loki_s3_secret: Secret,
) -> Generator[DSCInitialization, Any, Any]:
    """Patch DSCI spec.monitoring to enable usage logs and restore the original spec on teardown."""
    monitoring_spec = usage_logs_monitoring_spec(
        secret_name=usage_logs_loki_s3_secret.name,
        storage_class_name=get_default_storage_class_name(client=admin_client),
    )
    with update_dsci_monitoring_spec(
        dsci_resource=dsci_resource,
        monitoring_spec=monitoring_spec,
    ):
        yield dsci_resource


@pytest.fixture(scope="class")
def usage_logs_monitoring_ready(
    dsci_usage_logs_monitoring: DSCInitialization,
) -> None:
    """Wait for DSCI to report the odh-observability usage-logs conditions as True."""
    wait_for_dsci_usage_logs_conditions(dsci_resource=dsci_usage_logs_monitoring)


@pytest.fixture(scope="class")
def usage_logs_model_url(
    maas_scheme: str,
    maas_host: str,
    maas_inference_service_tinyllama_modelcar: LLMInferenceService,
) -> str:
    """Chat-completions URL of the modelcar-backed MaaS model."""
    deployment = maas_inference_service_tinyllama_modelcar.name
    return f"{maas_scheme}://{maas_host}/llm/{deployment}{OpenAIEnpoints.CHAT_COMPLETIONS}"


@pytest.fixture(scope="class")
def usage_logs_maas_api_key(
    request_session_http: requests.Session,
    base_url: str,
    current_client_token: str,
    maas_api_gateway_reachable: None,
) -> Generator[str, Any, Any]:
    """Create a MaaS API key for the usage-logs test once per test class. Revoked on teardown.

    Uses POST /v1/api-keys rather than minted_token's POST /v1/tokens: the latter
    was removed from maas-api (opendatahub-io/models-as-a-service commit 535fab34,
    "feat: api-key management (#406)") in favor of /v1/api-keys.
    """
    _, body = create_api_key(
        base_url=base_url,
        ocp_user_token=current_client_token,
        request_session_http=request_session_http,
        api_key_name=f"e2e-usage-logs-{generate_random_name()}",
    )
    yield str(body["key"])
    revoke_response, _ = revoke_api_key(
        request_session_http=request_session_http,
        base_url=base_url,
        key_id=body["id"],
        ocp_user_token=current_client_token,
    )
    assert revoke_response.status_code in (200, 404), (
        f"Unexpected revocation status for key id={body['id']}: {revoke_response.status_code}"
    )


@pytest.fixture(scope="class")
def usage_logs_maas_headers(usage_logs_maas_api_key: str) -> dict:
    return build_maas_headers(token=usage_logs_maas_api_key)


@pytest.fixture(scope="class")
def usage_logs_maas_models(
    request_session_http: requests.Session,
    base_url: str,
    usage_logs_maas_headers: dict,
    maas_inference_service_tinyllama_modelcar: LLMInferenceService,
    maas_gateway_api: None,
    maas_request_ratelimit_policy: None,
    maas_token_ratelimit_policy: None,
) -> list[dict]:
    """Models listed by the MaaS gateway once the modelcar-backed model is deployed."""
    resp = get_maas_models_response(
        session=request_session_http,
        base_url=base_url,
        headers=usage_logs_maas_headers,
    )
    models = resp.json().get("data", [])
    assert models, "no models available"
    return models


@pytest.fixture
def maas_config_with_usage_logging_disabled(
    admin_client: DynamicClient,
    maas_controller_enabled_latest: DataScienceCluster,
) -> Generator[MaaSConfig]:
    """Ensure usageLogging is disabled on Config/default for the test and restore on teardown."""
    maas_config = get_maas_config_default(admin_client=admin_client)
    with maas_config_usage_logging_disabled(maas_config=maas_config):
        yield maas_config
