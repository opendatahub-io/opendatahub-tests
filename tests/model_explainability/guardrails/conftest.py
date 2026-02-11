from typing import Generator, Any, List

import portforward
import pytest
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.open_telemetry_collector import OpenTelemetryCollector
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.subscription import Subscription
from ocp_resources.tempo_stack import TempoStack
from ocp_utilities.operators import uninstall_operator, install_operator
from timeout_sampler import TimeoutSampler

from tests.model_explainability.guardrails.constants import (
    AUTOCONFIG_DETECTOR_LABEL,
    TEMPO,
    SUPER_SECRET,
    OTEL_EXPORTER_PORT,
)
from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import KServeDeploymentType, RuntimeTemplates, Timeout
from utilities.inference_utils import create_isvc, LOGGER
from utilities.operator_utils import get_cluster_service_version
from utilities.serving_runtime import ServingRuntimeFromTemplate


GUARDRAILS_ORCHESTRATOR_NAME = "guardrails-orchestrator"


# ServingRuntimes, InferenceServices, and related resources
# for generation and detection models
@pytest.fixture(scope="class")
def huggingface_sr(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="guardrails-detector-runtime-prompt-injection",
        template_name=RuntimeTemplates.GUARDRAILS_DETECTOR_HUGGINGFACE,
        namespace=model_namespace.name,
        supported_model_formats=[{"name": "guardrails-detector-huggingface", "autoSelect": True}],
    ) as serving_runtime:
        yield serving_runtime


@pytest.fixture(scope="class")
def prompt_injection_detector_isvc(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    minio_data_connection: Secret,
    huggingface_sr: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name="prompt-injection-detector",
        namespace=model_namespace.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_format="guardrails-detector-huggingface",
        runtime=huggingface_sr.name,
        storage_key=minio_data_connection.name,
        storage_path="deberta-v3-base-prompt-injection-v2",
        wait_for_predictor_pods=False,
        enable_auth=False,
        resources={
            "requests": {"cpu": "1", "memory": "2Gi", "nvidia.com/gpu": "0"},
            "limits": {"cpu": "1", "memory": "2Gi", "nvidia.com/gpu": "0"},
        },
        max_replicas=1,
        min_replicas=1,
        labels={
            "opendatahub.io/dashboard": "true",
            AUTOCONFIG_DETECTOR_LABEL: "true",
        },
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def prompt_injection_detector_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    prompt_injection_detector_isvc: InferenceService,
) -> Generator[Route, Any, Any]:
    yield Route(
        name="prompt-injection-detector-route",
        namespace=model_namespace.name,
        service=prompt_injection_detector_isvc.name,
        wait_for_resource=True,
    )


# Other "helper" fixtures
@pytest.fixture(scope="class")
def openshift_ca_bundle_file(
    admin_client: DynamicClient,
) -> str:
    return create_ca_bundle_file(client=admin_client, ca_type="openshift")


@pytest.fixture(scope="class")
def hap_detector_isvc(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    minio_data_connection: Secret,
    huggingface_sr: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name="hap-detector",
        namespace=model_namespace.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_format="guardrails-detector-huggingface",
        runtime=huggingface_sr.name,
        storage_key=minio_data_connection.name,
        storage_path="granite-guardian-hap-38m",
        wait_for_predictor_pods=False,
        enable_auth=False,
        resources={
            "requests": {"cpu": "1", "memory": "4Gi", "nvidia.com/gpu": "0"},
            "limits": {"cpu": "1", "memory": "4Gi", "nvidia.com/gpu": "0"},
        },
        max_replicas=1,
        min_replicas=1,
        labels={
            "opendatahub.io/dashboard": "true",
            AUTOCONFIG_DETECTOR_LABEL: "true",
        },
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def hap_detector_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    hap_detector_isvc: InferenceService,
) -> Generator[Route, Any, Any]:
    yield Route(
        name="hap-detector-route",
        namespace=model_namespace.name,
        service=hap_detector_isvc.name,
        wait_for_resource=True,
    )


@pytest.fixture(scope="class")
def installed_tempo_operator(admin_client: DynamicClient, model_namespace: Namespace) -> Generator[None, Any, None]:
    """
    Installs the Tempo operator and waits for its deployment.
    """

    operator_ns_name = "openshift-tempo-operator"
    operator_ns = Namespace(name=operator_ns_name)
    if not operator_ns.exists:
        operator_ns.create()

    package_name = "tempo-product"
    tempo_operator_subscription = Subscription(client=admin_client, namespace=operator_ns.name, name=package_name)

    if not tempo_operator_subscription.exists:
        install_operator(
            admin_client=admin_client,
            target_namespaces=None,
            name=package_name,
            channel="stable",
            source="redhat-operators",
            operator_namespace=operator_ns.name,
            timeout=Timeout.TIMEOUT_15MIN,
            install_plan_approval="Automatic",
            starting_csv="tempo-operator.v0.19.0-2",
        )

        deployment = Deployment(
            client=admin_client,
            namespace=operator_ns.name,
            name="tempo-operator-controller",
            wait_for_resource=True,
        )
        deployment.wait_for_replicas()

        yield

        uninstall_operator(
            admin_client=admin_client,
            name=package_name,
            operator_namespace=operator_ns.name,
            clean_up_namespace=False,
        )
    else:
        yield


@pytest.fixture(scope="class")
def tempo_stack(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    minio_secret_otel: Secret,
) -> Generator[Any, Any, None]:
    """
    Create a TempoStack CR in the test namespace, configured to use MinIO backend.
    """
    csv_prefix = "tempo-operator"
    tempo_name = "my-tempo-stack"

    # Get the installed Tempo operator CSV
    tempo_csv: ClusterServiceVersion = get_cluster_service_version(
        client=admin_client,
        prefix=csv_prefix,
        namespace="openshift-tempo-operator",
    )

    # Retrieve ALM examples and pick TempoStack CR
    alm_examples: list[dict[str, Any]] = tempo_csv.get_alm_examples()
    tempo_stack_dict = next(
        (
            example
            for example in alm_examples
            if example["kind"] == "TempoStack" and example["apiVersion"].startswith("tempo.grafana.com/")
        ),
        None,
    )
    if not tempo_stack_dict:
        raise ResourceNotFoundError(f"No TempoStack dict found in ALM examples for CSV {tempo_csv.name}")

    # Customize metadata
    tempo_stack_dict["metadata"]["namespace"] = model_namespace.name
    tempo_stack_dict["metadata"]["name"] = tempo_name

    # Override spec with MinIO backend and resource constraints
    tempo_stack_dict["spec"]["storage"] = {
        "secret": {
            "name": minio_secret_otel.name,
            "type": "s3",
        }
    }
    tempo_stack_dict["spec"]["storageSize"] = "1Gi"
    tempo_stack_dict["spec"]["resources"] = {
        "total": {
            "limits": {"memory": "2Gi", "cpu": "2000m"},
        }
    }
    tempo_stack_dict["spec"]["template"] = {
        "queryFrontend": {"jaegerQuery": {"enabled": True}},
    }

    with TempoStack(kind_dict=tempo_stack_dict) as tempo_cr:
        tempo_cr.wait_for_condition(
            condition="Ready",
            status="True",
            timeout=Timeout.TIMEOUT_10MIN,
        )

        yield tempo_cr


@pytest.fixture(scope="class")
def installed_opentelemetry_operator(admin_client: DynamicClient) -> Generator[None, Any, None]:
    """
    Installs the Red Hat OpenTelemetry Operator and waits for its deployment.
    """
    operator_namespace = "openshift-opentelemetry-operator"
    operator_ns = Namespace(name=operator_namespace)
    if not operator_ns.exists:
        operator_ns.create()

    package_name = "opentelemetry-product"

    opentelemetry_subscription = Subscription(client=admin_client, namespace=operator_ns.name, name=package_name)

    if not opentelemetry_subscription.exists:
        install_operator(
            admin_client=admin_client,
            target_namespaces=None,
            name=package_name,
            channel="stable",
            source="redhat-operators",
            operator_namespace=operator_ns.name,
            timeout=Timeout.TIMEOUT_15MIN,
            install_plan_approval="Automatic",
            starting_csv="opentelemetry-operator.v0.140.0-1",
        )

        deployment = Deployment(
            client=admin_client,
            namespace=operator_ns.name,
            name="opentelemetry-operator-controller-manager",
            wait_for_resource=True,
        )
        deployment.wait_for_replicas()

        yield

        uninstall_operator(
            admin_client=admin_client,
            name=package_name,
            operator_namespace=operator_ns.name,
            clean_up_namespace=False,
        )
    else:
        yield


@pytest.fixture(scope="class")
def otel_collector(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    minio_service_otel,
) -> Generator[OpenTelemetryCollector, Any, Any]:
    """
    Create an OpenTelemetryCollector CR in the test namespace.
    Dynamically uses the Operator CSV example and adjusts configuration for Tempo.
    """
    # Get the OTel Operator CSV
    otel_csv: ClusterServiceVersion = get_cluster_service_version(
        client=admin_client,
        prefix="opentelemetry",
        namespace="openshift-opentelemetry-operator",
    )

    # Extract OpenTelemetryCollector CR example from ALM examples
    alm_examples: list[dict[str, Any]] = otel_csv.get_alm_examples()
    otel_cr_dict: dict[str, Any] = next(
        example
        for example in alm_examples
        if example["kind"] == "OpenTelemetryCollector" and example["apiVersion"] == "opentelemetry.io/v1beta1"
    )

    if not otel_cr_dict:
        raise ResourceNotFoundError(f"No OpenTelemetryCollector example found in ALM examples for {otel_csv.name}")

    # Update the metadata and spec to match test namespace and Tempo endpoint
    namespace = model_namespace.name
    otel_cr_dict["metadata"]["namespace"] = namespace
    otel_cr_dict["metadata"]["name"] = "my-otelcol"

    # Override the Tempo exporter endpoint (ensures proper linkage)
    otel_cr_dict["spec"]["config"] = {
        "exporters": {
            "otlp": {
                "endpoint": (
                    f"tempo-{tempo_stack.name}-distributor.{namespace}.svc.cluster.local:{OTEL_EXPORTER_PORT}"
                ),
                "tls": {"insecure": True},
            }
        },
        "receivers": {
            "otlp": {
                "protocols": {
                    "grpc": {"endpoint": "0.0.0.0:4317"},
                    "http": {"endpoint": "0.0.0.0:4318"},
                }
            }
        },
        "service": {
            "pipelines": {
                "traces": {
                    "exporters": ["otlp"],
                    "receivers": ["otlp"],
                }
            },
            "telemetry": {
                "metrics": {"readers": [{"pull": {"exporter": {"prometheus": {"host": "0.0.0.0", "port": 8888}}}}]}
            },
        },
    }
    otel_cr_dict["spec"]["mode"] = "deployment"

    with OpenTelemetryCollector(kind_dict=otel_cr_dict) as otel_cr:
        wait_for_pods_by_label(
            client=admin_client,
            namespace=namespace,
            label_selector="app.kubernetes.io/component=opentelemetry-collector",
        )
        yield otel_cr


def wait_for_pods_by_label(
    client: DynamicClient,
    namespace: str,
    label_selector: str,
    timeout: int = Timeout.TIMEOUT_15MIN,
) -> None:
    """
    Wait for all pods with a specific label selector in a namespace to be ready.

    Args:
        client: Dynamic client
        namespace: Namespace to search for pods
        label_selector: Label selector to filter pods
        timeout: Maximum wait time in seconds
    """

    def _get_pods() -> List[Pod]:
        return [
            pod
            for pod in Pod.get(
                client=client,
                namespace=namespace,
                label_selector=label_selector,
            )
        ]

    sampler = TimeoutSampler(wait_timeout=timeout, sleep=1, func=lambda: bool(_get_pods()))

    for sample in sampler:
        if sample:
            break

    pods = _get_pods()
    for pod in pods:
        pod.wait_for_condition(
            condition=Pod.Condition.READY,
            status="True",
        )


@pytest.fixture(scope="class")
def minio_pvc_otel(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    """
    Creates a PVC for MinIO storage backend in the given namespace.
    """
    pvc_kwargs = {
        "name": "minio",
        "namespace": model_namespace.name,
        "client": admin_client,
        "size": "2Gi",
        "accessmodes": "ReadWriteOnce",
        "label": {"app.kubernetes.io/name": "minio"},
    }

    with PersistentVolumeClaim(**pvc_kwargs) as pvc:
        yield pvc


@pytest.fixture(scope="class")
def minio_deployment_otel(admin_client, model_namespace, minio_pvc_otel):
    selector = {"matchLabels": {"app.kubernetes.io/name": "minio"}}
    pod_template = {
        "metadata": {"labels": {"app.kubernetes.io/name": "minio"}},
        "spec": {
            "containers": [
                {
                    "name": "minio",
                    "image": "quay.io/minio/minio",
                    "command": ["/bin/sh", "-c", "mkdir -p /storage/tempo && minio server /storage"],
                    "env": [
                        {"name": "MINIO_ACCESS_KEY", "value": TEMPO},
                        {"name": "MINIO_SECRET_KEY", "value": SUPER_SECRET},
                    ],
                    "ports": [{"containerPort": 9000}],
                    "volumeMounts": [{"mountPath": "/storage", "name": "storage"}],
                }
            ],
            "volumes": [
                {
                    "name": "storage",
                    "persistentVolumeClaim": {"claimName": "minio"},
                }
            ],
        },
    }

    deployment = Deployment(
        client=admin_client,
        name="minio",
        namespace=model_namespace.name,
        selector=selector,
        template=pod_template,
        strategy={"type": "Recreate"},
        teardown=True,
    )

    with deployment:
        deployment.wait_for_replicas()
        yield deployment


@pytest.fixture(scope="class")
def minio_service_otel(admin_client, model_namespace, minio_deployment_otel):
    ports = [
        {
            "port": 9000,
            "protocol": "TCP",
            "targetPort": 9000,
        }
    ]

    selector = {
        "app.kubernetes.io/name": "minio",
    }

    service = Service(
        client=admin_client,
        name="minio",
        namespace=model_namespace.name,
        ports=ports,
        selector=selector,
        type="ClusterIP",
        teardown=True,
    )
    service.deploy()
    yield service


@pytest.fixture(scope="class")
def minio_secret_otel(admin_client, model_namespace, minio_service_otel):
    secret = Secret(
        client=admin_client,
        name="minio-test",
        namespace=model_namespace.name,
        string_data={
            "endpoint": f"http://{minio_service_otel.name}.{model_namespace.name}.svc.cluster.local:9000",
            "bucket": TEMPO,
            "access_key_id": TEMPO,  # pragma: allowlist secret
            "access_key_secret": SUPER_SECRET,  # pragma: allowlist secret
        },
        type="Opaque",
        teardown=True,
    )
    secret.deploy()
    yield secret


@pytest.fixture(scope="class")
def otelcol_metrics_endpoint(admin_client: DynamicClient, model_namespace: Namespace):
    """
    Returns the metrics endpoint for the OpenTelemetryCollector by grepping the service name.
    """

    service = next(
        Service.get(
            client=admin_client,
            namespace=model_namespace.name,
            label_selector="app.kubernetes.io/component=opentelemetry-collector",
        )
    )

    service_name = service.name

    port = OTEL_EXPORTER_PORT
    return f"http://{service_name}.{model_namespace.name}.svc.cluster.local:{port}"


@pytest.fixture(scope="class")
def tempo_traces_endpoint(tempo_stack, model_namespace: Namespace, admin_client: DynamicClient):
    """
    Returns the TempoStack distributor endpoint dynamically using ocp_resources.Service.

    """
    service_name = f"tempo-{tempo_stack.name}-distributor"

    port = OTEL_EXPORTER_PORT

    return f"http://{service_name}.{model_namespace.name}.svc.cluster.local:{port}"


@pytest.fixture(scope="class")
def tempo_traces_service_portforward(admin_client, model_namespace, tempo_stack):
    """
    Port-forwards the Tempo Query Frontend service to access traces locally.
    Equivalent CLI:
      oc -n <ns> port-forward svc/tempo-my-tempo-stack-query-frontend 16686:16686
    """
    namespace = model_namespace.name
    service_name = f"tempo-{tempo_stack.name}-query-frontend"
    local_port = 16686
    local_url = f"http://localhost:{local_port}"

    try:
        with portforward.forward(
            pod_or_service=f"{service_name}",
            namespace=namespace,
            from_port=local_port,
            to_port=local_port,
            waiting=20,
        ):
            LOGGER.info(f"Tempo traces service port-forward established: {local_url}")
            yield local_url
    except Exception as e:
        LOGGER.error(f"Failed to set up port forwarding for {service_name}: {e}")
        raise
