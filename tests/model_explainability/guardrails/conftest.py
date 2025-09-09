from typing import Generator, Any

import pytest
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.jaeger import Jaeger
from ocp_resources.namespace import Namespace
from ocp_resources.open_telemetry_collector import OpenTelemetryCollector
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from ocp_utilities.operators import install_operator, uninstall_operator
from timeout_sampler import TimeoutSampler

from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import (
    KServeDeploymentType,
    Timeout,
    OPENSHIFT_OPERATORS,
    RuntimeTemplates,
)
from utilities.inference_utils import create_isvc
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
def installed_opentelemetry_operator(admin_client: DynamicClient) -> Generator[None, Any, None]:
    """
    Installs the Red Hat OpenTelemetry Operator and waits for its deployment.
    """
    operator_ns = Namespace(name="openshift-operators", ensure_exists=True)

    package_name = "opentelemetry-product"

    install_operator(
        admin_client=admin_client,
        target_namespaces=[operator_ns.name],
        name=package_name,
        channel="stable",
        source="redhat-operators",
        operator_namespace=operator_ns.name,
        timeout=Timeout.TIMEOUT_15MIN,
        install_plan_approval="Automatic",
        starting_csv="opentelemetry-operator.v0.127.0-2",
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


@pytest.fixture(scope="class")
def otel_operator_cr(
    admin_client: DynamicClient,
    installed_opentelemetry_operator: None,
    model_namespace: Namespace,
) -> Generator[OpenTelemetryCollector, Any, Any]:
    """Create an OpenTelemetryCollector CR in the test namespace."""
    otel_csv: ClusterServiceVersion = get_cluster_service_version(
        client=admin_client,
        prefix="opentelemetry",
        namespace=OPENSHIFT_OPERATORS,
    )

    alm_examples: list[dict[str, Any]] = otel_csv.get_alm_examples()
    otel_cr_dict: dict[str, Any] = next(
        example
        for example in alm_examples
        if example["kind"] == "OpenTelemetryCollector" and example["apiVersion"] == "opentelemetry.io/v1beta1"
    )

    if not otel_cr_dict:
        raise ResourceNotFoundError(f"No OpenTelemetryCollector dict found in alm_examples for CSV {otel_csv.name}")

    otel_cr_dict["metadata"]["namespace"] = model_namespace.name

    with OpenTelemetryCollector(kind_dict=otel_cr_dict) as otel_cr:
        wait_for_collector_pods(
            admin_client,
            namespace=model_namespace.name,
        )
        yield otel_cr


@pytest.fixture(scope="class")
def installed_jaeger_operator(admin_client: DynamicClient, model_namespace: Namespace) -> Generator[None, Any, None]:
    """
    Installs the Jaeger operator and waits for its deployment.
    """
    operator_ns = Namespace(name="openshift-operators", ensure_exists=True)
    package_name = "jaeger-product"

    install_operator(
        admin_client=admin_client,
        target_namespaces=[operator_ns.name],
        name=package_name,
        channel="stable",
        source="redhat-operators",
        operator_namespace=operator_ns.name,
        timeout=Timeout.TIMEOUT_15MIN,
        install_plan_approval="Automatic",
        starting_csv="jaeger-operator.v1.65.0-4",
    )

    deployment = Deployment(
        client=admin_client,
        namespace=operator_ns.name,
        name="jaeger-operator",
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


@pytest.fixture(scope="class")
def jaeger_instance(
    admin_client: DynamicClient,
    installed_jaeger_operator: None,
    model_namespace: Namespace,
) -> Generator[Jaeger, Any, None]:
    """Create a Jaeger instance in the test namespace."""

    # The CSV name is jaeger-operator.v1.65.0-4, so the prefix should be `jaeger-operator`.
    csv_prefix = "jaeger-operator"

    jaeger_csv: ClusterServiceVersion = get_cluster_service_version(
        client=admin_client,
        prefix=csv_prefix,
        namespace=OPENSHIFT_OPERATORS,
    )

    alm_examples: list[dict[str, Any]] = jaeger_csv.get_alm_examples()
    jaeger_dict: dict[str, Any] = next(example for example in alm_examples if example["kind"] == "Jaeger")

    if not jaeger_dict:
        raise ResourceNotFoundError(f"No Jaeger dict found in alm_examples for CSV {jaeger_csv.name}")

    jaeger_dict["metadata"]["namespace"] = model_namespace.name
    jaeger_dict["metadata"]["name"] = "my-jaeger"

    with Jaeger(kind_dict=jaeger_dict) as jaeger:
        wait_for_jaeger_pods(
            client=admin_client,
            jaeger_name=jaeger.name,
            namespace=model_namespace.name,
        )
        yield jaeger


def wait_for_jaeger_operator_deployments(namespace: str) -> None:
    """
    Wait for the Jaeger operator deployment to be ready.
    """
    operator_deployment_name = "jaeger-operator"

    deployment = Deployment(name=operator_deployment_name, namespace=namespace)
    deployment.wait_for_replicas()


def wait_for_jaeger_pods(
    client: DynamicClient,
    jaeger_name: str,
    namespace: str,
    timeout: int = Timeout.TIMEOUT_15MIN,
) -> None:
    """
    Wait for pods created by a Jaeger instance to be ready.
    """

    def _get_jaeger_pods() -> list[Pod]:
        return [
            _pod
            for _pod in Pod.get(
                dyn_client=client,
                namespace=namespace,
                label_selector=f"app.kubernetes.io/instance={jaeger_name}",
            )
        ]

    sampler = TimeoutSampler(wait_timeout=timeout, sleep=1, func=lambda: bool(_get_jaeger_pods()))

    for sample in sampler:
        if sample:
            break

    pods = _get_jaeger_pods()
    for pod in pods:
        pod.wait_for_condition(
            condition=Pod.Condition.READY,
            status="True",
        )


def wait_for_collector_pods(
    client: DynamicClient,
    namespace: str,
    timeout: int = Timeout.TIMEOUT_15MIN,
) -> None:
    """
    Wait for pods created by a collector instance to be ready.
    """

    def _get_collector_pods() -> list[Pod]:
        pods = [
            _pod
            for _pod in Pod.get(
                dyn_client=client,
                namespace=namespace,
                label_selector="app.kubernetes.io/component=opentelemetry-collector",
            )
        ]
        return pods

    sampler = TimeoutSampler(wait_timeout=timeout, sleep=1, func=lambda: bool(_get_collector_pods()))

    for sample in sampler:
        if sample:
            break

    pods = _get_collector_pods()
    for pod in pods:
        pod.wait_for_condition(
            condition=Pod.Condition.READY,
            status="True",
        )
