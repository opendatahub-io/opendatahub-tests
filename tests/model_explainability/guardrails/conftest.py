import os
import subprocess
from base64 import b64encode
from typing import Generator, Any

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.guardrails_orchestrator import GuardrailsOrchestrator
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor, NamespacedResource
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.subscription import Subscription
from ocp_utilities.operators import install_operator, uninstall_operator
from pytest_testconfig import py_config
from timeout_sampler import TimeoutSampler

from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import (
    KServeDeploymentType,
    Labels,
    Timeout,
)
from utilities.inference_utils import create_isvc
from utilities.operator_utils import get_cluster_service_version

GUARDRAILS_ORCHESTRATOR_NAME = "guardrails-orchestrator"


# GuardrailsOrchestrator related fixtures
@pytest.fixture(scope="class")
def guardrails_orchestrator(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    orchestrator_config: ConfigMap,
) -> Generator[GuardrailsOrchestrator, Any, Any]:
    gorch_kwargs = {
        "client": admin_client,
        "name": GUARDRAILS_ORCHESTRATOR_NAME,
        "namespace": model_namespace.name,
        "orchestrator_config": orchestrator_config.name,
        "replicas": 1,
        "wait_for_resource": True,
    }

    if enable_built_in_detectors := request.param.get("enable_built_in_detectors"):
        gorch_kwargs["enable_built_in_detectors"] = enable_built_in_detectors

    if request.param.get("enable_guardrails_gateway"):
        guardrails_gateway_config = request.getfixturevalue(argname="guardrails_gateway_config")
        gorch_kwargs["enable_guardrails_gateway"] = True
        gorch_kwargs["guardrails_gateway_config"] = guardrails_gateway_config.name

    with GuardrailsOrchestrator(**gorch_kwargs) as gorch:
        gorch_deployment = Deployment(name=gorch.name, namespace=gorch.namespace, wait_for_resource=True)
        gorch_deployment.wait_for_replicas()
        yield gorch


@pytest.fixture(scope="class")
def orchestrator_config(
    request: FixtureRequest, admin_client: DynamicClient, model_namespace: Namespace
) -> Generator[ConfigMap, Any, Any]:
    with ConfigMap(
        client=admin_client,
        name="fms-orchestr8-config-nlp",
        namespace=model_namespace.name,
        data=request.param["orchestrator_config_data"],
    ) as cm:
        yield cm


@pytest.fixture(scope="class")
def guardrails_gateway_config(
    request: FixtureRequest, admin_client: DynamicClient, model_namespace: Namespace
) -> Generator[ConfigMap, Any, Any]:
    with ConfigMap(
        client=admin_client,
        name="fms-orchestr8-config-gateway",
        namespace=model_namespace.name,
        label={Labels.Openshift.APP: "fmstack-nlp"},
        data=request.param["guardrails_gateway_config_data"],
    ) as cm:
        yield cm


@pytest.fixture(scope="class")
def guardrails_orchestrator_pod(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    guardrails_orchestrator: GuardrailsOrchestrator,
) -> Pod:
    return list(
        Pod.get(
            namespace=model_namespace.name, label_selector=f"app.kubernetes.io/instance={GUARDRAILS_ORCHESTRATOR_NAME}"
        )
    )[0]


@pytest.fixture(scope="class")
def guardrails_orchestrator_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    guardrails_orchestrator: GuardrailsOrchestrator,
) -> Generator[Route, Any, Any]:
    yield Route(
        name=f"{guardrails_orchestrator.name}",
        namespace=guardrails_orchestrator.namespace,
        wait_for_resource=True,
    )


@pytest.fixture(scope="class")
def guardrails_orchestrator_health_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    guardrails_orchestrator: GuardrailsOrchestrator,
) -> Generator[Route, Any, Any]:
    yield Route(
        name=f"{guardrails_orchestrator.name}-health",
        namespace=guardrails_orchestrator.namespace,
        wait_for_resource=True,
    )


# ServingRuntimes, InferenceServices, and related resources
# for generation and detection models
@pytest.fixture(scope="class")
def huggingface_sr(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntime(
        client=admin_client,
        name="guardrails-detector-runtime-prompt-injection",
        namespace=model_namespace.name,
        containers=[
            {
                "name": "kserve-container",
                "image": "quay.io/trustyai/guardrails-detector-huggingface-runtime:v0.2.0",
                "command": ["uvicorn", "app:app"],
                "args": [
                    "--workers=4",
                    "--host=0.0.0.0",
                    "--port=8000",
                    "--log-config=/common/log_conf.yaml",
                ],
                "env": [
                    {"name": "MODEL_DIR", "value": "/mnt/models"},
                    {"name": "HF_HOME", "value": "/tmp/hf_home"},
                ],
                "ports": [{"containerPort": 8000, "protocol": "TCP"}],
            }
        ],
        supported_model_formats=[{"name": "guardrails-detector-huggingface", "autoSelect": True}],
        multi_model=False,
        annotations={
            "openshift.io/display-name": "Guardrails Detector ServingRuntime for KServe",
            "opendatahub.io/recommended-accelerators": '["nvidia.com/gpu"]',
            "prometheus.io/port": "8080",
            "prometheus.io/path": "/metrics",
        },
        label={
            "opendatahub.io/dashboard": "true",
        },
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
def guardrails_orchestrator_ssl_cert(guardrails_orchestrator_route: Route):
    hostname = guardrails_orchestrator_route.host

    try:
        result = subprocess.run(
            args=["openssl", "s_client", "-showcerts", "-connect", f"{hostname}:443"],
            input="",
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0 and "CONNECTED" not in result.stdout:
            raise RuntimeError(f"Failed to connect to {hostname}: {result.stderr}")

        cert_lines = []
        in_cert = False
        for line in result.stdout.splitlines():
            if "-----BEGIN CERTIFICATE-----" in line:
                in_cert = True
            if in_cert:
                cert_lines.append(line)
            if "-----END CERTIFICATE-----" in line:
                in_cert = False

        if not cert_lines:
            raise RuntimeError(f"No certificate found in response from {hostname}")

        filepath = os.path.join(py_config["tmp_base_dir"], "gorch_cert.crt")
        with open(filepath, "w") as f:
            f.write("\n".join(cert_lines))

        return filepath

    except Exception as e:
        raise RuntimeError(f"Could not get certificate from {hostname}: {e}")


@pytest.fixture(scope="class")
def guardrails_orchestrator_ssl_cert_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    guardrails_orchestrator_ssl_cert: str,  # ← Add dependency and use correct cert
) -> Generator[Secret, Any, None]:
    with open(guardrails_orchestrator_ssl_cert, "r") as f:
        cert_content = f.read()

    with Secret(
        client=admin_client,
        name="orch-certificate",
        namespace=model_namespace.name,
        data_dict={"orch-certificate.crt": b64encode(cert_content.encode("utf-8")).decode("utf-8")},
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def patched_llamastack_deployment_tls_certs(llamastack_distribution, guardrails_orchestrator_ssl_cert_secret):
    lls_deployment = Deployment(
        name=llamastack_distribution.name,
        namespace=llamastack_distribution.namespace,
        ensure_exists=True,
    )

    current_spec = lls_deployment.instance.spec.template.spec.to_dict()

    current_spec["volumes"].append({
        "name": "router-ca",
        "secret": {"secretName": "orch-certificate"},  # pragma: allowlist secret
    })

    for container in current_spec["containers"]:
        if container["name"] == "llama-stack":
            container["volumeMounts"].append({"name": "router-ca", "mountPath": "/etc/llama/certs", "readOnly": True})
            break

    with ResourceEditor(patches={lls_deployment: {"spec": {"template": {"spec": current_spec}}}}) as _:
        initial_replicas = lls_deployment.replicas
        lls_deployment.scale_replicas(replica_count=0)
        lls_deployment.scale_replicas(replica_count=initial_replicas)
        lls_deployment.wait_for_replicas()
        yield lls_deployment


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


class OpenTelemetryCollector(NamespacedResource):
    """
    OpenTelemetryCollector is the Schema for the OpenTelemetry Collectors
    """
    api_group: str = "opentelemetry.io"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


class OpenTelemetryOperator(NamespacedResource):
    """
    OpenTelemetryOperator is the Schema for the opentelemetryoperators API
    """

    api_group: str = "opentelemetry.io"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

@pytest.fixture(scope="class")
def installed_opentelemetry_operator(admin_client: DynamicClient, model_namespace: Namespace):
    """
    Install the Red Hat OpenTelemetry operator in the same namespace as the test
    so CRs like OpenTelemetryCollector/Instrumentation are watched properly.
    """
    operator_ns = model_namespace
    operator_name = "opentelemetry-operator"

    otel_subscription = Subscription(
        client=admin_client,
        namespace=operator_ns.name,
        name=operator_name,
    )

    if not otel_subscription.exists:
        install_operator(
            admin_client=admin_client,
            target_namespaces=[operator_ns.name],
            name=operator_name,
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
        name=operator_name,
        operator_namespace=operator_ns.name,
        clean_up_namespace=True,
    )

@pytest.fixture(scope="class")
def otel_operator_cr(
    admin_client: DynamicClient,
    installed_opentelemetry_operator: None,
    model_namespace: Namespace,  # use the test namespace
) -> Generator[OpenTelemetryCollector, Any, Any]:
    """Create an OpenTelemetryCollector CR in the test namespace based on ALM examples from the CSV."""
    otel_csv: ClusterServiceVersion = get_cluster_service_version(
        client=admin_client,
        prefix="opentelemetry",
        namespace=model_namespace.name,  # fetch CSV from the same namespace as operator
    )

    alm_examples: list[dict[str, Any]] = otel_csv.get_alm_examples()
    otel_cr_dict: dict[str, Any] = next(
        example for example in alm_examples if example["kind"] == "OpenTelemetryCollector"
    )

    if not otel_cr_dict:
        raise ResourceNotFoundError(f"No OpenTelemetryCollector dict found in alm_examples for CSV {otel_csv.name}")

    otel_cr_dict["metadata"]["namespace"] = model_namespace.name  # create CR in test namespace

    with OpenTelemetryCollector(kind_dict=otel_cr_dict) as otel_cr:
        otel_cr.wait_for_condition(
            condition="Available",
            status=OpenTelemetryCollector.Condition.Status.TRUE,
            timeout=Timeout.TIMEOUT_10MIN,
        )
        yield otel_cr


class Jaeger(NamespacedResource):
    """
    Jaeger instance CR for Red Hat OpenShift distributed tracing platform.
    """
    api_group: str = "io.jaegertracing.openshift.v1"  # usually something like this

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

@pytest.fixture(scope="session")
def installed_jaeger_operator(admin_client: DynamicClient) -> Generator[None, Any, None]:
    """Install Red Hat OpenShift distributed tracing platform (Jaeger operator)."""
    operator_ns = Namespace(name="openshift-distributed-tracing", ensure_exists=True)
    operator_name = "jaeger-product"

    jaeger_subscription = Subscription(
        client=admin_client,
        namespace=operator_ns.name,
        name=operator_name,
        source="redhat-operators",
        channel="stable",
        starting_csv="jaeger-product.v1.65.0-4",
        installPlanApproval="Automatic",
    )

    if not jaeger_subscription.exists:
        jaeger_subscription.create()

    yield

@pytest.fixture(scope="class")
def jaeger_instance(
    admin_client: DynamicClient, model_namespace: Namespace
) -> Generator[Jaeger, Any, None]:
    """Create a Jaeger instance in the test namespace using default all-in-one strategy."""
    jaeger_csv: ClusterServiceVersion = get_cluster_service_version(
        client=admin_client,
        prefix="jaeger",
        namespace="openshift-distributed-tracing"
    )
    alm_examples: list[dict[str, Any]] = jaeger_csv.get_alm_examples()
    jaeger_dict: dict[str, Any] = next(example for example in alm_examples if example["kind"] == "Jaeger")

    if not jaeger_dict:
        raise ResourceNotFoundError(f"No Jaeger dict found in alm_examples for CSV {jaeger_csv.name}")

    jaeger_dict["metadata"]["namespace"] = model_namespace.name
    jaeger_dict["metadata"]["name"] = "simplest"
    jaeger_dict["spec"]["strategy"] = "allInOne"

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


def wait_for_jaeger_pods(client: DynamicClient, jaeger_name: str, namespace: str, timeout: int = Timeout.TIMEOUT_15MIN) -> None:
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

