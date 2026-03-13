from collections.abc import Generator
from typing import Any

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.guardrails_orchestrator import GuardrailsOrchestrator
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.route import Route

from utilities.constants import Annotations, Labels
from utilities.guardrails import check_guardrails_health_endpoint

GUARDRAILS_ORCHESTRATOR_NAME: str = "guardrails-orchestrator"


@pytest.fixture(scope="class")
def guardrails_orchestrator(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    teardown_resources: bool,
    pytestconfig: pytest.Config,
) -> Generator[GuardrailsOrchestrator, Any, Any]:
    gorch_kwargs = {
        "client": admin_client,
        "name": GUARDRAILS_ORCHESTRATOR_NAME,
        "namespace": model_namespace.name,
    }
    if pytestconfig.option.post_upgrade:
        gorch = GuardrailsOrchestrator(**gorch_kwargs, ensure_exists=True)
        yield gorch
        gorch.clean_up()
    else:
        gorch_kwargs["log_level"] = "DEBUG"
        gorch_kwargs["replicas"] = 1
        gorch_kwargs["wait_for_resource"] = True
        if request.param.get("auto_config"):
            gorch_kwargs["auto_config"] = request.param.get("auto_config")

        if request.param.get("orchestrator_config"):
            orchestrator_config = request.getfixturevalue(argname="orchestrator_config")
            gorch_kwargs["orchestrator_config"] = orchestrator_config.name

        if request.param.get("enable_guardrails_gateway"):
            gorch_kwargs["enable_guardrails_gateway"] = True

        if request.param.get("guardrails_gateway_config"):
            guardrails_gateway_config = request.getfixturevalue(argname="guardrails_gateway_config")
            gorch_kwargs["guardrails_gateway_config"] = guardrails_gateway_config.name

        if enable_built_in_detectors := request.param.get("enable_built_in_detectors"):
            gorch_kwargs["enable_built_in_detectors"] = enable_built_in_detectors

        if request.param.get("otel_exporter_config"):
            metrics_endpoint = request.getfixturevalue(argname="otelcol_metrics_endpoint")
            traces_endpoint = request.getfixturevalue(argname="tempo_traces_endpoint")
            gorch_kwargs["otel_exporter"] = {
                "otlpProtocol": "grpc",
                "otlpMetricsEndpoint": metrics_endpoint,
                "otlpTracesEndpoint": traces_endpoint,
                "enableMetrics": True,
                "enableTracing": True,
            }

        with GuardrailsOrchestrator(**gorch_kwargs, teardown=teardown_resources) as gorch:
            gorch_deployment = Deployment(name=gorch.name, namespace=gorch.namespace, wait_for_resource=True)
            gorch_deployment.wait_for_replicas()
            yield gorch


@pytest.fixture(scope="class")
def orchestrator_config(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    teardown_resources: bool,
    pytestconfig: pytest.Config,
) -> Generator[ConfigMap, Any, Any]:
    if pytestconfig.option.post_upgrade:
        cm = ConfigMap(
            client=admin_client, name="fms-orchestr8-config-nlp", namespace=model_namespace.name, ensure_exists=True
        )
        yield cm
        cm.clean_up()
    else:
        with ConfigMap(
            client=admin_client,
            name="fms-orchestr8-config-nlp",
            namespace=model_namespace.name,
            data=request.param["orchestrator_config_data"],
            teardown=teardown_resources,
        ) as cm:
            yield cm


@pytest.fixture(scope="class")
def guardrails_gateway_config(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    teardown_resources: bool,
    pytestconfig: pytest.Config,
) -> Generator[ConfigMap, Any, Any]:
    if pytestconfig.option.post_upgrade:
        cm = ConfigMap(
            client=admin_client, name="fms-orchestr8-config-gateway", namespace=model_namespace.name, ensure_exists=True
        )
        yield cm
        cm.clean_up()
    else:
        with ConfigMap(
            client=admin_client,
            name="fms-orchestr8-config-gateway",
            namespace=model_namespace.name,
            label={Labels.Openshift.APP: "fmstack-nlp"},
            data=request.param["guardrails_gateway_config_data"],
            teardown=teardown_resources,
        ) as cm:
            yield cm


@pytest.fixture(scope="class")
def guardrails_orchestrator_pod(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    guardrails_orchestrator: GuardrailsOrchestrator,
) -> Pod:
    pods = Pod.get(
        namespace=model_namespace.name, label_selector=f"app.kubernetes.io/instance={GUARDRAILS_ORCHESTRATOR_NAME}"
    )
    pod = next(iter(pods), None)
    if pod is None:
        raise RuntimeError(
            f"No guardrails orchestrator pod found with label app.kubernetes.io/instance={GUARDRAILS_ORCHESTRATOR_NAME}"
        )
    return pod


@pytest.fixture(scope="class")
def guardrails_orchestrator_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    guardrails_orchestrator: GuardrailsOrchestrator,
) -> Generator[Route, Any, Any]:
    guardrails_orchestrator_route = Route(
        name=f"{guardrails_orchestrator.name}",
        namespace=guardrails_orchestrator.namespace,
        wait_for_resource=True,
        ensure_exists=True,
    )
    with ResourceEditor(
        patches={
            guardrails_orchestrator_route: {
                "metadata": {
                    "annotations": {Annotations.HaproxyRouterOpenshiftIo.TIMEOUT: "10m"},
                }
            }
        }
    ):
        yield guardrails_orchestrator_route


@pytest.fixture(scope="class")
def guardrails_orchestrator_url(
    guardrails_orchestrator_route: Route,
) -> str:
    return f"https://{guardrails_orchestrator_route.host}"


@pytest.fixture(scope="class")
def guardrails_orchestrator_health_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    guardrails_orchestrator: GuardrailsOrchestrator,
) -> Route:
    return Route(
        name=f"{guardrails_orchestrator.name}-health",
        namespace=guardrails_orchestrator.namespace,
        wait_for_resource=True,
        ensure_exists=True,
    )


@pytest.fixture
def guardrails_healthcheck(
    current_client_token, openshift_ca_bundle_file, guardrails_orchestrator_health_route: Route
) -> None:
    check_guardrails_health_endpoint(
        token=current_client_token,
        host=guardrails_orchestrator_health_route.host,
        ca_bundle_file=openshift_ca_bundle_file,
    )


@pytest.fixture(scope="class")
def guardrails_orchestrator_gateway_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    guardrails_orchestrator: GuardrailsOrchestrator,
) -> Route:
    return Route(
        name=f"{guardrails_orchestrator.name}-gateway",
        namespace=guardrails_orchestrator.namespace,
        wait_for_resource=True,
        ensure_exists=True,
    )
