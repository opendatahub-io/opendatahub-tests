import logging
from collections import namedtuple
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from typing import Any

import pytest
import structlog
import yaml
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.gateway import Gateway
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.resource import Resource
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.service_account import ServiceAccount
from pytest_testconfig import config as py_config

from tests.model_serving.model_server.llmd.llmd_configs import TinyLlamaOciConfig
from tests.model_serving.model_server.llmd.utils import wait_for_llmisvc, wait_for_llmisvc_pods_ready
from utilities.constants import Timeout
from utilities.infra import create_inference_token, s3_endpoint_secret, update_configmap_data
from utilities.llmd_utils import create_llmd_gateway
from utilities.logger import RedactedString
from utilities.resources.kuadrant import Kuadrant
from utilities.resources.leader_worker_set_operator import LeaderWorkerSetOperator

LOGGER = structlog.get_logger(name=__name__)
logging.getLogger("timeout_sampler").setLevel(logging.WARNING)

LLMD_DSC_CONDITION: str = "KserveLLMInferenceServiceDependencies"

LLMD_REQUIRED_OPERATORS: dict[str, str] = {
    "cert-manager-operator": "cert-manager-operator",
    "authorino-operator": "kuadrant-system",
    "rhcl-operator": "openshift-operators",
}

LLMD_REQUIRED_DEPLOYMENTS: dict[str, str] = {
    "cert-manager-operator-controller-manager": "cert-manager-operator",
    "cert-manager": "cert-manager",
    "cert-manager-webhook": "cert-manager",
    "authorino-operator": "kuadrant-system",
    "kuadrant-operator-controller-manager": "kuadrant-system",
}

# Same KServe stack as tests/model_serving/model_server/kserve/conftest.py plus LLM-ISVC controller.
LLMD_KSERVE_CONTROLLER_DEPLOYMENTS: list[str] = [
    "kserve-controller-manager",
    "odh-model-controller",
    "llmisvc-controller-manager",
]


def _verify_operator_csv(admin_client: DynamicClient, csv_prefix: str, namespace: str) -> None:
    for csv in ClusterServiceVersion.get(client=admin_client, namespace=namespace):
        if csv.name.startswith(csv_prefix) and csv.status == csv.Status.SUCCEEDED:
            return
    pytest.xfail(f"Operator CSV {csv_prefix} not found or not Succeeded in {namespace}")


def verify_llmd_health(admin_client: DynamicClient, dsc_resource: Resource) -> None:
    """Verify LLMD infrastructure dependencies are healthy.

    Checks DSC condition, required operator CSVs, dependency and KServe controller
    deployments, optional LeaderWorkerSetOperator CR (LWS is optional),
    and Kuadrant CR.
    """
    # 1. DSC condition for LLMD dependencies
    for condition in dsc_resource.instance.status.conditions:
        if condition.type == LLMD_DSC_CONDITION:
            if condition.status != "True":
                pytest.xfail(
                    f"{LLMD_DSC_CONDITION} is not ready: {condition.status}, reason: {condition.get('reason')}"
                )
            break
    else:
        pytest.xfail(f"{LLMD_DSC_CONDITION} condition not found in DSC status")

    # 2. Operator CSVs
    for csv_prefix, namespace in LLMD_REQUIRED_OPERATORS.items():
        _verify_operator_csv(admin_client=admin_client, csv_prefix=csv_prefix, namespace=namespace)

    # 3. Controller deployments
    for name, namespace in LLMD_REQUIRED_DEPLOYMENTS.items():
        deployment = Deployment(client=admin_client, name=name, namespace=namespace)
        if not deployment.exists:
            pytest.xfail(f"LLMD dependency deployment {name} not found in {namespace}")

        dep_available = False
        for condition in deployment.instance.status.get("conditions", []):
            if condition.type == "Available":
                if condition.status != "True":
                    pytest.xfail(f"Deployment {name} in {namespace} is not Available: {condition.get('reason')}")
                dep_available = True
                break

        if not dep_available:
            pytest.xfail(f"Deployment {name} in {namespace} has no Available condition")

    applications_namespace = py_config["applications_namespace"]
    for name in LLMD_KSERVE_CONTROLLER_DEPLOYMENTS:
        deployment = Deployment(client=admin_client, name=name, namespace=applications_namespace)
        if not deployment.exists:
            pytest.xfail(f"KServe/LLMD controller deployment {name} not found in {applications_namespace}")

        kserve_dep_available = False
        for condition in deployment.instance.status.get("conditions", []):
            if condition.type == "Available":
                if condition.status != "True":
                    pytest.xfail(
                        f"Deployment {name} in {applications_namespace} is not Available: {condition.get('reason')}"
                    )
                kserve_dep_available = True
                break

        if not kserve_dep_available:
            pytest.xfail(f"Deployment {name} in {applications_namespace} has no Available condition")

    # 4. LeaderWorkerSetOperator CR (optional)
    lws_operator = LeaderWorkerSetOperator(client=admin_client, name="cluster")
    if lws_operator.exists:
        lws_available = False
        for condition in lws_operator.instance.status.get("conditions", []):
            if condition.type == "Available":
                if condition.status != "True":
                    pytest.xfail(f"LeaderWorkerSetOperator is not Available: {condition.get('reason')}")
                lws_available = True
                break

        if not lws_available:
            pytest.xfail("LeaderWorkerSetOperator has no Available condition")
    else:
        LOGGER.warning("LeaderWorkerSetOperator cluster CR not found; LWS is optional for LLMD (RHOAIENG-52057)")

    # 5. Kuadrant CR
    kuadrant = Kuadrant(client=admin_client, name="kuadrant", namespace="kuadrant-system")
    if not kuadrant.exists:
        pytest.xfail("Kuadrant 'kuadrant' CR not found")

    LOGGER.info("LLMD component health check passed")


@pytest.fixture(scope="session", autouse=True)
def llmd_health_check(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    dsc_resource: Resource,
) -> None:
    """Session-scoped health gate for all LLMD tests.

    Marks LLMD tests as xfail when required infrastructure dependencies are unhealthy
    (see verify_llmd_health). Use --skip-llmd-health-check to disable.
    """
    if request.session.config.getoption("--skip-llmd-health-check"):
        LOGGER.warning("Skipping LLMD health check, got --skip-llmd-health-check")
        return

    selected_markers = {mark.name for item in request.session.items for mark in item.iter_markers()}
    if "component_health" in selected_markers:
        LOGGER.info("Skipping LLMD health gate because selected tests include component_health marker")
        return

    verify_llmd_health(admin_client=admin_client, dsc_resource=dsc_resource)


AuthEntry = namedtuple(typename="AuthEntry", field_names=["service", "token"])


# ===========================================
#  Gateway
# ===========================================
@pytest.fixture(scope="session", autouse=True)
def shared_llmd_gateway(admin_client: DynamicClient) -> Generator[Gateway]:
    """Shared LLMD gateway for all tests."""
    with create_llmd_gateway(
        client=admin_client,
        timeout=Timeout.TIMEOUT_1MIN,
    ) as gateway:
        yield gateway


# ===========================================
#  Storage — S3 secret + service account
# ===========================================
@pytest.fixture(scope="class")
def s3_service_account(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[str]:
    """Create S3 secret + service account. Resolved automatically for S3 configs."""
    with ExitStack() as stack:
        secret = stack.enter_context(
            cm=s3_endpoint_secret(
                client=admin_client,
                name="llmd-s3-secret",
                namespace=unprivileged_model_namespace.name,
                aws_access_key=request.getfixturevalue(argname="aws_access_key_id"),
                aws_secret_access_key=request.getfixturevalue(argname="aws_secret_access_key"),
                aws_s3_region=request.getfixturevalue(argname="models_s3_bucket_region"),
                aws_s3_bucket=request.getfixturevalue(argname="models_s3_bucket_name"),
                aws_s3_endpoint=request.getfixturevalue(argname="models_s3_bucket_endpoint"),
            )
        )
        sa = stack.enter_context(
            cm=ServiceAccount(
                client=admin_client,
                namespace=unprivileged_model_namespace.name,
                name="llmd-s3-service-account",
                secrets=[{"name": secret.name}],
            )
        )
        yield sa.name


# ===========================================
#  GPU guards
# ===========================================
@pytest.fixture(scope="session")
def skip_if_less_than_2_gpus(gpu_count_on_cluster: int) -> None:
    """Skip test if fewer than 2 GPUs are available on the cluster."""
    if gpu_count_on_cluster < 2:
        pytest.skip(f"Test requires at least 2 GPUs (found {gpu_count_on_cluster})")


# ===========================================
#  LLMInferenceService creation
# ===========================================
@pytest.fixture(scope="class")
def llmisvc(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[LLMInferenceService]:
    """LLMInferenceService fixture driven by a config class.

    Usage:
        NAMESPACE = ns_from_file(__file__)

        @pytest.mark.parametrize(
            "unprivileged_model_namespace, llmisvc",
            [({"name": NAMESPACE}, SomeConfig)],
            indirect=True,
        )
    """
    config_cls = request.param
    namespace = unprivileged_model_namespace.name

    service_account = None
    if config_cls.storage_uri.startswith("s3://"):
        service_account = request.getfixturevalue(argname="s3_service_account")

    with _create_llmisvc_from_config(
        config_cls=config_cls, namespace=namespace, client=admin_client, service_account=service_account
    ) as svc:
        yield svc


@pytest.fixture(scope="class")
def llmisvc_auth_pair(
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[tuple[AuthEntry, AuthEntry]]:
    """Two auth-enabled LLMISVCs with independent tokens for cross-auth testing."""
    namespace = unprivileged_model_namespace.name
    with ExitStack() as stack:
        entries = []
        for i in range(2):
            cfg = TinyLlamaOciConfig.with_overrides(
                name=f"llmisvc-auth-{i}",
                enable_auth=True,
            )
            svc = stack.enter_context(
                cm=_create_llmisvc_from_config(
                    config_cls=cfg,
                    namespace=namespace,
                    client=admin_client,
                )
            )
            token = stack.enter_context(
                cm=_create_auth_resources(
                    client=admin_client,
                    namespace=namespace,
                    svc=svc,
                    sa_name=f"auth-sa-{i}",
                )
            )
            entries.append(AuthEntry(service=svc, token=token))
        yield tuple(entries)


# ===========================================
#  Auth — SA + RBAC + token
# ===========================================
@pytest.fixture(scope="class")
def llmisvc_token(
    admin_client: DynamicClient,
    llmisvc: LLMInferenceService,
) -> Generator[str]:
    """Create a dedicated SA with RBAC and return an auth token for the llmisvc."""
    with _create_auth_resources(
        client=admin_client,
        namespace=llmisvc.namespace,
        svc=llmisvc,
        sa_name=f"{llmisvc.name}-auth-sa",
    ) as token:
        yield token


# ===========================================
#  Monitoring
# ===========================================
@pytest.fixture(scope="session", autouse=True)
def llmd_user_workload_monitoring_config_map(
    admin_client: DynamicClient, cluster_monitoring_config: ConfigMap
) -> Generator[ConfigMap]:
    """Ephemeral user workload monitoring for LLMD tests."""
    data = {
        "config.yaml": yaml.dump({
            "prometheus": {
                "logLevel": "debug",
                "retention": "15d",
            }
        })
    }

    with update_configmap_data(
        client=admin_client,
        name="user-workload-monitoring-config",
        namespace="openshift-user-workload-monitoring",
        data=data,
    ) as cm:
        yield cm


# ===========================================
#  Helpers (not fixtures)
# ===========================================
@contextmanager
def _create_auth_resources(
    client: DynamicClient,
    namespace: str,
    svc: LLMInferenceService,
    sa_name: str,
) -> Generator[RedactedString, Any]:
    """Create SA + Role + RoleBinding and yield an auth token."""
    with (
        ServiceAccount(client=client, namespace=namespace, name=sa_name) as sa,
        Role(
            client=client,
            name=f"{svc.name}-view",
            namespace=namespace,
            rules=[
                {
                    "apiGroups": [svc.api_group],
                    "resources": ["llminferenceservices"],
                    "verbs": ["get"],
                    "resourceNames": [svc.name],
                }
            ],
        ) as role,
        RoleBinding(
            client=client,
            namespace=namespace,
            name=f"{sa_name}-view",
            role_ref_name=role.name,
            role_ref_kind=role.kind,
            subjects_kind="ServiceAccount",
            subjects_name=sa_name,
        ),
    ):
        yield RedactedString(value=create_inference_token(model_service_account=sa))


@contextmanager
def _create_llmisvc_from_config(
    config_cls: type,
    namespace: str,
    client: DynamicClient,
    service_account: str | None = None,
    teardown: bool = True,
) -> Generator[LLMInferenceService, Any]:
    """Create an LLMInferenceService from a config class."""
    LOGGER.info(f"\n{config_cls.describe(namespace=namespace)}")

    model: dict[str, Any] = {"uri": config_cls.storage_uri}
    if config_cls.model_name:
        model["name"] = config_cls.model_name

    main_container: dict[str, Any] = {"name": "main"}
    main_container.update({
        k: v
        for k, v in {
            "image": config_cls.container_image,
            "resources": config_cls.container_resources(),
            "env": config_cls.container_env(),
            "livenessProbe": config_cls.liveness_probe(),
            "readinessProbe": config_cls.readiness_probe(),
        }.items()
        if v
    })

    template: dict[str, Any] = {
        "configRef": config_cls.template_config_ref,
        "containers": [main_container],
    }
    if service_account:
        template["serviceAccountName"] = service_account

    prefill = config_cls.prefill_config()

    svc_kwargs: dict[str, Any] = {
        "client": client,
        "name": config_cls.name,
        "namespace": namespace,
        "annotations": config_cls.annotations(),
        "label": config_cls.labels(),
        "teardown": teardown,
        "model": model,
        "replicas": config_cls.replicas,
        "router": config_cls.router_config(),
        "template": template,
    }
    if prefill is not None:
        if service_account and "template" in prefill:
            prefill["template"]["serviceAccountName"] = service_account
        svc_kwargs["prefill"] = prefill

    with LLMInferenceService(**svc_kwargs) as llm_service:
        wait_for_llmisvc(llmisvc=llm_service, timeout=config_cls.wait_timeout)
        wait_for_llmisvc_pods_ready(client=client, llmisvc=llm_service)
        yield llm_service
