import logging
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from typing import Any, NamedTuple

import pytest
import structlog
import yaml
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.gateway import Gateway
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.resource import Resource
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.service_account import ServiceAccount

from tests.model_serving.model_server.llmd.constants import (
    LLMD_DSC_CONDITION,
    LLMD_KSERVE_CONTROLLER_DEPLOYMENTS,
)
from tests.model_serving.model_server.llmd.llmd_configs import TinyLlamaOciConfig
from tests.model_serving.model_server.llmd.utils import (
    wait_for_llmisvc,
    wait_for_llmisvc_pods_ready,
)
from utilities.constants import Timeout
from utilities.infra import create_inference_token, s3_endpoint_secret, update_configmap_data
from utilities.llmd_utils import create_llmd_gateway
from utilities.logger import RedactedString
from utilities.resources.kuadrant import Kuadrant
from utilities.resources.leader_worker_set_operator import LeaderWorkerSetOperator

LOGGER = structlog.get_logger(name=__name__)
logging.getLogger("timeout_sampler").setLevel(logging.WARNING)


# ===========================================
#  Health Check
# ===========================================


class HealthCheckResult(NamedTuple):
    status: str
    name: str
    detail: str


def _format_health_report(checks: list[HealthCheckResult]) -> tuple[bool, str]:
    """Format health check results into a bordered report.

    Args:
        checks: List of HealthCheckResult from all check functions.

    Returns:
        Tuple of (passed, report) where passed is True if no failures,
        and report is the formatted string for logging and xfail.
    """
    lines = []
    for check in checks:
        line = f"  {check.status:<6}{check.name}"
        if check.detail:
            line += f": {check.detail}"
        lines.append(line)

    passed = not any(check.status == "FAIL" for check in checks)
    border = "=" * 60
    title = "LLMD Health Check — PASSED" if passed else "LLMD Health Check — FAILED"
    report = "\n".join(["", border, f"  {title}", border] + lines + [border, ""])
    return passed, report


@pytest.fixture(scope="session", autouse=True)
def llmd_health_check(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    dsc_resource: Resource,
) -> None:
    """Session-scoped health gate for all llm-d tests.

    Runs health checks on LLMD infrastructure dependencies, formats a report,
    and marks tests as xfail when unhealthy. Use --skip-llmd-health-check to disable.
    """
    if request.session.config.getoption("--skip-llmd-health-check"):
        LOGGER.warning("Skipping LLMD health check, got --skip-llmd-health-check")
        return

    selected_markers = {mark.name for item in request.session.items for mark in item.iter_markers()}
    if "component_health" in selected_markers:
        LOGGER.info("Skipping LLMD health gate because selected tests include component_health marker")
        return

    checks: list[HealthCheckResult] = []
    checks.extend(_check_llmd_dsc_dependencies(dsc_resource=dsc_resource))
    checks.extend(_check_llmd_controller_deployments(admin_client=admin_client))
    checks.extend(_check_llmd_lws_operator(admin_client=admin_client))
    checks.extend(_check_llmd_kuadrant(admin_client=admin_client))

    passed, report = _format_health_report(checks=checks)
    if passed:
        LOGGER.info(report)
    else:
        LOGGER.error(report)
        pytest.xfail(report)


def _check_llmd_dsc_dependencies(dsc_resource: Resource) -> list[HealthCheckResult]:
    """Check DSC KserveLLMInferenceServiceDependencies condition (covers cert-manager and RHCL).

    Args:
        dsc_resource: DataScienceCluster resource instance.

    Returns:
        List of HealthCheckResult. Status is "OK", "FAIL", or "WARN".
    """
    conditions = getattr(getattr(dsc_resource.instance, "status", None), "conditions", None)
    if not conditions:
        return [HealthCheckResult("FAIL", f"DSC {LLMD_DSC_CONDITION}", "DSC has no status conditions")]
    for condition in conditions:
        if condition.type == LLMD_DSC_CONDITION:
            if condition.status != "True":
                detail = f"{condition.get('reason')} — {condition.get('message')}"
                return [HealthCheckResult("FAIL", f"DSC {LLMD_DSC_CONDITION}", detail)]
            return [HealthCheckResult("OK", f"DSC {LLMD_DSC_CONDITION}", "")]
    return [HealthCheckResult("FAIL", f"DSC {LLMD_DSC_CONDITION}", "condition not found")]


def _check_llmd_controller_deployments(admin_client: DynamicClient) -> list[HealthCheckResult]:
    """Check KServe and LLMD controller deployments are Available (searched cluster-wide).

    Args:
        admin_client: Kubernetes dynamic client with cluster-admin privileges.

    Returns:
        List of HealthCheckResult. Status is "OK", "FAIL", or "WARN".
    """
    checks: list[HealthCheckResult] = []
    all_deployments = {dep.name: dep for dep in Deployment.get(client=admin_client)}
    for name in LLMD_KSERVE_CONTROLLER_DEPLOYMENTS:
        deployment = all_deployments.get(name)
        if not deployment:
            checks.append(HealthCheckResult("FAIL", f"Deployment {name}", "not found (cluster-wide)"))
            continue

        dep_available = False
        for condition in deployment.instance.status.get("conditions", []):
            if condition.type == "Available":
                if condition.status != "True":
                    checks.append(
                        HealthCheckResult(
                            "FAIL", f"Deployment {name} in {deployment.namespace}", condition.get("reason")
                        )
                    )
                else:
                    checks.append(HealthCheckResult("OK", f"Deployment {name} in {deployment.namespace}", ""))
                dep_available = True
                break

        if not dep_available:
            checks.append(
                HealthCheckResult("FAIL", f"Deployment {name} in {deployment.namespace}", "no Available condition")
            )
    return checks


def _check_llmd_lws_operator(admin_client: DynamicClient) -> list[HealthCheckResult]:
    """Check LeaderWorkerSetOperator CR is Available (optional — warns if absent).

    Args:
        admin_client: Kubernetes dynamic client with cluster-admin privileges.

    Returns:
        List of HealthCheckResult. Status is "OK", "FAIL", or "WARN".
    """
    lws_operator = LeaderWorkerSetOperator(client=admin_client, name="cluster")
    if not lws_operator.exists:
        return [HealthCheckResult("WARN", "LeaderWorkerSetOperator", "not found (optional)")]

    for condition in lws_operator.instance.status.get("conditions", []):
        if condition.type == "Available":
            if condition.status != "True":
                return [
                    HealthCheckResult("FAIL", "LeaderWorkerSetOperator", f"not Available ({condition.get('reason')})")
                ]
            return [HealthCheckResult("OK", "LeaderWorkerSetOperator", "")]
    return [HealthCheckResult("FAIL", "LeaderWorkerSetOperator", "no Available condition")]


def _check_llmd_kuadrant(admin_client: DynamicClient) -> list[HealthCheckResult]:
    """Check a Kuadrant CR exists and is Ready (searched cluster-wide, no hardcoded namespace).

    Args:
        admin_client: Kubernetes dynamic client with cluster-admin privileges.

    Returns:
        List of HealthCheckResult. Status is "OK", "FAIL", or "WARN".
    """
    for kuadrant in Kuadrant.get(client=admin_client):
        kuadrant_label = f"Kuadrant CR: {kuadrant.name} in {kuadrant.namespace}"
        for condition in kuadrant.instance.status.get("conditions", []):
            if condition.type == "Ready":
                if condition.status != "True":
                    return [HealthCheckResult("FAIL", kuadrant_label, f"not Ready ({condition.get('reason')})")]
                return [HealthCheckResult("OK", kuadrant_label, "")]
        return [HealthCheckResult("FAIL", kuadrant_label, "no Ready condition")]
    return [HealthCheckResult("FAIL", "Kuadrant CR", "not found (cluster-wide)")]


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
#  Skip Conditions
# ===========================================
@pytest.fixture(scope="class")
def skip_if_fast_cr_missing(
    request: FixtureRequest,
    admin_client: DynamicClient,
) -> None:
    """Skip the test class if no fast-image LLMInferenceServiceConfig CR exists on the cluster.

    This fixture is opt-in — only test classes decorated with
    ``@pytest.mark.usefixtures("skip_if_fast_cr_missing")`` invoke it.

    It reads the config class from the ``llmisvc`` parametrize arg, detects the
    cluster's GPU accelerator, and queries the DSCI applications namespace for a
    matching LLMInferenceServiceConfig CR using the config's
    ``accelerator_config_name_regex``.

    The fixture is a no-op (returns immediately) when the config class does not
    define a fast-image regex — i.e. when ``accelerator_config_name_regex`` equals
    the ``GpuConfig`` default ``^(?!.*fast-)``.  This prevents the fixture from
    running unnecessary discovery for standard (non-fast) GPU configs.
    """
    callspec = getattr(request.node, "callspec", None)
    config_cls = callspec.params.get("llmisvc") if callspec else None

    # The GpuConfig default regex "^(?!.*fast-)" is a negative lookahead that
    # excludes fast CRs — it is NOT a fast-image regex.  Only proceed when the
    # config overrides the default with a positive fast-matching pattern.
    default_regex = "^(?!.*fast-)"
    config_regex = getattr(config_cls, "accelerator_config_name_regex", default_regex)
    if not config_cls or config_regex == default_regex:
        return

    from tests.model_serving.model_server.llmd.constants import LLMD_TESTS_SUPPORTED_ACCELERATORS
    from tests.model_serving.model_server.llmd.utils import detect_accelerators, list_matching_accelerator_configs
    from utilities.infra import get_dsci_applications_namespace

    # Detect which GPU accelerators are available on worker nodes
    node_accelerators = detect_accelerators(client=admin_client)
    cluster_accelerator = None
    for node in node_accelerators:
        for resource_name in node:
            if resource_name in LLMD_TESTS_SUPPORTED_ACCELERATORS:
                cluster_accelerator = resource_name
                break
        if cluster_accelerator:
            break

    if not cluster_accelerator:
        pytest.skip(
            f"No supported GPU accelerator found on any worker node."
            f" Config class: {config_cls.__name__},"
            f" accelerator_config_name_regex: '{config_regex}'."
            f" Supported accelerator types: {LLMD_TESTS_SUPPORTED_ACCELERATORS}."
            f" Worker node accelerators detected: {node_accelerators or 'none (no GPUs found)'}"
        )

    # Query the DSCI applications namespace for a matching fast CR
    topology = getattr(config_cls, "supported_topology", "workload-single-node")
    apps_namespace = get_dsci_applications_namespace(client=admin_client)
    result = list_matching_accelerator_configs(
        client=admin_client,
        namespace=apps_namespace,
        accelerator=cluster_accelerator,
        topology=topology,
        name_regex=config_regex,
    )
    if not result.name:
        pytest.skip(
            f"No LLMInferenceServiceConfig CR matches the fast-image requirements."
            f" Config class: {config_cls.__name__},"
            f" accelerator_config_name_regex: '{config_regex}',"
            f" detected accelerator: '{cluster_accelerator}',"
            f" required topology: '{topology}',"
            f" search namespace: '{apps_namespace}'."
            f" The cluster does not have an LLMInferenceServiceConfig CR with"
            f" annotation opendatahub.io/recommended-accelerators containing"
            f" '{cluster_accelerator}'"
            f" AND annotation opendatahub.io/supported-topologies containing"
            f" '{topology}'"
            f" AND name matching regex '{config_regex}'."
            f" CRs that passed the name regex (with annotations):"
            f" {result.candidates or 'none'}."
            f" All LLMInferenceServiceConfig CRs in namespace '{apps_namespace}':"
            f" {result.all_cr_names or 'none'}"
        )


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
    config_cls = request.param.build(client=admin_client)
    namespace = unprivileged_model_namespace.name

    service_account = None
    if config_cls.storage_uri.startswith("s3://"):
        service_account = request.getfixturevalue(argname="s3_service_account")

    with _create_llmisvc_from_config(
        config_cls=config_cls, namespace=namespace, client=admin_client, service_account=service_account
    ) as svc:
        yield svc


class AuthEntry(NamedTuple):
    service: LLMInferenceService
    token: RedactedString


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
            ).build(client=admin_client)
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
        "containers": [main_container],
    }
    if service_account:
        template["serviceAccountName"] = service_account

    prefill = config_cls.prefill_config()
    if prefill and service_account and "template" in prefill:
        prefill["template"]["serviceAccountName"] = service_account

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
        "base_refs": config_cls.base_refs,
        "prefill": prefill,
    }

    LOGGER.info(f"\n{config_cls.format_describe(namespace=namespace)}")

    with LLMInferenceService(**svc_kwargs) as llm_service:
        wait_for_llmisvc(llmisvc=llm_service, timeout=config_cls.wait_timeout)
        wait_for_llmisvc_pods_ready(client=client, llmisvc=llm_service)
        yield llm_service
