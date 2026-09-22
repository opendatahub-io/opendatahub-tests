import socket
from collections.abc import Generator
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import NotFoundError
from ocp_resources.deployment import Deployment
from ocp_resources.evalhub import EvalHub
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.evalhub.constants import EVALHUB_USER_ROLE_RULES
from tests.ai_safety.evalhub.mcp.constants import (
    EVALHUB_MCP_CR_NAME,
    EVALHUB_MCP_HEALTH_PATH,
)
from tests.ai_safety.evalhub.mcp.utils import (
    EvalHubMcpClient,
    build_mcp_proxy_role_rules,
    evalhub_mcp_pod_label_selector,
)
from tests.ai_safety.evalhub.utils import is_evalhub_crd_available, wait_for_service_account
from utilities.certificates_utils import create_ca_bundle_file
from utilities.infra import create_inference_token

LOGGER = structlog.get_logger(name=__name__)


class _TransientEvalhubMcpHealthError(Exception):
    """Recoverable failure while polling the EvalHub MCP health endpoint."""


_TRANSIENT_MCP_HEALTH_REQUEST_EXCEPTIONS = (
    requests.exceptions.ConnectTimeout,
    requests.exceptions.ReadTimeout,
)
_TRANSIENT_MCP_HEALTH_EXCEPTIONS = {_TransientEvalhubMcpHealthError: []}


def _is_dns_resolution_error(err: BaseException) -> bool:
    """Return True when the exception chain includes a DNS resolution failure."""
    exc: BaseException | None = err
    while exc is not None:
        if isinstance(exc, socket.gaierror):
            return True
        exc = exc.__cause__
    return False


def _probe_evalhub_mcp_health(
    url: str,
    host: str,
    ca_bundle_file: str,
) -> requests.Response:
    """GET the MCP health endpoint, retrying only on transient network failures."""
    try:
        return requests.get(url, verify=ca_bundle_file, timeout=10)
    except requests.exceptions.ConnectionError as err:
        if isinstance(err, requests.exceptions.SSLError) or _is_dns_resolution_error(err):
            raise
        LOGGER.warning(f"Transient error checking EvalHub MCP health at {host}: {err}")
        raise _TransientEvalhubMcpHealthError(str(err)) from err
    except _TRANSIENT_MCP_HEALTH_REQUEST_EXCEPTIONS as err:
        LOGGER.warning(f"Transient error checking EvalHub MCP health at {host}: {err}")
        raise _TransientEvalhubMcpHealthError(str(err)) from err


def _mcp_deployment_name(cr_name: str) -> str:
    return f"{cr_name}-mcp"


def _mcp_auth_secret_name(cr_name: str) -> str:
    return f"{cr_name}-mcp-token"


def _evalhub_service_account_name(cr_name: str) -> str:
    return f"{cr_name}-service"


def _template_references_secret(deployment_instance: Any, secret_name: str) -> bool:
    """Return True if the Deployment's pod template uses ``secret_name`` in an env var or a volume."""
    pod_spec = deployment_instance.spec.template.spec
    env_refs = [
        env.valueFrom.secretKeyRef.name
        for container in pod_spec.containers or []
        for env in container.env or []
        if env.valueFrom and env.valueFrom.secretKeyRef
    ]
    volume_refs = [volume.secret.secretName for volume in pod_spec.volumes or [] if volume.secret]
    return secret_name in env_refs + volume_refs


def _wait_for_deployment_rollout(
    deployment: Deployment,
    timeout: int = 300,
    required_secret: str | None = None,
) -> None:
    """Wait until Kubernetes has fully applied the latest Deployment change.

    We wait until:
    - The pod template uses ``required_secret`` (if given), so we know the operator
      has already written the auth change into the Deployment.
    - Kubernetes has noticed the latest change.
    - All new pods are ready.
    - No pods are unavailable.
    - The old pods have been removed.

    Only NotFoundError is retried; any other error is raised straight away.
    """
    last_status: Any = None
    try:
        for instance in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=lambda: deployment.instance,
            exceptions_dict={NotFoundError: []},
        ):
            status = instance.status
            last_status = status
            if status is None:
                continue
            if required_secret and not _template_references_secret(
                deployment_instance=instance, secret_name=required_secret
            ):
                LOGGER.info(f"Waiting for Deployment {deployment.name} pod template to use secret {required_secret}")
                continue
            desired = instance.spec.replicas or 1
            generation = instance.metadata.generation or 0
            observed = getattr(status, "observedGeneration", None) or 0
            updated = getattr(status, "updatedReplicas", None) or 0
            total = getattr(status, "replicas", None) or 0
            unavailable = getattr(status, "unavailableReplicas", None) or 0
            if observed >= generation and updated >= desired and total <= updated and unavailable == 0:
                return
    except TimeoutExpiredError as err:
        raise RuntimeError(
            f"Deployment {deployment.name} rollout did not finish within {timeout}s. Last status: {last_status}"
        ) from err


def _mcp_pod_states(admin_client: DynamicClient, namespace: str, label_selector: str) -> list[tuple[str, str, bool]]:
    """Return the name, status, and whether each matching pod is shutting down."""
    # raw=True gives the pod data straight from one list call, so a pod deleted
    # mid-poll can't raise NotFoundError on a second per-pod fetch.
    states = []
    for pod in Pod.get(client=admin_client, namespace=namespace, label_selector=label_selector, raw=True):
        phase = pod.status.phase if pod.status else None
        states.append((pod.metadata.name, phase, pod.metadata.deletionTimestamp is not None))
    return states


def _wait_for_mcp_pods_settled(
    admin_client: DynamicClient,
    namespace: str,
    instance_name: str,
    desired: int,
    timeout: int = 120,
) -> None:
    """Wait until the expected number of MCP pods are running and no old pods are shutting down.

    Kubernetes can still show an old pod for a short time after a rollout finishes, and a
    single matching read can catch a transient state. This requires the desired state to hold
    for several consecutive reads before returning, so only the new MCP pods remain before the
    tests continue.
    """
    label_selector = evalhub_mcp_pod_label_selector(instance_name=instance_name)
    last_seen: list[tuple[str, str, bool]] = []
    required_stable_reads = 3
    stable_reads = 0
    try:
        for states in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=lambda: _mcp_pod_states(admin_client=admin_client, namespace=namespace, label_selector=label_selector),
            exceptions_dict={NotFoundError: []},
        ):
            last_seen = states
            if len(states) == desired and all(
                phase == Pod.Status.RUNNING and not terminating for _, phase, terminating in states
            ):
                stable_reads += 1
                if stable_reads >= required_stable_reads:
                    return
            else:
                stable_reads = 0
            LOGGER.info(
                f"Waiting for {desired} settled MCP pod(s) in {namespace} "
                f"({stable_reads}/{required_stable_reads} stable reads); "
                f"current (name, phase, terminating): {states}"
            )
    except TimeoutExpiredError as err:
        raise RuntimeError(
            f"MCP pods in {namespace} did not settle to {desired} Running, non-terminating pod(s) "
            f"within {timeout}s. Last seen (name, phase, terminating): {last_seen}"
        ) from err


def _wait_for_mcp_reconciled(evalhub: EvalHub, generation: int, timeout: int = 300) -> None:
    """Wait until the operator reports MCP reconciled for ``generation`` of the EvalHub CR.

    Top-level ``status.ready``/``status.phase`` describe only the main EvalHub
    deployment and are unaffected by MCP-only spec changes, so they cannot tell us
    when an MCP patch has been applied. The operator stamps each ``status.mcp``
    condition with the CR generation it reconciled, and updates the MCP Deployment
    before writing the ``Reconciled`` condition.
    """
    last_mcp_status: Any = None
    try:
        for status in TimeoutSampler(
            wait_timeout=timeout,
            sleep=2,
            func=lambda: evalhub.instance.status,
            exceptions_dict={NotFoundError: []},
        ):
            mcp_status = status.get("mcp") if status is not None else None
            last_mcp_status = mcp_status
            for condition in (mcp_status.get("conditions") if mcp_status else None) or []:
                if (
                    condition.get("type") == "Reconciled"
                    and condition.get("status") == "True"
                    and (condition.get("observedGeneration") or 0) >= generation
                ):
                    return
    except TimeoutExpiredError as err:
        raise RuntimeError(
            f"EvalHub MCP was not reconciled for generation {generation} within {timeout}s. "
            f"Last MCP status: {last_mcp_status}"
        ) from err


@pytest.fixture(scope="class")
def evalhub_tenant_rbac_instance_name() -> str:  # noqa: UFN001
    """EvalHub CR name used when waiting for operator job RBAC in tenant namespaces."""
    return EVALHUB_MCP_CR_NAME


@pytest.fixture(scope="class")
def evalhub_tenant_deployment(evalhub_mcp_mt_deployment: Deployment) -> Deployment:  # noqa: UFN001
    """EvalHub deployment whose operator RBAC must be ready in tenant namespaces."""
    return evalhub_mcp_mt_deployment


@pytest.fixture(scope="class")
def evalhub_mcp_mt_cr(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    tenant_a_namespace: Namespace,
) -> Generator[EvalHub, Any, Any]:
    """Create an EvalHub CR with MCP enabled for integration tests."""
    if not is_evalhub_crd_available(admin_client):
        pytest.fail(
            "EvalHub CRD 'evalhubs.trustyai.opendatahub.io' not available on this cluster. "
            "Install the TrustyAI/EvalHub operator first."
        )

    # kind_dict is required: EvalHub's generated to_dict() has no "mcp" kwarg and
    # resets res["spec"] from its known attributes on every create() call.
    with EvalHub(
        client=admin_client,
        kind_dict={
            "apiVersion": f"{EvalHub.api_group}/v1",
            "kind": "EvalHub",
            "metadata": {
                "name": EVALHUB_MCP_CR_NAME,
                "namespace": model_namespace.name,
            },
            "spec": {
                "database": {"type": "sqlite"},
                "collections": ["leaderboard-v2"],
                "mcp": {
                    "enabled": True,
                    "replicas": 1,
                    "env": [
                        {
                            "name": "EVALHUB_TENANT",
                            "value": tenant_a_namespace.name,
                        }
                    ],
                },
            },
        },
        wait_for_resource=False,
    ) as evalhub:
        # Poll until the EvalHub operator reports the CR as ready.
        # Pending and None are expected and should not stop polling.
        for sample in TimeoutSampler(wait_timeout=300, sleep=2, func=lambda: evalhub.instance.status):
            if sample is None:
                continue
            if sample.get("ready") == "True":
                break
            phase = sample.get("phase", "")
            if phase == "Error":
                mcp_status = sample.get("mcp", {})
                pytest.fail(
                    f"EvalHub entered Error phase during setup.\n"
                    f"  Top-level status: {sample}\n"
                    f"  MCP sub-status:   {mcp_status}"
                )
        yield evalhub


@pytest.fixture(scope="class")
def evalhub_mcp_service_account(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    evalhub_mcp_mt_cr: EvalHub,
) -> ServiceAccount:
    """Wait for the operator-created EvalHub service account in the model namespace."""
    return wait_for_service_account(
        admin_client=admin_client,
        namespace=model_namespace.name,
        sa_name=_evalhub_service_account_name(EVALHUB_MCP_CR_NAME),
        timeout=120,
    )


@pytest.fixture(scope="class")
def evalhub_mcp_mt_cr_with_auth(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    tenant_a_namespace: Namespace,
    evalhub_mcp_mt_cr: EvalHub,
    evalhub_mcp_service_account: ServiceAccount,
) -> Generator[EvalHub, Any, Any]:
    """Patch the EvalHub CR with MCP auth secret configuration."""
    token = create_inference_token(model_service_account=evalhub_mcp_service_account)
    secret_name = _mcp_auth_secret_name(cr_name=EVALHUB_MCP_CR_NAME)
    with Secret(
        client=admin_client,
        name=secret_name,
        namespace=model_namespace.name,
        string_data={"token": token},
        wait_for_resource=False,
    ):
        evalhub_mcp_mt_cr.update(
            resource_dict={
                "metadata": {
                    "name": EVALHUB_MCP_CR_NAME,
                    "namespace": model_namespace.name,
                },
                "spec": {
                    "mcp": {
                        "enabled": True,
                        "replicas": 1,
                        "authSecret": secret_name,
                        "env": [
                            {
                                "name": "EVALHUB_TENANT",
                                "value": tenant_a_namespace.name,
                            }
                        ],
                    }
                },
            }
        )
        # Wait for the operator to reconcile *this* spec generation. Top-level
        # status.ready only reflects the main EvalHub deployment, so it is already
        # "True" before the operator has applied the MCP authSecret change.
        _wait_for_mcp_reconciled(
            evalhub=evalhub_mcp_mt_cr,
            generation=evalhub_mcp_mt_cr.instance.metadata.generation,
        )
        yield evalhub_mcp_mt_cr


@pytest.fixture(scope="class")
def evalhub_mcp_mt_deployment(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    evalhub_mcp_mt_cr_with_auth: EvalHub,
) -> Deployment:
    """Wait for the EvalHub MCP deployment rollout to complete."""
    deployment = Deployment(
        client=admin_client,
        name=_mcp_deployment_name(EVALHUB_MCP_CR_NAME),
        namespace=model_namespace.name,
    )
    deployment.wait_for_replicas(timeout=300)
    _wait_for_deployment_rollout(
        deployment=deployment,
        timeout=300,
        required_secret=_mcp_auth_secret_name(cr_name=EVALHUB_MCP_CR_NAME),
    )
    _wait_for_mcp_pods_settled(
        admin_client=admin_client,
        namespace=model_namespace.name,
        instance_name=EVALHUB_MCP_CR_NAME,
        desired=deployment.instance.spec.replicas or 1,
        timeout=120,
    )
    return deployment


@pytest.fixture(scope="class")
def evalhub_mcp_mt_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    evalhub_mcp_mt_deployment: Deployment,
) -> Route:
    """Get the Route for the EvalHub MCP service."""
    return Route(
        client=admin_client,
        name=_mcp_deployment_name(EVALHUB_MCP_CR_NAME),
        namespace=model_namespace.name,
        ensure_exists=True,
    )


@pytest.fixture(scope="class")
def evalhub_mcp_mt_ca_bundle_file(
    admin_client: DynamicClient,
) -> str:
    """CA bundle file for verifying TLS on the EvalHub MCP route."""
    return create_ca_bundle_file(client=admin_client)


@pytest.fixture(scope="class")
def evalhub_mcp_mt_ready(
    evalhub_mcp_mt_route: Route,
    evalhub_mcp_mt_ca_bundle_file: str,
) -> None:
    """Wait until the MCP health endpoint responds on the route."""
    url = f"https://{evalhub_mcp_mt_route.host}{EVALHUB_MCP_HEALTH_PATH}"
    host = evalhub_mcp_mt_route.host
    try:
        for sample in TimeoutSampler(
            wait_timeout=120,
            sleep=5,
            func=lambda: _probe_evalhub_mcp_health(
                url=url,
                host=host,
                ca_bundle_file=evalhub_mcp_mt_ca_bundle_file,
            ),
            exceptions_dict=_TRANSIENT_MCP_HEALTH_EXCEPTIONS,
        ):
            if sample.ok:
                LOGGER.info(f"EvalHub MCP at {host} is healthy")
                return
    except TimeoutExpiredError as err:
        if err.last_exp is not None:
            raise err.last_exp from err
        raise RuntimeError(f"EvalHub MCP at {host} did not become healthy within 120s") from err


@pytest.fixture(scope="class")
def evalhub_mcp_proxy_role(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[Role, Any, Any]:
    """Role in the EvalHub namespace granting evalhubs/proxy access to the MCP instance."""
    with Role(
        client=admin_client,
        name="evalhub-mcp-proxy-access",
        namespace=model_namespace.name,
        rules=build_mcp_proxy_role_rules(evalhub_instance_name=EVALHUB_MCP_CR_NAME),
        wait_for_resource=True,
    ) as role:
        yield role


@pytest.fixture(scope="class")
def evalhub_mcp_proxy_role_binding(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    tenant_a_service_account: ServiceAccount,
    evalhub_mcp_proxy_role: Role,
) -> Generator[RoleBinding, Any, Any]:
    """Bind MCP proxy access to the tenant-a test ServiceAccount."""
    with RoleBinding(
        client=admin_client,
        name="evalhub-mcp-proxy-binding",
        namespace=model_namespace.name,
        subjects_kind="ServiceAccount",
        subjects_name=tenant_a_service_account.name,
        subjects_namespace=tenant_a_service_account.namespace,
        role_ref_kind="Role",
        role_ref_name=evalhub_mcp_proxy_role.name,
        wait_for_resource=True,
    ) as binding:
        yield binding


@pytest.fixture(scope="class")
def mcp_server_tenant_rbac(
    admin_client: DynamicClient,
    tenant_a_namespace: Namespace,
    model_namespace: Namespace,
    evalhub_mcp_mt_deployment: Deployment,
) -> Generator[RoleBinding, Any, Any]:
    """Grant MCP server service account access to tenant namespace EvalHub API resources.

    This is a workaround until the EvalHub operator automatically provisions these permissions.
    The MCP server needs to access evaluations, providers, benchmarks, and collections in the
    tenant namespace on behalf of authenticated users.
    """
    mcp_server_sa_name = _evalhub_service_account_name(cr_name=EVALHUB_MCP_CR_NAME)

    # Create role in tenant namespace granting EvalHub API access
    with (
        Role(
            client=admin_client,
            name=f"{EVALHUB_MCP_CR_NAME}-server-access",
            namespace=tenant_a_namespace.name,
            rules=EVALHUB_USER_ROLE_RULES,  # Same permissions as test user
            wait_for_resource=True,
        ) as role,
        RoleBinding(
            client=admin_client,
            name=f"{EVALHUB_MCP_CR_NAME}-server-binding",
            namespace=tenant_a_namespace.name,
            subjects_kind="ServiceAccount",
            subjects_name=mcp_server_sa_name,
            subjects_namespace=model_namespace.name,
            role_ref_kind="Role",
            role_ref_name=role.name,
            wait_for_resource=True,
        ) as binding,
    ):
        yield binding


@pytest.fixture(scope="class")
def evalhub_mcp_client(
    tenant_a_token: str,
    tenant_a_namespace: Namespace,
    evalhub_mcp_mt_route: Route,
    evalhub_mcp_mt_ca_bundle_file: str,
    evalhub_mcp_proxy_role_binding: RoleBinding,
    evalhub_mcp_mt_ready: None,
    mcp_server_tenant_rbac: RoleBinding,
) -> EvalHubMcpClient:
    """Authenticated MCP client for tenant-a."""
    client = EvalHubMcpClient(
        host=evalhub_mcp_mt_route.host,
        token=tenant_a_token,
        ca_bundle_file=evalhub_mcp_mt_ca_bundle_file,
        tenant=tenant_a_namespace.name,
    )
    client.initialize()
    return client
