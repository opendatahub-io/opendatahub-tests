from collections.abc import Generator
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.evalhub import EvalHub
from ocp_resources.namespace import Namespace
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.route import Route
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.evalhub.constants import (
    EVALHUB_JOBS_WRITER_CLUSTERROLE,
    EVALHUB_PROVIDERS_PATH,
    EVALHUB_TENANT_LABEL_KEY,
    EVALHUB_USER_ROLE_RULES,
    EVALHUB_VLLM_EMULATOR_PORT,
    VLLM_EMULATOR_IMAGE,
)
from tests.ai_safety.evalhub.k8s_lifecycle_signals.constants import (
    LIFECYCLE_BAD_IMAGE,
    LIFECYCLE_LM_EVAL_K8S_ENTRYPOINT,
    LIFECYCLE_SIGNALS_CP_NAMESPACE,
    LIFECYCLE_SIGNALS_CR_NAME,
    LIFECYCLE_SIGNALS_NAMESPACE,
)
from tests.ai_safety.evalhub.kueue.constants import VLLM_EMULATOR
from tests.ai_safety.evalhub.utils import (
    TRANSIENT_HEALTH_EXCEPTIONS,
    build_headers,
    is_evalhub_crd_available,
    probe_evalhub_health_endpoint,
    tenant_rbac_ready,
)
from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import Labels, Protocols, Timeout
from utilities.infra import create_inference_token, create_ns

LOGGER = structlog.get_logger(name=__name__)

_VLLM_SERVICE_NAME = f"{VLLM_EMULATOR}-service"


def _cleanup_ns_if_exists(admin_client: DynamicClient, name: str) -> None:
    """Delete a namespace left over from a previous run, if it still exists."""
    ns = Namespace(client=admin_client, name=name)
    if ns.exists:
        LOGGER.warning(f"Namespace {name!r} already exists — deleting before test setup")
        ns.delete(wait=True)


_USER_SA_NAME = "evalhub-lifecycle-user"
_USER_ROLE_NAME = "evalhub-lifecycle-user-role"
_USER_BINDING_NAME = "evalhub-lifecycle-user-binding"
_USER_JOBS_WRITER_BINDING = "evalhub-lifecycle-jobs-writer-binding"


@pytest.fixture(scope="session")
def lifecycle_signals_cp_namespace(admin_client: DynamicClient) -> Generator[Namespace, Any, Any]:
    """Control plane namespace — EvalHub CR and Deployment live here."""
    _cleanup_ns_if_exists(admin_client=admin_client, name=LIFECYCLE_SIGNALS_CP_NAMESPACE)
    with create_ns(
        admin_client=admin_client,
        name=LIFECYCLE_SIGNALS_CP_NAMESPACE,
    ) as ns:
        yield ns


@pytest.fixture(scope="session")
def lifecycle_signals_namespace(admin_client: DynamicClient) -> Generator[Namespace, Any, Any]:
    """Tenant namespace — workloads run here (labelled with tenant label)."""
    _cleanup_ns_if_exists(admin_client=admin_client, name=LIFECYCLE_SIGNALS_NAMESPACE)
    with create_ns(
        admin_client=admin_client,
        name=LIFECYCLE_SIGNALS_NAMESPACE,
        labels={EVALHUB_TENANT_LABEL_KEY: "true"},
    ) as ns:
        yield ns


@pytest.fixture(scope="session")
def lifecycle_signals_evalhub_cr(
    admin_client: DynamicClient,
    lifecycle_signals_cp_namespace: Namespace,
) -> Generator[EvalHub, Any, Any]:
    """Single EvalHub CR in the control plane namespace, shared across all lifecycle signal tests."""
    if not is_evalhub_crd_available(admin_client):
        pytest.fail(
            "EvalHub CRD 'evalhubs.trustyai.opendatahub.io' not available. Install the TrustyAI/EvalHub operator first."
        )
    with EvalHub(
        client=admin_client,
        name=LIFECYCLE_SIGNALS_CR_NAME,
        namespace=lifecycle_signals_cp_namespace.name,
        database={"type": "sqlite"},
        wait_for_resource=True,
    ) as evalhub:
        yield evalhub


@pytest.fixture(scope="session")
def lifecycle_signals_deployment(
    admin_client: DynamicClient,
    lifecycle_signals_cp_namespace: Namespace,
    lifecycle_signals_evalhub_cr: EvalHub,
) -> Deployment:
    """Wait for the EvalHub Deployment in the control plane namespace."""
    deployment = Deployment(
        client=admin_client,
        name=lifecycle_signals_evalhub_cr.name,
        namespace=lifecycle_signals_cp_namespace.name,
    )
    deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
    return deployment


@pytest.fixture(scope="session")
def lifecycle_signals_tenant_rbac(
    admin_client: DynamicClient,
    lifecycle_signals_namespace: Namespace,
    lifecycle_signals_deployment: Deployment,
) -> None:
    """Wait for operator to provision tenant RBAC in the lifecycle signals namespace."""
    try:
        for ready in TimeoutSampler(
            wait_timeout=120,
            sleep=5,
            func=tenant_rbac_ready,
            admin_client=admin_client,
            namespace=lifecycle_signals_namespace.name,
            evalhub_instance_name=LIFECYCLE_SIGNALS_CR_NAME,
        ):
            if ready:
                LOGGER.info(f"Operator RBAC provisioned in {lifecycle_signals_namespace.name}")
                return
    except TimeoutExpiredError as exc:
        raise RuntimeError(f"Operator RBAC not provisioned in {lifecycle_signals_namespace.name} within 120s") from exc


@pytest.fixture(scope="class")
def lifecycle_signals_tenant_a_rbac(
    admin_client: DynamicClient,
    tenant_a_namespace: Namespace,
    lifecycle_signals_deployment: Deployment,
) -> None:
    """Wait for operator to provision EvalHub job RBAC in tenant-a for lifecycle signals."""
    try:
        for ready in TimeoutSampler(
            wait_timeout=120,
            sleep=5,
            func=tenant_rbac_ready,
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_instance_name=LIFECYCLE_SIGNALS_CR_NAME,
        ):
            if ready:
                LOGGER.info(f"Operator RBAC provisioned in {tenant_a_namespace.name}")
                return
    except TimeoutExpiredError as exc:
        raise RuntimeError(f"Operator RBAC not provisioned in {tenant_a_namespace.name} within 120s") from exc


@pytest.fixture(scope="session")
def lifecycle_signals_route(
    admin_client: DynamicClient,
    lifecycle_signals_deployment: Deployment,
    lifecycle_signals_cp_namespace: Namespace,
) -> Route:
    """Route for the shared EvalHub instance (lives in the control plane namespace)."""
    return Route(
        client=admin_client,
        name=lifecycle_signals_deployment.name,
        namespace=lifecycle_signals_cp_namespace.name,
        ensure_exists=True,
    )


@pytest.fixture(scope="session")
def lifecycle_signals_ca_bundle_file(admin_client: DynamicClient) -> str:
    """CA bundle file for verifying TLS on the shared EvalHub route."""
    return create_ca_bundle_file(client=admin_client)


@pytest.fixture(scope="session")
def lifecycle_signals_vllm_deployment(
    admin_client: DynamicClient,
    lifecycle_signals_namespace: Namespace,
    lifecycle_signals_tenant_rbac: None,
) -> Generator[Deployment, Any, Any]:
    """vLLM emulator deployment in the lifecycle signals namespace."""
    label = {Labels.Openshift.APP: VLLM_EMULATOR}
    with Deployment(
        client=admin_client,
        namespace=lifecycle_signals_namespace.name,
        name=VLLM_EMULATOR,
        label=label,
        selector={"matchLabels": label},
        template={
            "metadata": {"labels": label, "name": VLLM_EMULATOR},
            "spec": {
                "containers": [
                    {
                        "image": VLLM_EMULATOR_IMAGE,
                        "name": VLLM_EMULATOR,
                        "ports": [{"containerPort": EVALHUB_VLLM_EMULATOR_PORT, "protocol": Protocols.TCP}],
                        "readinessProbe": {
                            "tcpSocket": {"port": EVALHUB_VLLM_EMULATOR_PORT},
                            "initialDelaySeconds": 5,
                            "periodSeconds": 5,
                            "timeoutSeconds": 3,
                            "failureThreshold": 6,
                        },
                        "securityContext": {
                            "allowPrivilegeEscalation": False,
                            "runAsNonRoot": True,
                            "capabilities": {"drop": ["ALL"]},
                            "seccompProfile": {"type": "RuntimeDefault"},
                        },
                    }
                ]
            },
        },
        replicas=1,
    ) as deployment:
        deployment.wait_for_replicas(timeout=300)
        yield deployment


@pytest.fixture(scope="session")
def lifecycle_signals_vllm_service(
    admin_client: DynamicClient,
    lifecycle_signals_namespace: Namespace,
    lifecycle_signals_vllm_deployment: Deployment,
) -> Generator[Service, Any, Any]:
    """Service fronting the vLLM emulator in the lifecycle signals namespace."""
    with Service(
        client=admin_client,
        namespace=lifecycle_signals_namespace.name,
        name=_VLLM_SERVICE_NAME,
        ports=[
            {
                "name": f"{VLLM_EMULATOR}-endpoint",
                "port": EVALHUB_VLLM_EMULATOR_PORT,
                "protocol": Protocols.TCP,
                "targetPort": EVALHUB_VLLM_EMULATOR_PORT,
            }
        ],
        selector={Labels.Openshift.APP: VLLM_EMULATOR},
    ) as service:
        yield service


@pytest.fixture(scope="session")
def lifecycle_signals_token(
    admin_client: DynamicClient,
    lifecycle_signals_namespace: Namespace,
) -> Generator[str, Any, Any]:
    """ServiceAccount token for EvalHub API access in lifecycle signal tests."""
    with (
        ServiceAccount(
            client=admin_client,
            name=_USER_SA_NAME,
            namespace=lifecycle_signals_namespace.name,
            wait_for_resource=True,
        ) as sa,
        Role(
            client=admin_client,
            name=_USER_ROLE_NAME,
            namespace=lifecycle_signals_namespace.name,
            rules=EVALHUB_USER_ROLE_RULES,
            wait_for_resource=True,
        ) as role,
        RoleBinding(
            client=admin_client,
            name=_USER_BINDING_NAME,
            namespace=lifecycle_signals_namespace.name,
            subjects_kind="ServiceAccount",
            subjects_name=sa.name,
            subjects_namespace=lifecycle_signals_namespace.name,
            role_ref_kind="Role",
            role_ref_name=role.name,
            wait_for_resource=True,
        ),
        RoleBinding(
            client=admin_client,
            name=_USER_JOBS_WRITER_BINDING,
            namespace=lifecycle_signals_namespace.name,
            subjects_kind="ServiceAccount",
            subjects_name=sa.name,
            subjects_namespace=lifecycle_signals_namespace.name,
            role_ref_kind="ClusterRole",
            role_ref_name=EVALHUB_JOBS_WRITER_CLUSTERROLE,
            wait_for_resource=True,
        ),
    ):
        yield create_inference_token(model_service_account=sa)


@pytest.fixture(scope="session")
def lifecycle_signals_ready(
    lifecycle_signals_route: Route,
    lifecycle_signals_ca_bundle_file: str,
) -> None:
    """Gate fixture: confirms EvalHub is healthy before any lifecycle signal test runs."""
    from tests.ai_safety.evalhub.constants import EVALHUB_HEALTH_PATH

    host = lifecycle_signals_route.host
    url = f"https://{host}{EVALHUB_HEALTH_PATH}"
    for sample in TimeoutSampler(
        wait_timeout=120,
        sleep=5,
        func=lambda: probe_evalhub_health_endpoint(
            url=url,
            host=host,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
        ),
        exceptions_dict=TRANSIENT_HEALTH_EXCEPTIONS,
    ):
        if sample.ok:
            LOGGER.info(f"EvalHub at {host} is healthy")
            return


@pytest.fixture(scope="session")
def lifecycle_signals_bad_image_provider(
    lifecycle_signals_ready: None,
    lifecycle_signals_route: Route,
    lifecycle_signals_ca_bundle_file: str,
    lifecycle_signals_token: str,
    lifecycle_signals_namespace: Namespace,
) -> Generator[str, Any, Any]:
    """Tenant-scoped provider whose k8s runtime image cannot be pulled (ImagePullBackOff)."""
    host = lifecycle_signals_route.host
    tenant = lifecycle_signals_namespace.name
    headers = build_headers(token=lifecycle_signals_token, tenant=tenant)
    provider_payload = {
        "name": "Lifecycle signals bad-image provider",
        "benchmarks": [
            {
                "id": "arc_easy",
                "name": "lm_evaluation_harness",
            },
        ],
        "runtime": {
            "k8s": {
                "image": LIFECYCLE_BAD_IMAGE,
                "entrypoint": LIFECYCLE_LM_EVAL_K8S_ENTRYPOINT,
                "cpu_request": "100m",
                "memory_request": "128Mi",
                "cpu_limit": "500m",
                "memory_limit": "4Gi",
            },
        },
    }
    create_resp = requests.post(
        url=f"https://{host}{EVALHUB_PROVIDERS_PATH}",
        headers=headers,
        json=provider_payload,
        verify=lifecycle_signals_ca_bundle_file,
        timeout=Timeout.TIMEOUT_30SEC,
    )
    provider_id: str | None = None
    try:
        if create_resp.status_code != 201:
            pytest.fail(f"Failed to create bad-image provider: {create_resp.status_code} {create_resp.text}")
        provider_id = create_resp.json()["resource"]["id"]
        yield provider_id
    finally:
        if provider_id is not None:
            requests.delete(
                url=f"https://{host}{EVALHUB_PROVIDERS_PATH}/{provider_id}",
                headers=headers,
                verify=lifecycle_signals_ca_bundle_file,
                timeout=Timeout.TIMEOUT_30SEC,
            )
