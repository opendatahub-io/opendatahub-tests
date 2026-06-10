from collections.abc import Generator
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.service import Service
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.evalhub.constants import (
    EVALHUB_HEALTH_PATH,
    EVALHUB_VLLM_EMULATOR_PORT,
)
from tests.ai_safety.evalhub.utils import tenant_rbac_ready
from utilities.constants import Labels, Protocols, Timeout

LOGGER = structlog.get_logger(name=__name__)


# ---------------------------------------------------------------------------
# Note: evalhub_mt_cr, evalhub_mt_deployment, evalhub_mt_route, and
# evalhub_mt_ca_bundle_file fixtures are defined in ../conftest.py (parent)
# and shared across all evalhub test subdirectories.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def evalhub_mt_ready(
    evalhub_mt_route: Route,
    evalhub_mt_ca_bundle_file: str,
) -> None:
    """Wait for the EvalHub service to respond via its route.

    The deployment may report ready replicas before the OpenShift router
    has fully configured the backend, causing 503 errors. This fixture
    polls the health endpoint until it responds successfully.
    """
    url = f"https://{evalhub_mt_route.host}{EVALHUB_HEALTH_PATH}"
    try:
        for sample in TimeoutSampler(
            wait_timeout=120,
            sleep=5,
            func=lambda: requests.get(url, verify=evalhub_mt_ca_bundle_file, timeout=10),
            exceptions_dict={Exception: []},
        ):
            if sample.ok:
                LOGGER.info(f"EvalHub at {evalhub_mt_route.host} is healthy")
                return
    except TimeoutExpiredError as err:
        raise RuntimeError(f"EvalHub at {evalhub_mt_route.host} did not become healthy within 120s") from err


# ---------------------------------------------------------------------------
# Wait for operator to provision tenant RBAC
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def tenant_a_rbac_ready(
    admin_client: DynamicClient,
    tenant_a_namespace: Namespace,
    evalhub_mt_deployment: Deployment,
) -> None:
    """Wait for the operator to provision job RBAC in tenant-a.

    The operator watches for namespaces with the tenant label and
    creates jobs-writer + job-config RoleBindings. This fixture
    blocks until those RoleBindings exist.
    """
    try:
        for ready in TimeoutSampler(
            wait_timeout=120,
            sleep=5,
            func=tenant_rbac_ready,
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
        ):
            if ready:
                LOGGER.info(f"Operator RBAC provisioned in {tenant_a_namespace.name}")
                return
    except TimeoutExpiredError as err:
        msg = (
            f"Operator RBAC provision failed: RoleBindings, ServiceAccount, or service-CA ConfigMap"
            f" not found in namespace '{tenant_a_namespace.name}' within timeout"
        )
        LOGGER.error(msg)
        raise RuntimeError(msg) from err


# ---------------------------------------------------------------------------
# vLLM emulator (deployed in tenant-a for job submission tests)
# ---------------------------------------------------------------------------

VLLM_EMULATOR: str = "vllm-emulator"
VLLM_EMULATOR_IMAGE: str = (
    "quay.io/trustyai_testing/vllm_emulator@sha256:c4bdd5bb93171dee5b4c8454f36d7c42b58b2a4ceb74f29dba5760ac53b5c12d"
)


@pytest.fixture(scope="class")
def evalhub_vllm_emulator_deployment(
    admin_client: DynamicClient,
    tenant_a_namespace: Namespace,
    tenant_a_rbac_ready: None,
) -> Generator[Deployment, Any, Any]:
    """Deploy the vLLM emulator in tenant-a.

    Depends on tenant_a_rbac_ready to ensure the operator has provisioned
    the jobs-writer and job-config RoleBindings before any job is submitted.
    """
    label = {Labels.Openshift.APP: VLLM_EMULATOR}
    with Deployment(
        client=admin_client,
        namespace=tenant_a_namespace.name,
        name=VLLM_EMULATOR,
        label=label,
        selector={"matchLabels": label},
        template={
            "metadata": {
                "labels": label,
                "name": VLLM_EMULATOR,
            },
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
                            "capabilities": {"drop": ["ALL"]},
                            "seccompProfile": {"type": "RuntimeDefault"},
                        },
                    }
                ]
            },
        },
        replicas=1,
    ) as deployment:
        deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
        yield deployment


@pytest.fixture(scope="class")
def evalhub_vllm_emulator_service(
    admin_client: DynamicClient,
    tenant_a_namespace: Namespace,
    evalhub_vllm_emulator_deployment: Deployment,
) -> Generator[Service, Any, Any]:
    """Service fronting the vLLM emulator in tenant-a."""
    with Service(
        client=admin_client,
        namespace=tenant_a_namespace.name,
        name=f"{VLLM_EMULATOR}-service",
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
