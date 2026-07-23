import pytest
import requests
import structlog
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.service import Service
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.evalhub.constants import EVALHUB_HEALTHZ_PATH, EVALHUB_LOG_MAX_TAIL_LINES
from tests.ai_safety.evalhub.utils import (
    assert_plain_text_logs_response,
    build_evalhub_job_payload,
    get_evalhub_job_logs_http,
    submit_evalhub_job,
    wait_for_evalhub_job,
)

LOGGER = structlog.get_logger(name=__name__)


# ---------------------------------------------------------------------------
# Note: evalhub_mt_cr, evalhub_mt_deployment, evalhub_mt_route,
# evalhub_mt_ca_bundle_file, tenant_a_rbac_ready, evalhub_vllm_emulator_deployment,
# and evalhub_vllm_emulator_service fixtures are defined in ../conftest.py (parent)
# and shared across all evalhub test subdirectories.
#
# Multi-tenancy test user fixtures (tenant_a_namespace, tenant_a_service_account,
# tenant_a_evalhub_role, tenant_a_evalhub_role_binding, tenant_a_token) live in
# tests/fixtures/trustyai.py.
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

    Uses /healthz instead of /api/v1/health because this readiness wait
    runs before any tenant namespace is created, and /api/v1/health
    requires X-Tenant in cluster mode.
    """
    url = f"https://{evalhub_mt_route.host}{EVALHUB_HEALTHZ_PATH}"
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


@pytest.fixture(scope="class")
def evalhub_logs_completed_job_id(
    tenant_a_token: str,
    tenant_a_namespace: Namespace,
    evalhub_mt_ca_bundle_file: str,
    evalhub_mt_route: Route,
    evalhub_vllm_emulator_service: Service,
) -> str:
    """Submit one arc_easy job and wait for completion (shared by log retrieval tests)."""
    payload = build_evalhub_job_payload(
        model_service_name=evalhub_vllm_emulator_service.name,
        tenant_namespace=tenant_a_namespace.name,
        job_name="evalhub-logs-completed-job",
    )
    data = submit_evalhub_job(
        host=evalhub_mt_route.host,
        token=tenant_a_token,
        ca_bundle_file=evalhub_mt_ca_bundle_file,
        tenant=tenant_a_namespace.name,
        payload=payload,
    )
    job_id = data["resource"]["id"]
    job_result = wait_for_evalhub_job(
        host=evalhub_mt_route.host,
        token=tenant_a_token,
        ca_bundle_file=evalhub_mt_ca_bundle_file,
        tenant=tenant_a_namespace.name,
        job_id=job_id,
        timeout=600,
    )
    assert job_result.get("status", {}).get("state") == "completed"
    return job_id


@pytest.fixture(scope="class")
def evalhub_logs_completed_job_logs(
    tenant_a_token: str,
    tenant_a_namespace: Namespace,
    evalhub_mt_ca_bundle_file: str,
    evalhub_mt_route: Route,
    evalhub_logs_completed_job_id: str,
) -> str:
    """Validated full job logs for a completed evaluation job (fetched once per class)."""
    response = get_evalhub_job_logs_http(
        host=evalhub_mt_route.host,
        token=tenant_a_token,
        ca_bundle_file=evalhub_mt_ca_bundle_file,
        tenant=tenant_a_namespace.name,
        job_id=evalhub_logs_completed_job_id,
        params={"tail_lines": str(EVALHUB_LOG_MAX_TAIL_LINES)},
    )
    return assert_plain_text_logs_response(response=response)


@pytest.fixture(scope="class")
def evalhub_logs_completed_benchmark_logs(
    tenant_a_token: str,
    tenant_a_namespace: Namespace,
    evalhub_mt_ca_bundle_file: str,
    evalhub_mt_route: Route,
    evalhub_logs_completed_job_id: str,
) -> str:
    """Validated benchmark logs for a completed evaluation job (fetched once per class)."""
    response = get_evalhub_job_logs_http(
        host=evalhub_mt_route.host,
        token=tenant_a_token,
        ca_bundle_file=evalhub_mt_ca_bundle_file,
        tenant=tenant_a_namespace.name,
        job_id=evalhub_logs_completed_job_id,
        benchmark_index=0,
    )
    return assert_plain_text_logs_response(response=response)
