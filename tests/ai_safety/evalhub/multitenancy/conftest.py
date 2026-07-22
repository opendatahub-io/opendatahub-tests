import pytest
import requests
import structlog
from ocp_resources.route import Route
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.evalhub.constants import EVALHUB_HEALTHZ_PATH

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
