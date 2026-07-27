"""Fixtures for EvalHub single-tenancy integration tests.

All fixtures are self-contained — they do not depend on any multitenancy fixture.
The SingleTenantEvalHub subclass injects spec.tenancy: single because the auto-generated
ocp_resources.EvalHub class has no tenancy parameter in its constructor.
"""

from collections.abc import Generator
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.evalhub import EvalHub
from ocp_resources.namespace import Namespace
from ocp_resources.role_binding import RoleBinding
from ocp_resources.route import Route
from ocp_resources.service_account import ServiceAccount
from timeout_sampler import TimeoutSampler

from tests.ai_safety.evalhub.constants import EVALHUB_HEALTH_PATH, EVALHUB_TENANT_LABEL_KEY, EVALHUB_TENANT_LABEL_VALUE
from tests.ai_safety.evalhub.single_tenancy.constants import (
    EVALHUB_ST_CR_NAME,
    EVALHUB_USER_ROLE_NAME,
)
from tests.ai_safety.evalhub.single_tenancy.utils import SingleTenantEvalHub, _is_evalhub_crd_available
from tests.ai_safety.evalhub.utils import TRANSIENT_HEALTH_EXCEPTIONS, probe_evalhub_health_endpoint
from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import Timeout
from utilities.infra import create_inference_token, create_ns

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="class")
def evalhub_st_cr(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[SingleTenantEvalHub, Any, Any]:
    """Create a single-tenant EvalHub CR (spec.tenancy: single, sqlite database)."""
    if not _is_evalhub_crd_available(admin_client):
        pytest.fail(
            "EvalHub CRD 'evalhubs.trustyai.opendatahub.io' not available on this cluster. "
            "Install the TrustyAI/EvalHub operator first."
        )

    with SingleTenantEvalHub(
        client=admin_client,
        name=EVALHUB_ST_CR_NAME,
        namespace=model_namespace.name,
        database={"type": "sqlite"},
        wait_for_resource=True,
    ) as evalhub:
        yield evalhub


@pytest.fixture(scope="class")
def evalhub_st_deployment(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    evalhub_st_cr: SingleTenantEvalHub,
) -> Deployment:
    """Wait for the single-tenant EvalHub deployment to become ready."""
    deployment = Deployment(
        client=admin_client,
        name=evalhub_st_cr.name,
        namespace=model_namespace.name,
    )
    deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
    return deployment


@pytest.fixture(scope="class")
def evalhub_st_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    evalhub_st_deployment: Deployment,
) -> Route:
    """OpenShift Route for the single-tenant EvalHub instance."""
    return Route(
        client=admin_client,
        name=evalhub_st_deployment.name,
        namespace=model_namespace.name,
        ensure_exists=True,
    )


@pytest.fixture(scope="class")
def evalhub_st_ca_bundle_file(
    admin_client: DynamicClient,
) -> str:
    """CA bundle file for TLS verification of the single-tenant EvalHub route."""
    return create_ca_bundle_file(client=admin_client)


@pytest.fixture(scope="class")
def evalhub_st_ready(
    evalhub_st_route: Route,
    evalhub_st_ca_bundle_file: str,
) -> None:
    """Poll the health endpoint until the single-tenant EvalHub responds successfully.

    The deployment may report ready before the OpenShift router backend is configured,
    causing transient 503 errors. This fixture waits until the route is actually serving.
    """
    url = f"https://{evalhub_st_route.host}{EVALHUB_HEALTH_PATH}"
    host = evalhub_st_route.host
    for sample in TimeoutSampler(
        wait_timeout=120,
        sleep=5,
        func=lambda: probe_evalhub_health_endpoint(
            url=url,
            host=host,
            ca_bundle_file=evalhub_st_ca_bundle_file,
        ),
        exceptions_dict=TRANSIENT_HEALTH_EXCEPTIONS,
    ):
        if sample.ok:
            LOGGER.info(f"Single-tenant EvalHub at {host} is healthy")
            return


@pytest.fixture(scope="class")
def evalhub_st_user_sa(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    evalhub_st_cr: SingleTenantEvalHub,
) -> Generator[ServiceAccount, Any, Any]:
    """ServiceAccount bound to the operator-created evalhub-user Role.

    The operator binds system:serviceaccounts:{ns} to evalhub-tenant-admin in single mode,
    so any SA in the namespace already has admin access. We still create an explicit SA
    so tests have a concrete identity to token-review.
    """
    with ServiceAccount(
        client=admin_client,
        name="evalhub-test-user",
        namespace=model_namespace.name,
        wait_for_resource=True,
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def evalhub_st_user_role_binding(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    evalhub_st_user_sa: ServiceAccount,
    evalhub_st_cr: SingleTenantEvalHub,
) -> Generator[RoleBinding, Any, Any]:
    """Explicit RoleBinding granting the test SA the evalhub-user Role.

    The operator already binds system:serviceaccounts:{ns} to evalhub-tenant-admin,
    so this binding is technically redundant for access — but it pins the test to the
    evalhub-user Role so we verify that specific role works for API calls.
    """
    with RoleBinding(
        client=admin_client,
        name="evalhub-test-user-binding",
        namespace=model_namespace.name,
        role_ref_kind="Role",
        role_ref_name=EVALHUB_USER_ROLE_NAME,
        subjects_kind="ServiceAccount",
        subjects_name=evalhub_st_user_sa.name,
        subjects_namespace=model_namespace.name,
        wait_for_resource=True,
    ) as binding:
        yield binding


@pytest.fixture(scope="class")
def evalhub_st_user_token(
    evalhub_st_user_sa: ServiceAccount,
    evalhub_st_user_role_binding: RoleBinding,
) -> str:
    """Bearer token for the single-tenant test user ServiceAccount."""
    return create_inference_token(model_service_account=evalhub_st_user_sa)


@pytest.fixture(scope="class")
def second_namespace(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[Namespace, Any, Any]:
    """An unlabeled namespace used to assert that single-tenant EvalHub creates no resources there."""
    with create_ns(
        admin_client=admin_client,
        name=f"{model_namespace.name}-other",
    ) as ns:
        yield ns


@pytest.fixture(scope="class")
def labeled_tenant_namespace(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[Namespace, Any, Any]:
    """A namespace labelled as a multi-tenant EvalHub tenant.

    Used by mode-switch tests to verify cross-namespace RBAC provisioning when
    an instance switches to multi mode, and by invalid-placement tests.
    """
    with create_ns(
        admin_client=admin_client,
        name=f"{model_namespace.name}-tenant",
        labels={EVALHUB_TENANT_LABEL_KEY: EVALHUB_TENANT_LABEL_VALUE},
    ) as ns:
        yield ns


@pytest.fixture(scope="class")
def evalhub_mt_for_switch(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[EvalHub, Any, Any]:
    """A standard (multi-tenant) EvalHub CR for mode-switch tests.

    Starts in multi mode (default). Tests can patch spec.tenancy to single
    and verify that Roles are created and cross-NS RBAC is cleaned up.
    """
    with EvalHub(
        client=admin_client,
        name="evalhub-mt-switch",
        namespace=model_namespace.name,
        database={"type": "sqlite"},
        wait_for_resource=True,
    ) as evalhub:
        yield evalhub


@pytest.fixture(scope="class")
def evalhub_mt_switch_deployment(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    evalhub_mt_for_switch: EvalHub,
) -> Deployment:
    """Wait for the multi-tenant EvalHub (used in mode-switch tests) to be ready."""
    deployment = Deployment(
        client=admin_client,
        name=evalhub_mt_for_switch.name,
        namespace=model_namespace.name,
    )
    deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
    return deployment
