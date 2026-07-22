from collections.abc import Generator
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.gateway_gateway_networking_k8s_io import Gateway
from ocp_resources.maas_auth_policy import MaaSAuthPolicy
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.maas_subscription import MaaSSubscription
from ocp_resources.namespace import Namespace

from tests.model_serving.maas_billing.maas_subscription.utils import (
    create_maas_subscription,
)
from tests.model_serving.maas_billing.upgrade.utils import (
    MaaSBaseline,
    capture_maas_baseline,
    load_maas_baseline_from_configmap,
    save_maas_baseline_to_configmap,
)
from tests.model_serving.maas_billing.utils import host_from_ingress_domain
from utilities.constants import MAAS_GATEWAY_NAME, MAAS_GATEWAY_NAMESPACE
from utilities.infra import create_ns
from utilities.resources.maastenantconfig import MaasTenantConfig

LOGGER = structlog.get_logger(name=__name__)

MAAS_UPGRADE_NAMESPACE = "upgrade-maas"
MAAS_UPGRADE_MODEL_NAME = "upgrade-maas-model-ref"
MAAS_UPGRADE_AUTH_POLICY_NAME = "upgrade-maas-auth-policy"
MAAS_UPGRADE_SUBSCRIPTION_NAME = "upgrade-maas-subscription"


@pytest.fixture(scope="session")
def maas_upgrade_namespace(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    """Dedicated namespace for MaaS upgrade test resources."""
    namespace = Namespace(client=admin_client, name=MAAS_UPGRADE_NAMESPACE)
    if pytestconfig.option.post_upgrade:
        yield namespace
        namespace.clean_up()
    else:
        with create_ns(
            admin_client=admin_client,
            name=MAAS_UPGRADE_NAMESPACE,
            model_mesh_enabled=False,
            add_dashboard_label=True,
            teardown=teardown_resources,
        ) as namespace:
            yield namespace


@pytest.fixture(scope="session")
def maas_upgrade_gateway(
    admin_client: DynamicClient,
    maas_gateway_api: None,
) -> Gateway:
    """Return the MaaS Gateway object for upgrade test assertions.

    Depends on maas_gateway_api to ensure the Gateway exists before returning it.
    """
    return Gateway(
        client=admin_client,
        name=MAAS_GATEWAY_NAME,
        namespace=MAAS_GATEWAY_NAMESPACE,
        ensure_exists=True,
    )


@pytest.fixture(scope="session")
def maas_upgrade_base_url(admin_client: DynamicClient) -> str:
    """Session-scoped MaaS API base URL derived from the cluster ingress domain."""
    gateway_hostname = host_from_ingress_domain(client=admin_client)
    return f"https://{gateway_hostname}/maas-api"


@pytest.fixture(scope="session")
def maas_upgrade_model_ref(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    maas_upgrade_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[MaaSModelRef, Any, Any]:
    """MaaSModelRef deployed pre-upgrade and referenced for post-upgrade validation."""
    model_ref_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": MAAS_UPGRADE_MODEL_NAME,
        "namespace": maas_upgrade_namespace.name,
    }
    if pytestconfig.option.post_upgrade:
        model_ref = MaaSModelRef(**model_ref_kwargs)
        yield model_ref
        model_ref.clean_up()
    else:
        with MaaSModelRef(
            **model_ref_kwargs,
            model_ref={
                "name": MAAS_UPGRADE_MODEL_NAME,
                "namespace": maas_upgrade_namespace.name,
                "kind": "LLMInferenceService",
            },
            teardown=teardown_resources,
            wait_for_resource=True,
        ) as model_ref:
            yield model_ref


@pytest.fixture(scope="session")
def maas_upgrade_auth_policy(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    maas_upgrade_model_ref: MaaSModelRef,
    maas_subscription_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[MaaSAuthPolicy, Any, Any]:
    """MaaSAuthPolicy deployed pre-upgrade and referenced for post-upgrade validation."""
    auth_policy_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": MAAS_UPGRADE_AUTH_POLICY_NAME,
        "namespace": maas_subscription_namespace.name,
    }
    if pytestconfig.option.post_upgrade:
        auth_policy = MaaSAuthPolicy(**auth_policy_kwargs)
        yield auth_policy
        auth_policy.clean_up()
    else:
        with MaaSAuthPolicy(
            **auth_policy_kwargs,
            model_refs=[
                {
                    "name": maas_upgrade_model_ref.name,
                    "namespace": maas_upgrade_model_ref.namespace,
                }
            ],
            subjects={"groups": [{"name": "system:authenticated"}]},
            teardown=teardown_resources,
            wait_for_resource=True,
        ) as auth_policy:
            yield auth_policy


@pytest.fixture(scope="session")
def maas_upgrade_subscription(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    maas_upgrade_model_ref: MaaSModelRef,
    maas_subscription_namespace: Namespace,
    maas_subscription_controller_enabled_latest: DataScienceCluster,
    teardown_resources: bool,
) -> Generator[MaaSSubscription, Any, Any]:
    """MaaSSubscription deployed pre-upgrade and referenced for post-upgrade validation.

    Depends on maas_subscription_controller_enabled_latest to ensure MaaS is in
    MANAGED state before the subscription is created or validated.
    """
    subscription_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": MAAS_UPGRADE_SUBSCRIPTION_NAME,
        "namespace": maas_subscription_namespace.name,
    }
    if pytestconfig.option.post_upgrade:
        subscription = MaaSSubscription(**subscription_kwargs)
        yield subscription
        subscription.clean_up()
    else:
        with create_maas_subscription(
            admin_client=admin_client,
            subscription_namespace=maas_subscription_namespace.name,
            subscription_name=MAAS_UPGRADE_SUBSCRIPTION_NAME,
            owner_group_name="system:authenticated",
            model_name=maas_upgrade_model_ref.name,
            model_namespace=maas_upgrade_model_ref.namespace,
            tokens_per_minute=1000,
            window="1m",
            priority=0,
            teardown=teardown_resources,
            wait_for_resource=True,
        ) as subscription:
            yield subscription


@pytest.fixture(scope="session")
def maas_upgrade_tenant(
    admin_client: DynamicClient,
    maas_subscription_namespace: Namespace,
    maas_subscription_controller_enabled_latest: DataScienceCluster,
) -> MaasTenantConfig:
    """Return the default-tenant MaasTenantConfig CR bootstrapped by AITenant / maas-controller.

    Depends on maas_subscription_controller_enabled_latest to ensure MaaS is
    MANAGED and MaasTenantConfig has been reconciled before it is accessed.
    """
    return MaasTenantConfig(
        client=admin_client,
        name="default-tenant",
        namespace=maas_subscription_namespace.name,
        ensure_exists=True,
    )


@pytest.fixture(scope="session")
def maas_upgrade_baseline_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
) -> MaaSBaseline:
    """Load the pre-upgrade MaaS baseline snapshot for post-upgrade assertions.

    Returns an empty dict during pre-upgrade so fixtures that depend on it can be
    unconditionally wired. Only populated during post-upgrade runs.
    """
    if not pytestconfig.option.post_upgrade:
        return {}  # type: ignore[return-value]
    else:
        return load_maas_baseline_from_configmap(
            client=admin_client,
            namespace=MAAS_UPGRADE_NAMESPACE,
        )


@pytest.fixture(scope="session")
def capture_maas_upgrade_baseline(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    maas_upgrade_gateway: Gateway,
    maas_upgrade_model_ref: MaaSModelRef,
    maas_upgrade_auth_policy: MaaSAuthPolicy,
    maas_upgrade_subscription: MaaSSubscription,
    maas_upgrade_tenant: MaasTenantConfig,
) -> None:
    """Capture and persist MaaS state snapshot to ConfigMap before upgrade.

    No-op during post-upgrade runs. During pre-upgrade, saves a baseline of all
    MaaS control plane resources to a ConfigMap in the upgrade namespace so that
    post-upgrade tests can load and compare against actual post-upgrade state.
    """
    if pytestconfig.option.post_upgrade:
        return
    else:
        baseline = capture_maas_baseline(
            gateway=maas_upgrade_gateway,
            model_ref=maas_upgrade_model_ref,
            auth_policy=maas_upgrade_auth_policy,
            subscription=maas_upgrade_subscription,
            tenant=maas_upgrade_tenant,
        )
        save_maas_baseline_to_configmap(
            client=admin_client,
            namespace=MAAS_UPGRADE_NAMESPACE,
            baseline=baseline,
        )
