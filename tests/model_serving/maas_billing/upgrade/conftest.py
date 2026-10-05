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
from ocp_resources.secret import Secret

from tests.model_serving.maas_billing.maas_subscription.utils import (
    create_maas_subscription,
)
from tests.model_serving.maas_billing.upgrade.utils import (
    LEGACY_MIGRATION_AUTH_POLICY_NAME,
    LEGACY_MIGRATION_ENDPOINT,
    LEGACY_MIGRATION_MODEL_NAME,
    LEGACY_MIGRATION_NAMESPACE,
    LEGACY_MIGRATION_SECRET_NAME,
    LEGACY_MIGRATION_SUBSCRIPTION_NAME,
    LEGACY_MIGRATION_TARGET_MODEL,
    capture_legacy_migration_baseline,
    capture_maas_baseline,
    cluster_has_legacy_external_model_crd,
    save_legacy_migration_baseline_to_configmap,
    save_maas_baseline_to_configmap,
    wait_for_legacy_maas_networking_present,
)
from tests.model_serving.maas_billing.utils import (
    MaaSTenantResource,
    get_default_maas_tenant,
)
from utilities.constants import MAAS_GATEWAY_NAME, MAAS_GATEWAY_NAMESPACE
from utilities.infra import create_ns
from utilities.resources.legacy_external_model import LegacyExternalModel

LOGGER = structlog.get_logger(name=__name__)

MAAS_UPGRADE_NAMESPACE = "upgrade-maas"
MAAS_UPGRADE_MODEL_NAME = "upgrade-maas-model-ref"
MAAS_UPGRADE_AUTH_POLICY_NAME = "upgrade-maas-auth-policy"
MAAS_UPGRADE_SUBSCRIPTION_NAME = "upgrade-maas-subscription"


@pytest.fixture(scope="session")
def maas_upgrade_namespace(
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    """Dedicated namespace for MaaS upgrade test resources."""
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
def maas_upgrade_model_ref(
    admin_client: DynamicClient,
    maas_upgrade_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[MaaSModelRef, Any, Any]:
    """MaaSModelRef deployed pre-upgrade for post-upgrade validation on 3.5."""
    model_ref_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": MAAS_UPGRADE_MODEL_NAME,
        "namespace": maas_upgrade_namespace.name,
    }
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
    admin_client: DynamicClient,
    maas_upgrade_model_ref: MaaSModelRef,
    maas_subscription_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[MaaSAuthPolicy, Any, Any]:
    """MaaSAuthPolicy deployed pre-upgrade for post-upgrade validation on 3.5."""
    auth_policy_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": MAAS_UPGRADE_AUTH_POLICY_NAME,
        "namespace": maas_subscription_namespace.name,
    }
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
    admin_client: DynamicClient,
    maas_upgrade_model_ref: MaaSModelRef,
    maas_subscription_namespace: Namespace,
    maas_subscription_controller_enabled_latest: DataScienceCluster,
    teardown_resources: bool,
) -> Generator[MaaSSubscription, Any, Any]:
    """MaaSSubscription deployed pre-upgrade for post-upgrade validation on 3.5.

    Depends on maas_subscription_controller_enabled_latest to ensure MaaS is in
    MANAGED state before the subscription is created.
    """
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
) -> MaaSTenantResource:
    """Return the default-tenant legacy Tenant CR bootstrapped by maas-controller.

    Depends on maas_subscription_controller_enabled_latest to ensure MaaS is
    MANAGED and the tenant CR has been reconciled before it is accessed.
    """
    return get_default_maas_tenant(
        admin_client=admin_client,
        namespace=maas_subscription_namespace.name,
    )


@pytest.fixture(scope="session")
def capture_maas_upgrade_baseline(
    admin_client: DynamicClient,
    maas_upgrade_gateway: Gateway,
    maas_upgrade_model_ref: MaaSModelRef,
    maas_upgrade_auth_policy: MaaSAuthPolicy,
    maas_upgrade_subscription: MaaSSubscription,
    maas_upgrade_tenant: MaaSTenantResource,
) -> None:
    """Capture and persist MaaS state snapshot to ConfigMap before upgrade.

    Saves a baseline of all MaaS control plane resources to a ConfigMap in the
    upgrade namespace so that post-upgrade tests on 3.5 can load and compare
    against actual post-upgrade state.
    """
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


@pytest.fixture(scope="session")
def legacy_migration_namespace(
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    """Dedicated namespace for legacy ExternalModel migration pre-upgrade tests."""
    assert cluster_has_legacy_external_model_crd(admin_client=admin_client), (
        "Legacy maas.opendatahub.io ExternalModel CRD is not installed on this cluster"
    )
    with create_ns(
        admin_client=admin_client,
        name=LEGACY_MIGRATION_NAMESPACE,
        model_mesh_enabled=False,
        add_dashboard_label=True,
        teardown=teardown_resources,
    ) as namespace:
        yield namespace


@pytest.fixture(scope="session")
def legacy_migration_credential_secret(
    admin_client: DynamicClient,
    legacy_migration_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[Secret, Any, Any]:
    """Opaque secret holding the API key required by the legacy ExternalModel."""
    with Secret(
        client=admin_client,
        name=LEGACY_MIGRATION_SECRET_NAME,
        namespace=legacy_migration_namespace.name,
        type="Opaque",
        string_data={"api-key": "e2e-test-key"},
        teardown=teardown_resources,
        wait_for_resource=True,
    ) as secret:
        yield secret


@pytest.fixture(scope="session")
def legacy_migration_external_model(
    admin_client: DynamicClient,
    legacy_migration_namespace: Namespace,
    legacy_migration_credential_secret: Secret,
    teardown_resources: bool,
) -> Generator[LegacyExternalModel, Any, Any]:
    """Legacy maas.opendatahub.io ExternalModel deployed pre-upgrade for migration validation."""
    with LegacyExternalModel(
        client=admin_client,
        name=LEGACY_MIGRATION_MODEL_NAME,
        namespace=legacy_migration_namespace.name,
        provider="openai",
        target_model=LEGACY_MIGRATION_TARGET_MODEL,
        endpoint=LEGACY_MIGRATION_ENDPOINT,
        credential_ref={"name": legacy_migration_credential_secret.name},
        teardown=teardown_resources,
        wait_for_resource=True,
    ) as external_model:
        wait_for_legacy_maas_networking_present(
            client=admin_client,
            model_name=external_model.name,
            namespace=legacy_migration_namespace.name,
        )
        yield external_model


@pytest.fixture(scope="session")
def legacy_migration_model_ref(
    admin_client: DynamicClient,
    legacy_migration_namespace: Namespace,
    legacy_migration_external_model: LegacyExternalModel,
    teardown_resources: bool,
) -> Generator[MaaSModelRef, Any, Any]:
    """MaaSModelRef linking to the legacy ExternalModel for migration validation."""
    with MaaSModelRef(
        client=admin_client,
        name=LEGACY_MIGRATION_MODEL_NAME,
        namespace=legacy_migration_namespace.name,
        model_ref={
            "name": legacy_migration_external_model.name,
            "namespace": legacy_migration_external_model.namespace,
            "kind": "ExternalModel",
        },
        teardown=teardown_resources,
        wait_for_resource=True,
    ) as model_ref:
        yield model_ref


@pytest.fixture(scope="session")
def legacy_migration_auth_policy(
    admin_client: DynamicClient,
    legacy_migration_model_ref: MaaSModelRef,
    maas_subscription_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[MaaSAuthPolicy, Any, Any]:
    """MaaSAuthPolicy granting access to the legacy external model migration stack."""
    with MaaSAuthPolicy(
        client=admin_client,
        name=LEGACY_MIGRATION_AUTH_POLICY_NAME,
        namespace=maas_subscription_namespace.name,
        model_refs=[
            {
                "name": legacy_migration_model_ref.name,
                "namespace": legacy_migration_model_ref.namespace,
            }
        ],
        subjects={"groups": [{"name": "system:authenticated"}]},
        teardown=teardown_resources,
        wait_for_resource=True,
    ) as auth_policy:
        yield auth_policy


@pytest.fixture(scope="session")
def legacy_migration_subscription(
    admin_client: DynamicClient,
    legacy_migration_model_ref: MaaSModelRef,
    maas_subscription_namespace: Namespace,
    maas_subscription_controller_enabled_latest: DataScienceCluster,
    teardown_resources: bool,
) -> Generator[MaaSSubscription, Any, Any]:
    """MaaSSubscription for the legacy external model migration stack."""
    with create_maas_subscription(
        admin_client=admin_client,
        subscription_namespace=maas_subscription_namespace.name,
        subscription_name=LEGACY_MIGRATION_SUBSCRIPTION_NAME,
        owner_group_name="system:authenticated",
        model_name=legacy_migration_model_ref.name,
        model_namespace=legacy_migration_model_ref.namespace,
        tokens_per_minute=1000,
        window="1m",
        priority=0,
        teardown=teardown_resources,
        wait_for_resource=True,
    ) as subscription:
        yield subscription


@pytest.fixture(scope="session")
def capture_legacy_migration_baseline_fixture(
    admin_client: DynamicClient,
    legacy_migration_namespace: Namespace,
    legacy_migration_external_model: LegacyExternalModel,
    legacy_migration_auth_policy: MaaSAuthPolicy,
    legacy_migration_subscription: MaaSSubscription,
) -> None:
    """Capture and persist legacy migration state before upgrade."""
    baseline = capture_legacy_migration_baseline(
        client=admin_client,
        model_name=legacy_migration_external_model.name,
        model_namespace=legacy_migration_namespace.name,
        auth_policy=legacy_migration_auth_policy,
        subscription=legacy_migration_subscription,
    )
    save_legacy_migration_baseline_to_configmap(
        client=admin_client,
        namespace=legacy_migration_namespace.name,
        baseline=baseline,
    )
