from collections.abc import Generator
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.gateway_gateway_networking_k8s_io import Gateway
from ocp_resources.maas_auth_policy import MaaSAuthPolicy
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.maas_subscription import MaaSSubscription
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from pytest import FixtureRequest

from tests.ai_gateway.models_as_a_service.maas_subscription.utils import (
    create_maas_subscription,
)
from tests.ai_gateway.models_as_a_service.upgrade.utils import (
    LEGACY_MIGRATION_AUTH_POLICY_NAME,
    LEGACY_MIGRATION_ENDPOINT,
    LEGACY_MIGRATION_MODEL_NAME,
    LEGACY_MIGRATION_NAMESPACE,
    LEGACY_MIGRATION_SECRET_NAME,
    LEGACY_MIGRATION_SUBSCRIPTION_NAME,
    LEGACY_MIGRATION_TARGET_MODEL,
    MAAS_LEGACY_MIGRATION_BASELINE_CM_NAME,
    LegacyMigrationBaseline,
    MaaSBaseline,
    capture_legacy_migration_baseline,
    capture_maas_baseline,
    cluster_has_inference_external_model_crd,
    cluster_has_legacy_external_model_crd,
    inference_external_model_for_baseline,
    load_legacy_migration_baseline_from_configmap,
    load_maas_baseline_from_configmap,
    save_legacy_migration_baseline_to_configmap,
    save_maas_baseline_to_configmap,
    wait_for_legacy_maas_networking_present,
)
from tests.ai_gateway.models_as_a_service.utils import (
    MaaSTenantResource,
    get_default_maas_tenant,
    host_from_ingress_domain,
)
from utilities.constants import MAAS_GATEWAY_NAME, MAAS_GATEWAY_NAMESPACE
from utilities.infra import create_ns
from utilities.resources.external_model import ExternalModel
from utilities.resources.legacy_external_model import LegacyExternalModel

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
        if teardown_resources and namespace.exists:
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
        model_ref = MaaSModelRef(**model_ref_kwargs, ensure_exists=True)
        yield model_ref
        if teardown_resources and model_ref.exists:
            model_ref.delete(wait=True)
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
        auth_policy = MaaSAuthPolicy(**auth_policy_kwargs, ensure_exists=True)
        yield auth_policy
        if teardown_resources and auth_policy.exists:
            auth_policy.delete(wait=True)
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
        subscription = MaaSSubscription(**subscription_kwargs, ensure_exists=True)
        yield subscription
        if teardown_resources and subscription.exists:
            subscription.delete(wait=True)
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
) -> MaaSTenantResource:
    """Return the default-tenant MaaS CR bootstrapped by maas-controller or AITenant.

    Depends on maas_subscription_controller_enabled_latest to ensure MaaS is
    MANAGED and the tenant CR has been reconciled before it is accessed.
    """
    return get_default_maas_tenant(
        admin_client=admin_client,
        namespace=maas_subscription_namespace.name,
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
    maas_upgrade_tenant: MaaSTenantResource,
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


@pytest.fixture(scope="session")
def legacy_migration_namespace(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    """Dedicated namespace for legacy ExternalModel migration upgrade tests.

    Post-upgrade teardown deletes the namespace only after child fixtures remove their resources.
    """
    namespace = Namespace(client=admin_client, name=LEGACY_MIGRATION_NAMESPACE)
    if pytestconfig.option.post_upgrade:
        yield namespace
        if teardown_resources and namespace.exists:
            namespace.clean_up()
    else:
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
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    legacy_migration_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[Secret, Any, Any]:
    """Opaque secret holding the API key required by the legacy ExternalModel."""
    secret_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": LEGACY_MIGRATION_SECRET_NAME,
        "namespace": legacy_migration_namespace.name,
    }
    if pytestconfig.option.post_upgrade:
        secret = Secret(**secret_kwargs, ensure_exists=True)
        yield secret
        if teardown_resources and secret.exists:
            secret.delete(wait=True)
    else:
        with Secret(
            **secret_kwargs,
            type="Opaque",
            string_data={"api-key": "e2e-test-key"},
            teardown=teardown_resources,
            wait_for_resource=True,
        ) as secret:
            yield secret


@pytest.fixture(scope="session")
def legacy_migration_external_model(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    legacy_migration_namespace: Namespace,
    legacy_migration_credential_secret: Secret,
    teardown_resources: bool,
) -> Generator[LegacyExternalModel, Any, Any]:
    """Legacy maas.opendatahub.io ExternalModel deployed pre-upgrade for migration validation."""
    external_model_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": LEGACY_MIGRATION_MODEL_NAME,
        "namespace": legacy_migration_namespace.name,
    }
    if pytestconfig.option.post_upgrade:
        assert cluster_has_legacy_external_model_crd(admin_client=admin_client), (
            "Legacy maas.opendatahub.io ExternalModel CRD is not installed on this cluster"
        )
        external_model = LegacyExternalModel(**external_model_kwargs)
        yield external_model
        if teardown_resources and external_model.exists:
            external_model.delete(wait=True)
    else:
        with LegacyExternalModel(
            **external_model_kwargs,
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
    pytestconfig: pytest.Config,
    request: FixtureRequest,
    admin_client: DynamicClient,
    legacy_migration_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[MaaSModelRef, Any, Any]:
    """MaaSModelRef linking to the legacy ExternalModel for migration validation."""
    model_ref_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": LEGACY_MIGRATION_MODEL_NAME,
        "namespace": legacy_migration_namespace.name,
    }
    if pytestconfig.option.post_upgrade:
        model_ref = MaaSModelRef(**model_ref_kwargs, ensure_exists=True)
        yield model_ref
        if teardown_resources and model_ref.exists:
            model_ref.delete(wait=True)
    else:
        legacy_migration_external_model = request.getfixturevalue(
            argname="legacy_migration_external_model",
        )
        with MaaSModelRef(
            **model_ref_kwargs,
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
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    legacy_migration_model_ref: MaaSModelRef,
    maas_subscription_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[MaaSAuthPolicy, Any, Any]:
    """MaaSAuthPolicy granting access to the legacy external model migration stack."""
    auth_policy_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": LEGACY_MIGRATION_AUTH_POLICY_NAME,
        "namespace": maas_subscription_namespace.name,
    }
    if pytestconfig.option.post_upgrade:
        auth_policy = MaaSAuthPolicy(**auth_policy_kwargs, ensure_exists=True)
        yield auth_policy
        if teardown_resources and auth_policy.exists:
            auth_policy.delete(wait=True)
    else:
        with MaaSAuthPolicy(
            **auth_policy_kwargs,
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
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    legacy_migration_model_ref: MaaSModelRef,
    maas_subscription_namespace: Namespace,
    maas_subscription_controller_enabled_latest: DataScienceCluster,
    teardown_resources: bool,
) -> Generator[MaaSSubscription, Any, Any]:
    """MaaSSubscription for the legacy external model migration stack."""
    subscription_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": LEGACY_MIGRATION_SUBSCRIPTION_NAME,
        "namespace": maas_subscription_namespace.name,
    }
    if pytestconfig.option.post_upgrade:
        subscription = MaaSSubscription(**subscription_kwargs, ensure_exists=True)
        yield subscription
        if teardown_resources and subscription.exists:
            subscription.delete(wait=True)
    else:
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
def legacy_migration_baseline_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    legacy_migration_namespace: Namespace,
) -> LegacyMigrationBaseline:
    """Load the legacy migration baseline snapshot for post-upgrade assertions."""
    if not pytestconfig.option.post_upgrade:
        return {}  # type: ignore[return-value]
    return load_legacy_migration_baseline_from_configmap(
        client=admin_client,
        namespace=legacy_migration_namespace.name,
    )


@pytest.fixture
def require_inference_external_model_crd(admin_client: DynamicClient) -> None:
    """Assert the inference.opendatahub.io ExternalModel CRD is installed."""
    assert cluster_has_inference_external_model_crd(admin_client=admin_client), (
        "Inference ExternalModel CRD is not installed on this cluster"
    )


@pytest.fixture(scope="session")
def legacy_migration_inference_external_model(
    admin_client: DynamicClient,
    legacy_migration_baseline_fixture: LegacyMigrationBaseline,
) -> ExternalModel:
    """Return the inference ExternalModel referenced by the legacy migration baseline."""
    return inference_external_model_for_baseline(
        client=admin_client,
        baseline=legacy_migration_baseline_fixture,
    )


@pytest.fixture(scope="session")
def capture_legacy_migration_baseline_fixture(
    pytestconfig: pytest.Config,
    request: FixtureRequest,
    admin_client: DynamicClient,
    legacy_migration_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[None, Any, Any]:
    """Capture and persist legacy migration state before upgrade."""
    if pytestconfig.option.post_upgrade:
        yield
        baseline_config_map = ConfigMap(
            client=admin_client,
            name=MAAS_LEGACY_MIGRATION_BASELINE_CM_NAME,
            namespace=legacy_migration_namespace.name,
        )
        if teardown_resources and baseline_config_map.exists:
            baseline_config_map.delete(wait=True)
    else:
        legacy_migration_external_model = request.getfixturevalue(
            argname="legacy_migration_external_model",
        )
        legacy_migration_auth_policy = request.getfixturevalue(
            argname="legacy_migration_auth_policy",
        )
        legacy_migration_subscription = request.getfixturevalue(
            argname="legacy_migration_subscription",
        )
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
        yield
