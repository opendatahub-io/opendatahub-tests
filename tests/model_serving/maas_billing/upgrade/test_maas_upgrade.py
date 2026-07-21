import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.deployment import Deployment
from ocp_resources.gateway_gateway_networking_k8s_io import Gateway
from ocp_resources.maas_auth_policy import MaaSAuthPolicy
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.maas_subscription import MaaSSubscription
from ocp_resources.namespace import Namespace
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_serving.maas_billing.upgrade.utils import (
    MaaSBaseline,
    verify_maas_auth_policy_exists,
    verify_maas_model_ref_exists,
    verify_maas_subscription_not_mutated,
    verify_maas_subscription_ready,
)
from tests.model_serving.maas_billing.utils import (
    gateway_probe_reaches_maas_api,
    verify_maas_gateway_programmed,
    verify_maas_tenant_ready,
)
from utilities.constants import ApiGroups
from utilities.general import generate_random_name
from utilities.resources.maas_config import Config as MaaSConfig
from utilities.resources.models_as_service import ModelsAsService
from utilities.resources.tenant import Tenant

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures("capture_maas_upgrade_baseline")
@pytest.mark.pre_upgrade
class TestPreUpgradeMaaS:
    """Deploy and verify realistic MaaS control plane state before upgrade.

    Steps:
        1. Verify MaaS Gateway is Programmed.
        2. Verify default-tenant Tenant CR is Ready and Active.
        3. Verify MaaSModelRef was created successfully.
        4. Verify MaaSAuthPolicy was created successfully.
        5. Verify MaaSSubscription exists.
        6. Verify ModelsAsService CR is absent (pre-upgrade resource).
        7. Verify MaaS Config CR is absent (pre-upgrade resource).
        8. Capture state snapshot to ConfigMap for post-upgrade comparison.
    """

    def test_maas_gateway_programmed(
        self,
        maas_upgrade_gateway: Gateway,
    ) -> None:
        """Verify MaaS gateway is Programmed before upgrade."""
        verify_maas_gateway_programmed(gateway=maas_upgrade_gateway)

    def test_maas_tenant_ready(
        self,
        maas_upgrade_tenant: Tenant,
    ) -> None:
        """Verify default-tenant Tenant CR is Ready before upgrade."""
        verify_maas_tenant_ready(tenant=maas_upgrade_tenant)

    def test_maas_model_ref_created(
        self,
        maas_upgrade_model_ref: MaaSModelRef,
    ) -> None:
        """Verify MaaSModelRef is created before upgrade."""
        verify_maas_model_ref_exists(model_ref=maas_upgrade_model_ref)

    def test_maas_auth_policy_created(
        self,
        maas_upgrade_auth_policy: MaaSAuthPolicy,
    ) -> None:
        """Verify MaaSAuthPolicy is created before upgrade."""
        verify_maas_auth_policy_exists(auth_policy=maas_upgrade_auth_policy)

    def test_maas_subscription_ready(
        self,
        maas_upgrade_subscription: MaaSSubscription,
    ) -> None:
        """Verify MaaSSubscription exists before upgrade."""
        verify_maas_subscription_ready(subscription=maas_upgrade_subscription)

    def test_models_as_service_cr_absent_pre_upgrade(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Given cluster is on pre-upgrade version, when checking for ModelsAsService CR, then it should not exist."""
        models_as_service = ModelsAsService(
            client=admin_client,
            name="default-modelsasservice",
        )
        assert not models_as_service.exists, (
            "ModelsAsService/default-modelsasservice exists — "
            "pre-upgrade tests must not be run on an already-upgraded cluster"
        )

    def test_maas_config_cr_absent_pre_upgrade(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Given cluster is on pre-upgrade version, MaaS Config CRD and CR should not exist."""
        config_crd = CustomResourceDefinition(
            client=admin_client,
            name=f"configs.{ApiGroups.MAAS_IO}",
        )
        if not config_crd.exists:
            return
        maas_config = MaaSConfig(
            client=admin_client,
            name="default",
        )
        assert not maas_config.exists, (
            "MaaS Config/default exists — pre-upgrade tests must not be run on an already-upgraded cluster"
        )


@pytest.mark.post_upgrade
class TestPostUpgradeMaaS:
    """Validate that MaaS control plane state survived the operator upgrade.

    Steps:
        1. Verify default-tenant Tenant CR survived (root gate — controller health indicator).
        2. Verify MaaS Gateway still exists and is Programmed.
        3. Verify MaaSModelRef, MaaSAuthPolicy, MaaSSubscription all survived.
        4. Verify MaaSSubscription was not mutated (generation unchanged).
        5. Verify maas-controller and maas-api Deployments are Available.
        6. Verify MaaS CRDs still exist.
        7. Verify ModelsAsService CR is present (bootstrapped by ODH operator post-upgrade).
        8. Verify MaaS Config CR is present (bootstrapped by maas-controller post-upgrade).
        9. Verify the MaaS API gateway is reachable via probe.
    """

    @pytest.mark.dependency(name="test_default_tenant_survives_upgrade")
    def test_default_tenant_survives_upgrade(
        self,
        maas_upgrade_tenant: Tenant,
    ) -> None:
        """Verify default-tenant survived the operator upgrade."""
        verify_maas_tenant_ready(tenant=maas_upgrade_tenant)

    @pytest.mark.dependency(
        name="test_maas_gateway_survives_upgrade",
        depends=["test_default_tenant_survives_upgrade"],
    )
    def test_maas_gateway_survives_upgrade(
        self,
        maas_upgrade_gateway: Gateway,
    ) -> None:
        """Verify MaaS gateway is still Programmed after upgrade."""
        verify_maas_gateway_programmed(gateway=maas_upgrade_gateway)

    @pytest.mark.dependency(
        name="test_maas_model_ref_survives_upgrade",
        depends=["test_default_tenant_survives_upgrade"],
    )
    def test_maas_model_ref_survives_upgrade(
        self,
        maas_upgrade_model_ref: MaaSModelRef,
    ) -> None:
        """Verify MaaSModelRef survived the operator upgrade."""
        verify_maas_model_ref_exists(model_ref=maas_upgrade_model_ref)

    @pytest.mark.dependency(
        name="test_maas_auth_policy_survives_upgrade",
        depends=["test_maas_model_ref_survives_upgrade"],
    )
    def test_maas_auth_policy_survives_upgrade(
        self,
        maas_upgrade_auth_policy: MaaSAuthPolicy,
    ) -> None:
        """Verify MaaSAuthPolicy survived the operator upgrade."""
        verify_maas_auth_policy_exists(auth_policy=maas_upgrade_auth_policy)

    @pytest.mark.dependency(
        name="test_maas_subscription_survives_upgrade",
        depends=["test_maas_model_ref_survives_upgrade"],
    )
    def test_maas_subscription_survives_upgrade(
        self,
        maas_upgrade_subscription: MaaSSubscription,
    ) -> None:
        """Verify MaaSSubscription survived the operator upgrade."""
        verify_maas_subscription_ready(subscription=maas_upgrade_subscription)

    @pytest.mark.dependency(depends=["test_maas_subscription_survives_upgrade"])
    def test_maas_subscription_not_mutated(
        self,
        maas_upgrade_subscription: MaaSSubscription,
        maas_upgrade_baseline_fixture: MaaSBaseline,
    ) -> None:
        """Verify MaaSSubscription spec was not mutated during the operator upgrade."""
        verify_maas_subscription_not_mutated(
            subscription=maas_upgrade_subscription,
            baseline=maas_upgrade_baseline_fixture,
        )

    @pytest.mark.dependency(depends=["test_default_tenant_survives_upgrade"])
    def test_maas_controller_deployment_available(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Verify maas-controller Deployment is Available after upgrade."""
        controller_deployment = Deployment(
            client=admin_client,
            name="maas-controller",
            namespace=py_config["applications_namespace"],
            ensure_exists=True,
        )
        controller_deployment.wait_for_condition(condition="Available", status="True", timeout=300)

    @pytest.mark.dependency(depends=["test_default_tenant_survives_upgrade"])
    def test_maas_api_deployment_available(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Verify maas-api Deployment is Available after upgrade."""
        api_deployment = Deployment(
            client=admin_client,
            name="maas-api",
            namespace=py_config["applications_namespace"],
            ensure_exists=True,
        )
        api_deployment.wait_for_condition(condition="Available", status="True", timeout=300)

    @pytest.mark.dependency(depends=["test_default_tenant_survives_upgrade"])
    def test_maas_crds_exist(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Verify all expected MaaS CRDs exist after upgrade."""
        expected_crd_names = (
            f"maasmodelrefs.{ApiGroups.MAAS_IO}",
            f"maasauthpolicies.{ApiGroups.MAAS_IO}",
            f"maassubscriptions.{ApiGroups.MAAS_IO}",
            f"tenants.{ApiGroups.MAAS_IO}",
        )
        missing_crds = [
            crd_name
            for crd_name in expected_crd_names
            if not CustomResourceDefinition(client=admin_client, name=crd_name).exists
        ]
        assert not missing_crds, f"MaaS CRDs missing after upgrade: {', '.join(missing_crds)}"

    @pytest.mark.dependency(
        name="test_models_as_service_cr_present_post_upgrade",
        depends=["test_default_tenant_survives_upgrade"],
    )
    def test_models_as_service_cr_present_post_upgrade(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Given upgrade completed, when checking ModelsAsService CR, then it should exist."""
        models_as_service = ModelsAsService(
            client=admin_client,
            name="default-modelsasservice",
        )
        assert models_as_service.exists, (
            "ModelsAsService/default-modelsasservice not found after upgrade — "
            "ODH operator did not bootstrap the component CR."
        )

    @pytest.mark.dependency(depends=["test_models_as_service_cr_present_post_upgrade"])
    def test_maas_config_cr_present_post_upgrade(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Given upgrade completed, when checking MaaS Config CR, then it should exist."""
        maas_config = MaaSConfig(
            client=admin_client,
            name="default",
        )
        assert maas_config.exists, (
            "MaaS Config/default not found after upgrade — "
            "maas-controller did not bootstrap the cluster-scoped Config CR."
        )

    @pytest.mark.dependency(depends=["test_maas_gateway_survives_upgrade"])
    @pytest.mark.usefixtures("authorino_tls_configured")
    def test_maas_gateway_probe_reachable(
        self,
        request_session_http: requests.Session,
        maas_upgrade_base_url: str,
    ) -> None:
        """Verify MaaS gateway URL is reachable after upgrade."""
        probe_url = f"{maas_upgrade_base_url}/v1/models"
        last_status_code = 0
        last_response_text = ""
        try:
            for gateway_reachable, status_code, response_text in TimeoutSampler(
                wait_timeout=300,
                sleep=5,
                func=gateway_probe_reaches_maas_api,
                http_session=request_session_http,
                probe_url=probe_url,
                request_timeout_seconds=30,
            ):
                last_status_code = status_code
                last_response_text = response_text
                if gateway_reachable:
                    return
        except TimeoutExpiredError:
            pytest.fail(
                f"MaaS API gateway not reachable after upgrade at {probe_url}: "
                f"status={last_status_code} body={last_response_text[:200]}"
            )


@pytest.mark.post_upgrade
class TestPostUpgradeMaaSNewResourceCreation:
    """Verify the upgraded MaaS control plane can create new resources (API compatibility).

    Creates a fresh MaaSModelRef after upgrade to validate that the controller
    webhook accepts new resource creation on the upgraded version.
    """

    @pytest.mark.usefixtures("maas_subscription_controller_enabled_latest")
    def test_new_maas_model_ref_can_be_created(
        self,
        admin_client: DynamicClient,
        maas_upgrade_namespace: Namespace,
    ) -> None:
        """Verify a new MaaSModelRef can be created on the upgraded control plane."""
        new_model_ref_name = f"post-upgrade-model-ref-{generate_random_name()}"
        with MaaSModelRef(
            client=admin_client,
            name=new_model_ref_name,
            namespace=maas_upgrade_namespace.name,
            model_ref={
                "name": new_model_ref_name,
                "namespace": maas_upgrade_namespace.name,
                "kind": "LLMInferenceService",
            },
            teardown=True,
            wait_for_resource=True,
        ) as new_model_ref:
            assert new_model_ref.exists, (
                f"Newly created MaaSModelRef '{new_model_ref_name}' does not exist after upgrade."
            )
            LOGGER.info(
                f"Post-upgrade API compatibility confirmed: MaaSModelRef "
                f"'{new_model_ref_name}' created in '{maas_upgrade_namespace.name}'."
            )
