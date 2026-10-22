import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.trustyai_service import TrustyAIService
from timeout_sampler import retry

from tests.ai_safety.trustyai_service.constants import (
    DRIFT_BASE_DATA_PATH,
    TRUSTYAI_DB_MIGRATION_PATCH,
)
from tests.ai_safety.trustyai_service.service.utils import (
    patch_trustyai_service_cr,
    prepare_trustyai_db_migration,
    stage_trustyai_legacy_migration_file,
    wait_for_trustyai_db_migration_rollout,
)
from tests.ai_safety.trustyai_service.trustyai_service_utils import (
    TrustyAIServiceMetrics,
    get_num_observations_from_trustyai_service,
    verify_trustyai_service_metric_scheduling_request,
    verify_upload_data_to_trustyai_service,
)
from tests.ai_safety.trustyai_service.utils import (
    validate_trustyai_service_db_conn_failure,
    validate_trustyai_service_images,
)
from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import TRUSTYAI_SERVICE_NAME

logger = structlog.get_logger(name=__name__)


@retry(wait_timeout=60, sleep=5)
def _wait_for_route_ready(admin_client: DynamicClient, trustyai_service: TrustyAIService, token: str) -> bool:
    route = Route(
        client=admin_client,
        name=TRUSTYAI_SERVICE_NAME,
        namespace=trustyai_service.namespace,
        ensure_exists=True,
    )
    url = f"https://{route.instance.spec.host}/q/health"
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        verify=create_ca_bundle_file(client=admin_client),
        timeout=10,
    )
    content_type = response.headers.get("Content-Type", "")
    logger.info(f"Route readiness check: status={response.status_code}, content-type={content_type}")
    assert "application/json" in content_type, f"Route not ready, got content-type: {content_type}"
    return True


@pytest.mark.smoke
@pytest.mark.ai_safety
def test_trustyaiservice_crd_exists(
    admin_client: DynamicClient,
) -> None:
    """Verify TrustyAIService CRD exists on the cluster."""
    crd_name = "trustyaiservices.trustyai.opendatahub.io"

    crd_resource = CustomResourceDefinition(
        client=admin_client,
        name=crd_name,
        ensure_exists=True,
    )

    assert crd_resource.exists, f"CRD {crd_name} does not exist on the cluster"


@pytest.mark.tier1
@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-trustyai-service-invalid-db-cert"},
        )
    ],
    indirect=True,
)
def test_trustyai_service_with_invalid_db_cert(
    admin_client,
    current_client_token,
    model_namespace: Namespace,
    trustyai_service_with_invalid_db_cert,
):
    """Test to make sure TrustyAIService pod fails when incorrect database TLS certificate is used."""
    validate_trustyai_service_db_conn_failure(
        client=admin_client,
        namespace=model_namespace,
        label_selector=f"app.kubernetes.io/instance={trustyai_service_with_invalid_db_cert.name}",
        trustyai_service=trustyai_service_with_invalid_db_cert,
    )


@pytest.mark.tier1
@pytest.mark.parametrize(
    "model_namespace, trustyai_service",
    [
        pytest.param(
            {"name": "test-validate-trustyai-service-images"},
            {"storage": "pvc"},
        )
    ],
    indirect=True,
)
def test_validate_trustyai_service_image(
    admin_client,
    model_namespace: Namespace,
    related_images_refs: set[str],
    trustyai_service: TrustyAIService,
    trustyai_operator_configmap,
):
    return validate_trustyai_service_images(
        client=admin_client,
        related_images_refs=related_images_refs,
        model_namespace=model_namespace,
        label_selector=f"app.kubernetes.io/instance={trustyai_service.name}",
        trustyai_operator_configmap=trustyai_operator_configmap,
    )


@pytest.mark.tier1
@pytest.mark.parametrize(
    "model_namespace, trustyai_service",
    [
        pytest.param(
            {"name": "test-trustyai-db-migration"},
            {"storage": "pvc"},
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
def test_trustyai_service_db_migration(
    admin_client,
    current_client_token,
    mariadb,
    trustyai_db_ca_secret,
    trustyai_service,
    gaussian_credit_model,
) -> None:
    """Given PVC-backed observations and MariaDB, when storage changes to DATABASE, then data and metrics survive."""
    verify_upload_data_to_trustyai_service(
        client=admin_client,
        trustyai_service=trustyai_service,
        token=current_client_token,
        data_path=f"{DRIFT_BASE_DATA_PATH}/training_data.json",
    )
    expected_observations = get_num_observations_from_trustyai_service(
        client=admin_client,
        token=current_client_token,
        trustyai_service=trustyai_service,
        model_name="gaussian-credit-model",
    )
    assert expected_observations > 0, "Training observations were not present before database migration"

    stage_trustyai_legacy_migration_file(
        client=admin_client,
        namespace=trustyai_service.namespace,
        service_name=trustyai_service.name,
        model_name="gaussian-credit-model",
    )

    trustyai_deployment = Deployment(
        client=admin_client,
        name=TRUSTYAI_SERVICE_NAME,
        namespace=trustyai_service.namespace,
        ensure_exists=True,
    )
    original_deployment_spec = trustyai_deployment.instance.spec.template.spec.to_dict()

    trustyai_db_migration_patched_service = patch_trustyai_service_cr(
        trustyai_service=trustyai_service, patches=TRUSTYAI_DB_MIGRATION_PATCH
    )

    prepare_trustyai_db_migration(
        client=admin_client,
        namespace=trustyai_service.namespace,
        service_name=trustyai_service.name,
        original_deployment_spec=original_deployment_spec,
    )
    wait_for_trustyai_db_migration_rollout(
        client=admin_client,
        namespace=trustyai_service.namespace,
        service_name=trustyai_service.name,
    )

    _wait_for_route_ready(
        admin_client=admin_client,
        trustyai_service=trustyai_db_migration_patched_service,
        token=current_client_token,
    )

    migrated_observations = get_num_observations_from_trustyai_service(
        client=admin_client,
        token=current_client_token,
        trustyai_service=trustyai_db_migration_patched_service,
        model_name="gaussian-credit-model",
    )
    assert migrated_observations == expected_observations, (
        "Database migration changed the gaussian-credit-model observation count: "
        f"expected {expected_observations}, got {migrated_observations}"
    )

    verify_trustyai_service_metric_scheduling_request(
        client=admin_client,
        trustyai_service=trustyai_db_migration_patched_service,
        token=current_client_token,
        metric_name=TrustyAIServiceMetrics.Drift.MEANSHIFT,
        json_data={
            "modelId": gaussian_credit_model.name,
            "referenceTag": "TRAINING",
        },
    )
