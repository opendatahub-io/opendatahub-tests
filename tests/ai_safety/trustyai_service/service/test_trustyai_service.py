import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.route import Route
from ocp_resources.trustyai_service import TrustyAIService
from timeout_sampler import retry

from tests.ai_safety.trustyai_service.constants import (
    DRIFT_BASE_DATA_PATH,
    TRUSTYAI_DB_MIGRATION_PATCH,
)
from tests.ai_safety.trustyai_service.service.utils import (
    patch_trustyai_service_cr,
    wait_for_trustyai_db_migration_complete_log,
)
from tests.ai_safety.trustyai_service.trustyai_service_utils import (
    TrustyAIServiceMetrics,
    verify_trustyai_service_metric_scheduling_request,
    verify_upload_data_to_trustyai_service,
)
from tests.ai_safety.trustyai_service.utils import (
    validate_trustyai_service_db_conn_failure,
    validate_trustyai_service_images,
)
from utilities.constants import TRUSTYAI_SERVICE_NAME

logger = structlog.get_logger(name=__name__)


@retry(wait_timeout=60, sleep=5)
def _wait_for_route_ready(client: DynamicClient, trustyai_service: TrustyAIService, token: str) -> bool:
    route = Route(client=client, namespace=trustyai_service.namespace, name=TRUSTYAI_SERVICE_NAME, ensure_exists=True)
    url = f"https://{route.instance.spec.host}/q/health"
    response = requests.get(url, headers={"Authorization": f"Bearer {token}"}, verify=False, timeout=10)
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
    """Verify if TrustyAI DB Migration works as expected.
    This test initializes TrustyAI Service with PVC Storage at first with a database on standby but the service is not
    configured to use it.
    Data is uploaded to the PVC, then the TrustyAI CR is patched to trigger a migration from PVC to DB storage.
    config.
    Then waits for the migration success entry in the container logs and patches the service again to remove PVC config.
    Finally, a metric is scheduled and checked if the service works as expected post migration.

    Args:
        admin_client: DynamicClient
        current_client_token: RedactedString
        mariadb: MariaDB
        trustyai_db_ca_secret: None
        trustyai_service: TrustyAIService
        gaussian_credit_model: Generator[InferenceService, Any, Any]

    Returns:
        None
    """
    verify_upload_data_to_trustyai_service(
        client=admin_client,
        trustyai_service=trustyai_service,
        token=current_client_token,
        data_path=f"{DRIFT_BASE_DATA_PATH}/training_data.json",
    )

    trustyai_deployment = Deployment(
        client=admin_client,
        name=TRUSTYAI_SERVICE_NAME,
        namespace=trustyai_service.namespace,
        ensure_exists=True,
    )
    original_deployment_spec = trustyai_deployment.instance.spec.template.spec.to_dict()
    source_volume = next(volume for volume in original_deployment_spec["volumes"] if "persistentVolumeClaim" in volume)
    source_mount = next(
        mount
        for container in original_deployment_spec["containers"]
        if container["name"] == TRUSTYAI_SERVICE_NAME
        for mount in container.get("volumeMounts", [])
        if mount["mountPath"] == "/inputs"
    )

    trustyai_db_migration_patched_service = patch_trustyai_service_cr(
        trustyai_service=trustyai_service, patches=TRUSTYAI_DB_MIGRATION_PATCH
    )

    deployment_spec = trustyai_deployment.instance.spec.template.spec.to_dict()
    deployment_spec["volumes"] = [
        source_volume if volume["name"] == source_volume["name"] else volume for volume in deployment_spec["volumes"]
    ]
    if not any(volume["name"] == "trustyai-service-db-ca" for volume in deployment_spec["volumes"]):
        deployment_spec["volumes"].append({
            "name": "trustyai-service-db-ca",
            "secret": {"secretName": "trustyai-service-db-ca"},  # pragma: allowlist secret
        })
    for container in deployment_spec["containers"]:
        if container["name"] == TRUSTYAI_SERVICE_NAME:
            container["volumeMounts"] = [
                source_mount if mount["name"] == source_mount["name"] else mount
                for mount in container.get("volumeMounts", [])
            ]
            if not any(mount["name"] == "trustyai-service-db-ca" for mount in container["volumeMounts"]):
                container["volumeMounts"].append({
                    "name": "trustyai-service-db-ca",
                    "mountPath": "/etc/tls/db",
                    "readOnly": True,
                })
            container_env = container.setdefault("env", [])
            for environment_variable in container_env:
                if environment_variable.get("name") == "SERVICE_STORAGE_FORMAT":
                    environment_variable["value"] = "DATABASE"
                    environment_variable.pop("valueFrom", None)
                    break
            else:
                container_env.append({"name": "SERVICE_STORAGE_FORMAT", "value": "DATABASE"})
            container_env.append({"name": "DATABASE_ATTEMPT_MIGRATION", "value": "true"})
            break
    else:
        raise AssertionError(f"Container {TRUSTYAI_SERVICE_NAME} not found in TrustyAI deployment")
    ResourceEditor(
        patches={
            trustyai_deployment: {
                "spec": {"template": {"spec": deployment_spec}},
            }
        }
    ).update()

    wait_for_trustyai_db_migration_complete_log(
        client=admin_client,
        trustyai_service=trustyai_db_migration_patched_service,
    )

    trustyai_deployment.wait_for_replicas()

    _wait_for_route_ready(
        client=admin_client,
        trustyai_service=trustyai_db_migration_patched_service,
        token=current_client_token,
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
