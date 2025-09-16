from typing import Any, Generator
import json

import pytest
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.job import Job

from tests.model_registry.async_job.constants import (
    ASYNC_JOB_ANNOTATIONS,
    ASYNC_JOB_LABELS,
    ASYNC_UPLOAD_JOB_NAME,
    MODEL_SYNC_CONFIG,
    VOLUME_MOUNTS,
)

import shortuuid
from pytest import FixtureRequest
from pytest_testconfig import py_config

from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.role_binding import RoleBinding
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from ocp_resources.model_registry_modelregistry_opendatahub_io import ModelRegistry
from ocp_resources.config_map import ConfigMap
from model_registry.types import RegisteredModel
from model_registry import ModelRegistry as ModelRegistryClient

from utilities.infra import create_ns
from utilities.constants import OCIRegistry, MinIo, Protocols, Labels, ApiGroups
from utilities.general import b64_encoded_string
from tests.model_registry.async_job.utils import upload_test_model_to_minio_from_image
from tests.model_registry.utils import get_mr_service_by_label, get_endpoint_from_mr_service
from tests.model_registry.async_job.constants import REPO_NAME
from utilities.general import get_s3_secret_dict


@pytest.fixture(scope="class")
def s3_secret_for_async_job(
    admin_client: DynamicClient,
    service_account: ServiceAccount,
    minio_service: Service,
) -> Generator[Secret, Any, Any]:
    """Create S3 data connection for async upload job"""
    # Construct MinIO endpoint from service
    minio_endpoint = (
        f"http://{minio_service.name}.{minio_service.namespace}.svc.cluster.local:{MinIo.Metadata.DEFAULT_PORT}"
    )

    with Secret(
        client=admin_client,
        name=f"async-job-s3-connection-{shortuuid.uuid().lower()}",
        namespace=service_account.namespace,
        data_dict=get_s3_secret_dict(
            aws_access_key=MinIo.Credentials.ACCESS_KEY_VALUE,
            aws_secret_access_key=MinIo.Credentials.SECRET_KEY_VALUE,
            aws_s3_bucket=MinIo.Buckets.MODELMESH_EXAMPLE_MODELS,
            aws_s3_endpoint=minio_endpoint,
            aws_default_region="us-east-1",  # Default region for MinIO
        ),
        label={
            Labels.OpenDataHub.DASHBOARD: "true",
            Labels.OpenDataHubIo.MANAGED: "true",
        },
        annotations={
            f"{ApiGroups.OPENDATAHUB_IO}/connection-type": "s3",
            "openshift.io/display-name": "My S3 Credentials",
        },
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def oci_secret_for_async_job(
    admin_client: DynamicClient,
    service_account: ServiceAccount,
    oci_registry_host: str,
) -> Generator[Secret, Any, Any]:
    """Create OCI registry data connection for async upload job"""

    # Create anonymous dockerconfig for OCI registry (no authentication)
    dockerconfig = {
        "auths": {
            f"{oci_registry_host}:{OCIRegistry.Metadata.DEFAULT_PORT}": {
                "auth": "",
                "email": "user@example.com",
            }
        }
    }

    data_dict = {
        ".dockerconfigjson": b64_encoded_string(json.dumps(dockerconfig)),
        "ACCESS_TYPE": b64_encoded_string(json.dumps('["Push,Pull"]')),
        "OCI_HOST": b64_encoded_string(json.dumps(f"{oci_registry_host}:{OCIRegistry.Metadata.DEFAULT_PORT}")),
    }

    with Secret(
        client=admin_client,
        name=f"async-job-oci-connection-{shortuuid.uuid().lower()}",
        namespace=service_account.namespace,
        data_dict=data_dict,
        label={
            Labels.OpenDataHub.DASHBOARD: "true",
            Labels.OpenDataHubIo.MANAGED: "true",
        },
        annotations={
            f"{ApiGroups.OPENDATAHUB_IO}/connection-type-ref": "oci-v1",
            "openshift.io/display-name": "My OCI Credentials",
        },
        type="kubernetes.io/dockerconfigjson",
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def async_upload_image(admin_client: DynamicClient) -> str:
    """
    Get the async upload image dynamically from the model-registry-operator-parameters ConfigMap.

    This fetches the image from the cluster at runtime instead of using a hardcoded value.

    Args:
        admin_client: Kubernetes client for resource access

    Returns:
        str: The async upload image URL from the ConfigMap

    Raises:
        KeyError: If the ConfigMap or the required key doesn't exist
    """
    config_map = ConfigMap(
        client=admin_client,
        name="model-registry-operator-parameters",
        namespace=py_config["applications_namespace"],
    )

    if not config_map.exists:
        raise ResourceNotFoundError(
            f"ConfigMap 'model-registry-operator-parameters' not found in namespace '{py_config['applications_namespace']}'"  # noqa: E501
        )

    try:
        return config_map.instance.data["IMAGES_JOBS_ASYNC_UPLOAD"]
    except KeyError as e:
        raise KeyError(f"Key 'IMAGES_JOBS_ASYNC_UPLOAD' not found in ConfigMap data: {e}") from e


@pytest.fixture(scope="class")
def model_sync_async_job(
    admin_client: DynamicClient,
    sa_token: str,
    service_account: ServiceAccount,
    model_registry_namespace: str,
    model_registry_instance: list[ModelRegistry],
    s3_secret_for_async_job: Secret,
    oci_secret_for_async_job: Secret,
    oci_registry_host: str,
    mr_access_role_binding: RoleBinding,
    async_upload_image: str,
    teardown_resources: bool,
) -> Generator[Job, Any, Any]:
    """
    Job fixture for async model upload with mounted secret files.

    This fixture creates a Kubernetes Job that:
    1. Mounts S3 credentials for source model access
    2. Mounts OCI credentials for destination registry
    3. Configures environment variables for model sync parameters
    4. Waits for job completion before yielding

    Args:
        admin_client: Kubernetes client for resource management
        sa_token: Service account token for Model Registry authentication
        service_account: Service account for the job
        model_registry_namespace: Namespace containing Model Registry
        model_registry_instance: List of Model Registry instances
        s3_secret_for_async_job: Secret containing S3 credentials
        oci_secret_for_async_job: Secret containing OCI registry credentials
        oci_registry_host: OCI registry hostname
        mr_access_role_binding: Role binding for Model Registry access
        async_upload_image: Container image URL for async upload job (fetched dynamically)
        teardown_resources: Whether to clean up resources after test

    Returns:
        Generator yielding the created Job resource
    """

    # Get Model Registry service endpoint for connection
    mr_instance = model_registry_instance[0]
    mr_service = get_mr_service_by_label(
        client=admin_client, namespace_name=model_registry_namespace, mr_instance=mr_instance
    )
    mr_endpoint = get_endpoint_from_mr_service(svc=mr_service, protocol=Protocols.REST)
    mr_host = mr_endpoint.split(":")[0]

    # Volume mounts for credentials
    volume_mounts = [
        {
            "name": "source-credentials",
            "readOnly": True,
            "mountPath": VOLUME_MOUNTS["SOURCE_CREDS_PATH"],
        },
        {
            "name": "destination-credentials",
            "readOnly": True,
            "mountPath": VOLUME_MOUNTS["DEST_CREDS_PATH"],
        },
    ]

    environment_variables = [
        # Source configuration - S3 credentials and model location
        {"name": "MODEL_SYNC_SOURCE_TYPE", "value": MODEL_SYNC_CONFIG["SOURCE_TYPE"]},
        {"name": "MODEL_SYNC_SOURCE_AWS_KEY", "value": MODEL_SYNC_CONFIG["SOURCE_AWS_KEY"]},
        {"name": "MODEL_SYNC_SOURCE_S3_CREDENTIALS_PATH", "value": VOLUME_MOUNTS["SOURCE_CREDS_PATH"]},
        # Model identification parameters
        {"name": "MODEL_SYNC_MODEL_ID", "value": MODEL_SYNC_CONFIG["MODEL_ID"]},
        {"name": "MODEL_SYNC_MODEL_VERSION_ID", "value": MODEL_SYNC_CONFIG["MODEL_VERSION_ID"]},
        {"name": "MODEL_SYNC_MODEL_ARTIFACT_ID", "value": MODEL_SYNC_CONFIG["MODEL_ARTIFACT_ID"]},
        # Model Registry connection parameters
        {"name": "MODEL_SYNC_REGISTRY_SERVER_ADDRESS", "value": f"https://{mr_host}"},
        {"name": "MODEL_SYNC_REGISTRY_USER_TOKEN", "value": sa_token},
        {"name": "MODEL_SYNC_REGISTRY_IS_SECURE", "value": "False"},
        # OCI destination configuration
        {
            "name": "MODEL_SYNC_DESTINATION_OCI_REGISTRY",
            "value": f"{oci_registry_host}:{OCIRegistry.Metadata.DEFAULT_PORT}",
        },
        {"name": "MODEL_SYNC_DESTINATION_OCI_URI", "value": f"{oci_registry_host}/{REPO_NAME}"},
        {"name": "MODEL_SYNC_DESTINATION_OCI_BASE_IMAGE", "value": MODEL_SYNC_CONFIG["DESTINATION_OCI_BASE_IMAGE"]},
        {
            "name": "MODEL_SYNC_DESTINATION_OCI_ENABLE_TLS_VERIFY",
            "value": MODEL_SYNC_CONFIG["DESTINATION_OCI_ENABLE_TLS_VERIFY"],
        },
    ]

    with Job(
        client=admin_client,
        name=ASYNC_UPLOAD_JOB_NAME,
        namespace=service_account.namespace,
        label=ASYNC_JOB_LABELS,
        annotations=ASYNC_JOB_ANNOTATIONS,
        restart_policy="Never",
        containers=[
            {
                "name": "async-upload",
                "image": async_upload_image,
                "volumeMounts": volume_mounts,
                "env": environment_variables,
            }
        ],
        volumes=[
            {"name": "source-credentials", "secret": {"secretName": s3_secret_for_async_job.name}},
            {"name": "destination-credentials", "secret": {"secretName": oci_secret_for_async_job.name}},
        ],
        teardown=teardown_resources,
    ) as job:
        job.wait_for_condition(condition="Complete", status="True")
        yield job


# OCI Registry
@pytest.fixture(scope="class")
def oci_namespace(admin_client: DynamicClient) -> Generator[Namespace, Any, Any]:
    with create_ns(
        name=f"{OCIRegistry.Metadata.NAME}-{shortuuid.uuid().lower()}",
        admin_client=admin_client,
    ) as ns:
        yield ns


@pytest.fixture(scope="class")
def oci_registry_pod_with_minio(
    request: FixtureRequest,
    admin_client: DynamicClient,
    oci_namespace: Namespace,
    minio_service: Service,
) -> Generator[Pod, Any, Any]:
    pod_labels = {Labels.Openshift.APP: OCIRegistry.Metadata.NAME}

    if labels := request.param.get("labels"):
        pod_labels.update(labels)

    minio_fqdn = f"{minio_service.name}.{minio_service.namespace}.svc.cluster.local"
    minio_endpoint = f"{minio_fqdn}:{MinIo.Metadata.DEFAULT_PORT}"

    with Pod(
        client=admin_client,
        name=OCIRegistry.Metadata.NAME,
        namespace=oci_namespace.name,
        containers=[
            {
                "args": request.param.get("args"),
                "env": [
                    {"name": "ZOT_STORAGE_STORAGEDRIVER_NAME", "value": OCIRegistry.Storage.STORAGE_DRIVER},
                    {
                        "name": "ZOT_STORAGE_STORAGEDRIVER_ROOTDIRECTORY",
                        "value": OCIRegistry.Storage.STORAGE_DRIVER_ROOT_DIRECTORY,
                    },
                    {"name": "ZOT_STORAGE_STORAGEDRIVER_BUCKET", "value": MinIo.Buckets.MODELMESH_EXAMPLE_MODELS},
                    {"name": "ZOT_STORAGE_STORAGEDRIVER_REGION", "value": OCIRegistry.Storage.STORAGE_DRIVER_REGION},
                    {"name": "ZOT_STORAGE_STORAGEDRIVER_REGIONENDPOINT", "value": f"http://{minio_endpoint}"},
                    {"name": "ZOT_STORAGE_STORAGEDRIVER_ACCESSKEY", "value": MinIo.Credentials.ACCESS_KEY_VALUE},
                    {"name": "ZOT_STORAGE_STORAGEDRIVER_SECRETKEY", "value": MinIo.Credentials.SECRET_KEY_VALUE},
                    {
                        "name": "ZOT_STORAGE_STORAGEDRIVER_SECURE",
                        "value": OCIRegistry.Storage.STORAGE_STORAGEDRIVER_SECURE,
                    },
                    {
                        "name": "ZOT_STORAGE_STORAGEDRIVER_FORCEPATHSTYLE",
                        "value": OCIRegistry.Storage.STORAGE_STORAGEDRIVER_FORCEPATHSTYLE,
                    },
                    {"name": "ZOT_HTTP_ADDRESS", "value": OCIRegistry.Metadata.DEFAULT_HTTP_ADDRESS},
                    {"name": "ZOT_HTTP_PORT", "value": str(OCIRegistry.Metadata.DEFAULT_PORT)},
                    {"name": "ZOT_LOG_LEVEL", "value": "info"},
                ],
                "image": request.param.get("image", OCIRegistry.PodConfig.REGISTRY_IMAGE),
                "name": OCIRegistry.Metadata.NAME,
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                    "runAsNonRoot": True,
                    "seccompProfile": {"type": "RuntimeDefault"},
                },
                "volumeMounts": [
                    {
                        "name": "zot-data",
                        "mountPath": "/var/lib/registry",
                    }
                ],
            }
        ],
        volumes=[
            {
                "name": "zot-data",
                "emptyDir": {},
            }
        ],
        label=pod_labels,
        annotations=request.param.get("annotations"),
    ) as oci_pod:
        oci_pod.wait_for_condition(condition="Ready", status="True")
        yield oci_pod


@pytest.fixture(scope="class")
def oci_registry_service(admin_client: DynamicClient, oci_namespace: Namespace) -> Generator[Service, Any, Any]:
    with Service(
        client=admin_client,
        name=OCIRegistry.Metadata.NAME,
        namespace=oci_namespace.name,
        ports=[
            {
                "name": f"{OCIRegistry.Metadata.NAME}-port",
                "port": OCIRegistry.Metadata.DEFAULT_PORT,
                "protocol": Protocols.TCP,
                "targetPort": OCIRegistry.Metadata.DEFAULT_PORT,
            }
        ],
        selector={
            Labels.Openshift.APP: OCIRegistry.Metadata.NAME,
        },
        session_affinity="ClientIP",
    ) as oci_service:
        yield oci_service


@pytest.fixture(scope="class")
def oci_registry_route(admin_client: DynamicClient, oci_registry_service: Service) -> Generator[Route, Any, Any]:
    with Route(
        client=admin_client,
        name=OCIRegistry.Metadata.NAME,
        namespace=oci_registry_service.namespace,
        service=oci_registry_service.name,
    ) as oci_route:
        yield oci_route


@pytest.fixture(scope="class")
def oci_registry_host(oci_registry_route: Route) -> str:
    """Get the OCI registry host from the route"""
    return oci_registry_route.instance.spec.host


@pytest.fixture(scope="class")
def create_test_data_in_minio_from_image(
    minio_service: Service,
    admin_client: DynamicClient,
    model_registry_namespace: str,
) -> None:
    """Extract and upload test model from KSERVE_MINIO_IMAGE to MinIO"""
    upload_test_model_to_minio_from_image(
        admin_client=admin_client,
        namespace=model_registry_namespace,
        minio_service=minio_service,
        object_key="my-model/model.onnx",
    )


@pytest.fixture(scope="class")
def registered_model_from_image(
    request: FixtureRequest, model_registry_client: list[ModelRegistryClient]
) -> Generator[RegisteredModel, None, None]:
    """Create a registered model for testing with KSERVE_MINIO_IMAGE data"""
    yield model_registry_client[0].register_model(
        name=request.param.get("model_name"),
        uri=request.param.get("model_uri"),
        version=request.param.get("model_version"),
        description=request.param.get("model_description"),
        model_format_name=request.param.get("model_format"),
        model_format_version=request.param.get("model_format_version"),
        storage_key=request.param.get("model_storage_key"),
        storage_path=request.param.get("model_storage_path"),
    )
