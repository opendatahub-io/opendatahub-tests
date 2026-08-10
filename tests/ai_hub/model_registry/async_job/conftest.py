import json
import time
from collections.abc import Generator
from typing import Any

import pytest
import shortuuid
import structlog
from kubernetes.dynamic import DynamicClient
from model_registry import ModelRegistry as ModelRegistryClient
from model_registry.types import RegisteredModel
from ocp_resources.job import Job
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from pytest import FixtureRequest

from tests.ai_hub.model_registry.async_job.constants import (
    ASYNC_JOB_ANNOTATIONS,
    ASYNC_JOB_LABELS,
    ASYNC_UPLOAD_JOB_NAME,
    CA_BUNDLE_CONFIG,
    MODEL_SYNC_CONFIG,
    REPO_NAME,
    VOLUME_MOUNTS,
)
from tests.ai_hub.model_registry.async_job.utils import (
    build_async_job_config,
    upload_test_model_to_minio_from_image,
)
from tests.ai_hub.utils import get_endpoint_from_mr_service, get_mr_service_by_label
from utilities.constants import ApiGroups, Labels, MinIo, OCIRegistry, Protocols
from utilities.general import b64_encoded_string, get_s3_secret_dict
from utilities.resources.model_registry_modelregistry_opendatahub_io import ModelRegistry

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="class")
def s3_secret_for_async_job(
    admin_client: DynamicClient,
    service_account: ServiceAccount,
    minio_service: Service,
) -> Generator[Secret, Any, Any]:
    """Create S3 data connection for async upload job"""
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
            aws_default_region="us-east-1",
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
def model_sync_async_job(
    admin_client: DynamicClient,
    sa_token: str,
    service_account: ServiceAccount,
    model_registry_namespace: str,
    model_registry_instance: list[ModelRegistry],
    s3_secret_for_async_job: Secret,
    oci_secret_for_async_job: Secret,
    oci_registry_host: str,
    mr_access_role_binding,
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
    """
    mr_instance = model_registry_instance[0]
    mr_service = get_mr_service_by_label(
        client=admin_client, namespace_name=model_registry_namespace, mr_instance=mr_instance
    )
    mr_address, mr_port = get_endpoint_from_mr_service(svc=mr_service, protocol=Protocols.REST)
    mr_host, _, mr_path = mr_address.partition("/")
    mr_server_address = f"https://{mr_host}:{mr_port}/{mr_path}" if mr_path else f"https://{mr_host}:{mr_port}"

    volume_mounts = [
        {"name": "source-credentials", "readOnly": True, "mountPath": VOLUME_MOUNTS["SOURCE_CREDS_PATH"]},
        {"name": "destination-credentials", "readOnly": True, "mountPath": VOLUME_MOUNTS["DEST_CREDS_PATH"]},
    ]

    environment_variables = [
        {"name": "MODEL_SYNC_SOURCE_TYPE", "value": MODEL_SYNC_CONFIG["SOURCE_TYPE"]},
        {"name": "MODEL_SYNC_SOURCE_AWS_KEY", "value": MODEL_SYNC_CONFIG["SOURCE_AWS_KEY"]},
        {"name": "MODEL_SYNC_SOURCE_S3_CREDENTIALS_PATH", "value": VOLUME_MOUNTS["SOURCE_CREDS_PATH"]},
        {"name": "MODEL_SYNC_MODEL_ID", "value": MODEL_SYNC_CONFIG["MODEL_ID"]},
        {"name": "MODEL_SYNC_MODEL_VERSION_ID", "value": MODEL_SYNC_CONFIG["MODEL_VERSION_ID"]},
        {"name": "MODEL_SYNC_MODEL_ARTIFACT_ID", "value": MODEL_SYNC_CONFIG["MODEL_ARTIFACT_ID"]},
        {"name": "MODEL_SYNC_REGISTRY_SERVER_ADDRESS", "value": mr_server_address},
        {"name": "MODEL_SYNC_REGISTRY_USER_TOKEN", "value": sa_token},
        {"name": "MODEL_SYNC_REGISTRY_IS_SECURE", "value": "False"},
        {
            "name": "MODEL_SYNC_DESTINATION_OCI_REGISTRY",
            "value": f"{oci_registry_host}:{OCIRegistry.Metadata.DEFAULT_PORT}",
        },
        {"name": "MODEL_SYNC_DESTINATION_OCI_URI", "value": f"{oci_registry_host}/{REPO_NAME}"},
        {
            "name": "MODEL_SYNC_DESTINATION_OCI_BASE_IMAGE",
            "value": MODEL_SYNC_CONFIG["DESTINATION_OCI_BASE_IMAGE"],
        },
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
) -> Generator[RegisteredModel]:
    """Create a registered model for testing with KSERVE_MINIO_IMAGE data"""
    yield model_registry_client[0].register_model(
        name=request.param.get("model_name"),
        uri=request.param.get("model_uri"),
        version=request.param.get("model_version"),
        version_description=request.param.get("model_description"),
        model_format_name=request.param.get("model_format"),
        model_format_version=request.param.get("model_format_version"),
        storage_key=request.param.get("model_storage_key"),
        storage_path=request.param.get("model_storage_path"),
    )


@pytest.fixture(scope="function")
def async_job_with_optional_ca_bundle(
    admin_client: DynamicClient,
    sa_token: str,
    service_account: ServiceAccount,
    s3_secret_for_async_job: Secret,
    oci_secret_for_async_job: Secret,
    oci_registry_host: str,
    async_upload_image: str,
    model_registry_instance_rest_endpoint: list[tuple[str, int]],
    teardown_resources: bool,
) -> Generator[Job, Any, Any]:
    """Create async job with optional CA bundle volume mounted."""
    mr_address, mr_port = model_registry_instance_rest_endpoint[0]
    mr_host, _, mr_path = mr_address.partition("/")
    mr_server_address = f"https://{mr_host}:{mr_port}/{mr_path}" if mr_path else f"https://{mr_host}:{mr_port}"

    volume_mounts, environment_variables = build_async_job_config(
        mr_server_address=mr_server_address,
        sa_token=sa_token,
        oci_registry_host=oci_registry_host,
        repo_name=REPO_NAME,
        include_ca_bundle=True,
    )

    with Job(
        client=admin_client,
        name=f"async-job-ca-bundle-{int(time.time())}",
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
            {
                "name": "ca-bundle",
                "configMap": {
                    "name": CA_BUNDLE_CONFIG["CONFIG_MAP_NAME"],
                    "optional": True,
                },
            },
        ],
        teardown=teardown_resources,
    ) as job:
        yield job
