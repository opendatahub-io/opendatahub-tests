from typing import Any, Generator
import json

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job

from tests.model_registry.async_job.constants import (
    ASYNC_JOB_ANNOTATIONS,
    ASYNC_JOB_LABELS,
    ASYNC_UPLOAD_IMAGE,
    ASYNC_UPLOAD_JOB_NAME,
    MODEL_SYNC_CONFIG,
    VOLUME_MOUNTS,
)

import shortuuid
from pytest import FixtureRequest

from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.model_registry_modelregistry_opendatahub_io import ModelRegistry

from utilities.infra import create_ns
from utilities.constants import OCIRegistry, MinIo, Protocols, Labels
from utilities.general import b64_encoded_string
from tests.model_registry.async_job.utils import get_async_job_s3_secret_dict, upload_test_model_to_minio
from tests.model_registry.utils import get_mr_service_by_label, get_endpoint_from_mr_service
from tests.model_registry.async_job.constants import REPO_NAME


# We need to upstream this to the wrapper library
class JobWithVolumes(Job):
    """Extended Job class that supports volumes"""

    def __init__(self, volumes=None, **kwargs):
        super().__init__(**kwargs)
        self.volumes = volumes or []

    def to_dict(self) -> None:
        super().to_dict()
        if not self.kind_dict and not self.yaml_file and self.volumes:
            self.res["spec"].setdefault("template", {}).setdefault("spec", {})
            self.res["spec"]["template"]["spec"]["volumes"] = self.volumes


@pytest.fixture(scope="function")
def s3_secret_for_async_job(
    admin_client: DynamicClient,
    service_account,
    minio_service: Service,
) -> Generator[Secret, Any, Any]:
    """Create S3 credentials secret for async upload job"""
    # Construct MinIO endpoint from service
    minio_endpoint = (
        f"http://{minio_service.name}.{minio_service.namespace}.svc.cluster.local:{MinIo.Metadata.DEFAULT_PORT}"
    )

    # Create S3 secret data using existing utility
    secret_data = get_async_job_s3_secret_dict(
        access_key=MinIo.Credentials.ACCESS_KEY_VALUE,
        secret_access_key=MinIo.Credentials.SECRET_KEY_VALUE,
        s3_bucket=MinIo.Buckets.MODELMESH_EXAMPLE_MODELS,
        s3_endpoint=minio_endpoint,
        s3_region="us-east-1",  # Default region for MinIO
    )

    secret_name = f"async-job-s3-secret-{shortuuid.uuid().lower()}"

    with Secret(
        client=admin_client,
        name=secret_name,
        namespace=service_account.namespace,
        data_dict=secret_data,
        type="Opaque",
    ) as secret:
        yield secret


@pytest.fixture(scope="function")
def oci_secret_for_async_job(
    admin_client: DynamicClient,
    service_account,
    oci_registry_route: Route,
) -> Generator[Secret, Any, Any]:
    """Create OCI registry credentials secret for async upload job"""
    # Get OCI registry endpoint from route
    registry_host = oci_registry_route.instance.spec.host

    # Create anonymous dockerconfig for OCI registry (no authentication)
    # This matches the zot registry setup which allows anonymous access
    dockerconfig = {
        "auths": {
            f"{registry_host}:{OCIRegistry.Metadata.DEFAULT_PORT}": {
                "auth": "",
                "email": "user@example.com",  # Anonymous access
            }
        }
    }

    secret_data = {
        ".dockerconfigjson": b64_encoded_string(json.dumps(dockerconfig)),
        "ACCESS_TYPE": b64_encoded_string(json.dumps('["Push,Pull"]')),
        "OCI_HOST": b64_encoded_string(json.dumps(f"{registry_host}:{OCIRegistry.Metadata.DEFAULT_PORT}")),
    }

    secret_name = f"async-job-oci-secret-{shortuuid.uuid().lower()}"

    with Secret(
        client=admin_client,
        name=secret_name,
        namespace=service_account.namespace,
        data_dict=secret_data,
        type="kubernetes.io/dockerconfigjson",
    ) as secret:
        yield secret


@pytest.fixture(scope="function")
def model_sync_async_job(
    admin_client: DynamicClient,
    sa_token: str,
    service_account,
    model_registry_namespace: str,
    model_registry_instance_mysql: list[ModelRegistry],
    s3_secret_for_async_job: Secret,
    oci_secret_for_async_job: Secret,
    oci_registry_route: Route,
    mr_access_role_binding,
    teardown_resources: bool,
) -> Generator[Job, Any, Any]:
    """Core Job fixture focused on Job deployment and configuration"""
    # Get dynamic OCI URI from route
    oci_registry_host = oci_registry_route.instance.spec.host
    dynamic_oci_uri = f"{oci_registry_host}/{REPO_NAME}"

    # Get model registry service and endpoint
    mr_instance = model_registry_instance_mysql[0]  # Use first instance
    mr_service = get_mr_service_by_label(
        client=admin_client, namespace_name=model_registry_namespace, mr_instance=mr_instance
    )
    mr_endpoint = get_endpoint_from_mr_service(svc=mr_service, protocol=Protocols.REST)
    mr_host = mr_endpoint.split(":")[0]
    mr_port = mr_endpoint.split(":")[1]

    with JobWithVolumes(
        client=admin_client,
        name=ASYNC_UPLOAD_JOB_NAME,
        namespace=service_account.namespace,
        label=ASYNC_JOB_LABELS,
        annotations=ASYNC_JOB_ANNOTATIONS,
        restart_policy="Never",
        containers=[
            {
                "name": "async-upload",
                "image": ASYNC_UPLOAD_IMAGE,
                "volumeMounts": [
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
                ],
                "env": [
                    # Proxy settings
                    {"name": "HTTP_PROXY", "value": ""},
                    {"name": "HTTPS_PROXY", "value": ""},
                    {"name": "NO_PROXY", "value": "*.svc.cluster.local"},
                    # Source configuration
                    {"name": "MODEL_SYNC_SOURCE_TYPE", "value": MODEL_SYNC_CONFIG["SOURCE_TYPE"]},
                    {"name": "MODEL_SYNC_SOURCE_AWS_KEY", "value": MODEL_SYNC_CONFIG["SOURCE_AWS_KEY"]},
                    {
                        "name": "MODEL_SYNC_SOURCE_S3_CREDENTIALS_PATH",
                        "value": VOLUME_MOUNTS["SOURCE_CREDS_PATH"],
                    },
                    # Destination configuration
                    {"name": "MODEL_SYNC_DESTINATION_TYPE", "value": MODEL_SYNC_CONFIG["DESTINATION_TYPE"]},
                    {
                        "name": "MODEL_SYNC_DESTINATION_OCI_URI",
                        "value": f"{dynamic_oci_uri}",
                    },
                    {
                        "name": "MODEL_SYNC_DESTINATION_OCI_REGISTRY",
                        "value": f"{oci_registry_host}:{OCIRegistry.Metadata.DEFAULT_PORT}",
                    },
                    {
                        "name": "MODEL_SYNC_DESTINATION_OCI_CREDENTIALS_PATH",
                        "value": VOLUME_MOUNTS["DEST_DOCKERCONFIG_PATH"],
                    },
                    {
                        "name": "MODEL_SYNC_DESTINATION_OCI_BASE_IMAGE",
                        "value": MODEL_SYNC_CONFIG["DESTINATION_OCI_BASE_IMAGE"],
                    },
                    {
                        "name": "MODEL_SYNC_DESTINATION_OCI_ENABLE_TLS_VERIFY",
                        "value": MODEL_SYNC_CONFIG["DESTINATION_OCI_ENABLE_TLS_VERIFY"],
                    },
                    # Model parameters
                    {"name": "MODEL_SYNC_MODEL_ID", "value": MODEL_SYNC_CONFIG["MODEL_ID"]},
                    {"name": "MODEL_SYNC_MODEL_VERSION_ID", "value": MODEL_SYNC_CONFIG["MODEL_VERSION_ID"]},
                    {
                        "name": "MODEL_SYNC_MODEL_ARTIFACT_ID",
                        "value": MODEL_SYNC_CONFIG["MODEL_ARTIFACT_ID"],
                    },
                    # Model Registry client params
                    {
                        "name": "MODEL_SYNC_REGISTRY_SERVER_ADDRESS",
                        "value": f"https://{mr_host}",
                    },
                    {"name": "MODEL_SYNC_REGISTRY_PORT", "value": mr_port},
                    {"name": "MODEL_SYNC_REGISTRY_AUTHOR", "value": "RHOAI async job test"},
                    {"name": "MODEL_SYNC_REGISTRY_USER_TOKEN", "value": sa_token},
                    {"name": "MODEL_SYNC_REGISTRY_IS_SECURE", "value": "False"},
                ],
            }
        ],
        volumes=[
            {
                "name": "source-credentials",
                "secret": {
                    "secretName": s3_secret_for_async_job.name,
                },
            },
            {
                "name": "destination-credentials",
                "secret": {
                    "secretName": oci_secret_for_async_job.name,
                },
            },
        ],
        teardown=teardown_resources,
    ) as job:
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


@pytest.fixture(scope="function")
def create_test_data_in_minio(
    minio_service: Service,
    admin_client: DynamicClient,
    model_registry_namespace: str,
) -> None:
    """Upload the mnist-8.onnx test model file to MinIO"""
    upload_test_model_to_minio(
        admin_client=admin_client, namespace=model_registry_namespace, minio_service=minio_service
    )
