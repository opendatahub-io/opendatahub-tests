import base64
import shlex
import uuid
from collections.abc import Generator
from typing import Any

import pytest
import shortuuid
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.data_science_pipelines_application import DataSciencePipelinesApplication
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.route import Route
from ocp_resources.secret import Secret

from tests.pipelines_components.constants import (
    AUTOML_SMOKE_CSV,
    AUTOML_TRAIN_DATA_FILE_KEY,
    DSPA_MINIO_IMAGE,
    DSPA_NAME,
    DSPA_PIPELINE_DEPLOYMENT,
    DSPA_S3_BUCKET,
    DSPA_S3_SECRET,
    MINIO_MC_IMAGE,
    MINIO_UPLOADER_SECURITY_CONTEXT,
)
from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import Timeout
from utilities.general import collect_pod_information
from utilities.infra import create_ns, wait_for_dsc_status_ready

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="class")
def pipelines_namespace(
    admin_client: DynamicClient,
) -> Generator[Namespace, Any, Any]:
    """Dedicated namespace for pipelines component smoke tests."""
    with create_ns(
        admin_client=admin_client,
        name=f"pipelines-smoke-{shortuuid.uuid().lower()}",
    ) as namespace:
        yield namespace


@pytest.fixture(scope="class")
def enabled_pipelines_in_dsc(
    dsc_resource: DataScienceCluster,
) -> Generator[DataScienceCluster, Any, Any]:
    """Enable the AI Pipelines component in the DataScienceCluster."""
    with ResourceEditor(
        patches={
            dsc_resource: {
                "spec": {
                    "components": {
                        "aipipelines": {"managementState": "Managed"},
                    }
                }
            }
        }
    ):
        wait_for_dsc_status_ready(dsc_resource=dsc_resource)
        yield dsc_resource


@pytest.fixture(scope="class")
def dspa(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    enabled_pipelines_in_dsc: DataScienceCluster,
) -> Generator[DataSciencePipelinesApplication, Any, Any]:
    """DataSciencePipelinesApplication with built-in MinIO object storage."""
    with DataSciencePipelinesApplication(
        client=admin_client,
        name=DSPA_NAME,
        namespace=pipelines_namespace.name,
        dsp_version="v2",
        object_storage={
            "disableHealthCheck": False,
            "enableExternalRoute": True,
            "minio": {
                "deploy": True,
                "image": DSPA_MINIO_IMAGE,
            },
        },
    ) as dspa_resource:
        Deployment(
            client=admin_client,
            name=DSPA_PIPELINE_DEPLOYMENT,
            namespace=pipelines_namespace.name,
        ).wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)

        yield dspa_resource


@pytest.fixture(scope="class")
def dspa_route(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    dspa: DataSciencePipelinesApplication,
) -> Route:
    """External Route for the DSPA API server."""
    return Route(
        client=admin_client,
        name=DSPA_PIPELINE_DEPLOYMENT,
        namespace=pipelines_namespace.name,
        ensure_exists=True,
    )


@pytest.fixture(scope="class")
def dspa_api_url(dspa_route: Route) -> str:
    """Base URL for the DSPA v2 REST API."""
    return f"https://{dspa_route.host}"


@pytest.fixture(scope="class")
def dspa_auth_headers(current_client_token: str) -> dict[str, str]:
    """Authorization headers for DSPA API requests."""
    return {"Authorization": f"Bearer {current_client_token}"}


@pytest.fixture(scope="class")
def dspa_ca_bundle_file(
    admin_client: DynamicClient,
) -> str:
    """CA bundle file for TLS verification against the DSPA Route."""
    return create_ca_bundle_file(client=admin_client)


@pytest.fixture(scope="class")
def dspa_s3_credentials(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    dspa: DataSciencePipelinesApplication,
) -> Secret:
    """DSPA S3 secret patched with standard AWS credential fields for pipeline components."""
    secret = Secret(
        client=admin_client,
        name=DSPA_S3_SECRET,
        namespace=pipelines_namespace.name,
    )
    assert secret.exists, f"Secret '{DSPA_S3_SECRET}' not found in {pipelines_namespace.name}"

    access_key = base64.b64decode(secret.instance.data.get("accesskey", "")).decode()
    secret_key = base64.b64decode(secret.instance.data.get("secretkey", "")).decode()
    endpoint = f"http://minio-{DSPA_NAME}.{pipelines_namespace.name}.svc.cluster.local:9000"

    secret.update(
        resource_dict={
            "metadata": {"name": secret.name, "namespace": pipelines_namespace.name},
            "stringData": {
                "AWS_ACCESS_KEY_ID": access_key,
                "AWS_SECRET_ACCESS_KEY": secret_key,
                "AWS_S3_ENDPOINT": endpoint,
                "AWS_S3_BUCKET": DSPA_S3_BUCKET,
                "AWS_DEFAULT_REGION": "us-east-1",
            },
        }
    )
    return secret


@pytest.fixture(scope="class")
def automl_train_data(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    dspa_s3_credentials: Secret,
) -> str:
    """Synthetic training CSV uploaded to DSPA MinIO for the AutoML smoke test."""
    access_key = base64.b64decode(dspa_s3_credentials.instance.data.get("accesskey", "")).decode()
    secret_key = base64.b64decode(dspa_s3_credentials.instance.data.get("secretkey", "")).decode()
    minio_endpoint = f"http://minio-{DSPA_NAME}.{pipelines_namespace.name}.svc.cluster.local:9000"

    mc_setup = (
        f"export MC_CONFIG_DIR=/work/.mc && "
        f"mc alias set dspa {shlex.quote(minio_endpoint)} {shlex.quote(access_key)} {shlex.quote(secret_key)}"
    )
    mc_upload = (
        f"cat <<'CSVEOF' > /work/train.csv\n{AUTOML_SMOKE_CSV}CSVEOF\n"
        f"mc cp /work/train.csv dspa/{shlex.quote(DSPA_S3_BUCKET)}/{shlex.quote(AUTOML_TRAIN_DATA_FILE_KEY)}"
    )

    pod_name = f"automl-data-uploader-{uuid.uuid4().hex[:8]}"
    with Pod(
        client=admin_client,
        name=pod_name,
        namespace=pipelines_namespace.name,
        restart_policy="Never",
        volumes=[{"name": "work", "emptyDir": {}}],
        containers=[
            {
                "name": "minio-uploader",
                "image": MINIO_MC_IMAGE,
                "command": ["/bin/sh", "-c"],
                "args": [f"{mc_setup} && {mc_upload}"],
                "volumeMounts": [{"name": "work", "mountPath": "/work"}],
                "securityContext": MINIO_UPLOADER_SECURITY_CONTEXT,
            }
        ],
        wait_for_resource=True,
    ) as upload_pod:
        try:
            upload_pod.wait_for_status(status="Succeeded", timeout=120)
        except Exception:
            collect_pod_information(pod=upload_pod)
            raise

    return AUTOML_TRAIN_DATA_FILE_KEY
