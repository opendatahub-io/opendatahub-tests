import base64
import os
import re
import shlex
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.data_science_pipelines_application import DataSciencePipelinesApplication
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.resource import Resource, ResourceEditor
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from timeout_sampler import TimeoutExpiredError

from tests.pipelines_components.constants import (
    AUTOML_S3_BUCKET,
    AUTOML_TRAIN_DATA_FILE_KEY,
    DSPA_MINIO_IMAGE,
    DSPA_NAME,
    DSPA_PIPELINE_DEPLOYMENT,
    DSPA_S3_BUCKET,
    DSPA_S3_SECRET,
    EXTERNAL_S3_SECRET,
    MANAGED_PIPELINES_IMAGE,
    MINIO_MC_IMAGE,
    MINIO_UPLOADER_SECURITY_CONTEXT,
)
from utilities.certificates_utils import create_ca_bundle_file
from utilities.general import collect_pod_information
from utilities.infra import create_ns, get_rhods_subscription, wait_for_dsc_status_ready

LOGGER = structlog.get_logger(name=__name__)

_SENSITIVE_PATTERN = re.compile(r"(password|login|apikey|api_key|key|token|secret)", re.IGNORECASE)


def _mask_value(key: str, value: str) -> str:
    if _SENSITIVE_PATTERN.search(key):
        return "****"
    return value


def pytest_configure(config: pytest.Config) -> None:
    """Load .env variables and check if the RHOAI operator is installed."""
    env_file = Path(__file__).parent / ".env"
    if env_file.is_file():
        loaded = {}
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            loaded[key.strip()] = os.environ.get(key.strip(), value.strip())

        LOGGER.info(  # noqa: FCN001
            "Loaded .env file",
            path=str(env_file),
            variables={name: _mask_value(key=name, value=val) for name, val in loaded.items()},
        )

    if config.option.collectonly or getattr(config.option, "setupplan", False):
        return

    try:
        existing = get_rhods_subscription()
    except Exception as exc:
        raise RuntimeError("No cluster connection available — cannot run tests") from exc

    if existing:
        installed_csv = existing.instance.status.get("installedCSV", "unknown")
        LOGGER.info(f"RHOAI operator installed: {existing.name} (CSV: {installed_csv})")
    else:
        raise RuntimeError("RHOAI operator is not installed on this cluster")


@pytest.fixture(scope="session", autouse=True)
def _skip_teardown_if_requested():
    """When SKIP_TEARDOWN=true, prevent all ocp_resources from being deleted on teardown.

    This keeps the namespace, pods, routes, and all resources alive after the test
    so that teams can inspect logs and state.
    Delete the namespace manually when done: oc delete ns <namespace>
    """
    if os.getenv("SKIP_TEARDOWN", "").lower() in ("true", "1", "yes"):
        LOGGER.warning("SKIP_TEARDOWN is set — all resources will be kept after test completion")
        _original_clean_up = Resource.clean_up
        Resource.clean_up = lambda self, *args, **kwargs: True
        yield
        Resource.clean_up = _original_clean_up
    else:
        yield


# ---------------------------------------------------------------------------
# AutoML fixtures — create fresh namespace + DSPA per test run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def pipelines_namespace(  # noqa: UFN001
    admin_client: DynamicClient,
) -> Generator[Namespace, Any, Any]:
    """Dedicated namespace for pipelines component smoke tests."""
    with create_ns(
        admin_client=admin_client,
        name=f"automl-aqa-{uuid.uuid4().hex[:8]}",
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
        wait_for_dsc_status_ready(dsc_resource=dsc_resource)


@pytest.fixture(scope="class")
def dspa(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    enabled_pipelines_in_dsc: DataScienceCluster,
) -> Generator[DataSciencePipelinesApplication, Any, Any]:
    """DataSciencePipelinesApplication with built-in MinIO and managed pipelines."""
    managed_pipelines_spec: dict[str, Any] = {}
    if MANAGED_PIPELINES_IMAGE:
        managed_pipelines_spec["image"] = MANAGED_PIPELINES_IMAGE

    with DataSciencePipelinesApplication(
        client=admin_client,
        name=DSPA_NAME,
        namespace=pipelines_namespace.name,
        dsp_version="v2",
        api_server={
            "enableSamplePipeline": False,
            "managedPipelines": managed_pipelines_spec,
        },
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
        ).wait_for_replicas(timeout=300)

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
def dspa_api_url(dspa_route: Route) -> str:  # noqa: UFN001
    """Base URL for the DSPA v2 REST API."""
    return f"https://{dspa_route.host}"


@pytest.fixture(scope="class")
def dspa_auth_headers(current_client_token: str) -> dict[str, str]:  # noqa: UFN001
    """Authorization headers for DSPA API requests."""
    return {"Authorization": f"Bearer {current_client_token}"}


@pytest.fixture(scope="class")
def dspa_ca_bundle_file(  # noqa: UFN001
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
def external_s3_secret(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
) -> Generator[Secret, Any, Any]:
    """Transient secret holding external AWS S3 credentials for data upload pods."""
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    assert aws_access_key_id, (
        "Environment variable 'AWS_ACCESS_KEY_ID' is not set. "
        "Set it in .env or shell to provide external S3 credentials."
    )
    assert aws_secret_access_key, (
        "Environment variable 'AWS_SECRET_ACCESS_KEY' is not set. "
        "Set it in .env or shell to provide external S3 credentials."
    )
    with Secret(
        client=admin_client,
        name=EXTERNAL_S3_SECRET,
        namespace=pipelines_namespace.name,
        string_data={
            "AWS_ACCESS_KEY_ID": aws_access_key_id,
            "AWS_SECRET_ACCESS_KEY": aws_secret_access_key,
        },
    ) as secret:
        yield secret


@pytest.fixture(scope="function")
def automl_train_data(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    dspa_s3_credentials: Secret,
    external_s3_secret: Secret,
    task_type: str,
) -> str:
    """Download AutoML training CSV from external S3 and upload to DSPA MinIO.

    Picks S3 key based on task_type parameter (regression or classification).
    """
    env_var = f"AUTOML_{task_type.upper()}_S3_TRAIN_DATA_KEY"
    src_key_value = os.environ.get(env_var)
    assert src_key_value, (
        f"Environment variable '{env_var}' is not set. "
        f"Set it in .env or shell to provide the S3 key for {task_type} training data."
    )

    src_bucket = shlex.quote(s=AUTOML_S3_BUCKET)
    src_key = shlex.quote(s=src_key_value)
    dst_bucket = shlex.quote(s=DSPA_S3_BUCKET)
    dst_key = shlex.quote(s=AUTOML_TRAIN_DATA_FILE_KEY)

    minio_endpoint = f"http://minio-{DSPA_NAME}.{pipelines_namespace.name}.svc.cluster.local:9000"
    src_endpoint = os.environ.get("AWS_S3_ENDPOINT", "https://s3.amazonaws.com")

    script = (
        "export MC_CONFIG_DIR=/work/.mc && "
        "mc alias set src $SRC_ENDPOINT $AWS_ACCESS_KEY_ID $AWS_SECRET_ACCESS_KEY && "
        "mc alias set dspa $DST_ENDPOINT $DST_ACCESS_KEY $DST_SECRET_KEY && "
        f"mc cp src/{src_bucket}/{src_key} /work/train.csv && "
        f"mc cp /work/train.csv dspa/{dst_bucket}/{dst_key}"
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
                "args": [script],
                "volumeMounts": [{"name": "work", "mountPath": "/work"}],
                "securityContext": MINIO_UPLOADER_SECURITY_CONTEXT,
                "env": [
                    {"name": "SRC_ENDPOINT", "value": src_endpoint},
                    {
                        "name": "AWS_ACCESS_KEY_ID",
                        "valueFrom": {"secretKeyRef": {"name": EXTERNAL_S3_SECRET, "key": "AWS_ACCESS_KEY_ID"}},
                    },
                    {
                        "name": "AWS_SECRET_ACCESS_KEY",
                        "valueFrom": {"secretKeyRef": {"name": EXTERNAL_S3_SECRET, "key": "AWS_SECRET_ACCESS_KEY"}},
                    },
                    {"name": "DST_ENDPOINT", "value": minio_endpoint},
                    {
                        "name": "DST_ACCESS_KEY",
                        "valueFrom": {"secretKeyRef": {"name": DSPA_S3_SECRET, "key": "accesskey"}},
                    },
                    {
                        "name": "DST_SECRET_KEY",
                        "valueFrom": {"secretKeyRef": {"name": DSPA_S3_SECRET, "key": "secretkey"}},
                    },
                ],
            }
        ],
        wait_for_resource=True,
    ) as upload_pod:
        try:
            upload_pod.wait_for_status(status="Succeeded", timeout=120)
        except TimeoutExpiredError:
            collect_pod_information(pod=upload_pod)
            raise

    return AUTOML_TRAIN_DATA_FILE_KEY
