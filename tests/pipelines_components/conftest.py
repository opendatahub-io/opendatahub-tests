import base64
import os
import re
import shlex
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import shortuuid
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.data_science_pipelines_application import DataSciencePipelinesApplication
from ocp_resources.deployment import Deployment
from ocp_resources.dsc_initialization import DSCInitialization
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor, get_client
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_utilities.operators import install_operator, uninstall_operator
from timeout_sampler import TimeoutExpiredError

from tests.pipelines_components.constants import (
    AUTOML_SMOKE_CSV,
    AUTOML_TRAIN_DATA_FILE_KEY,
    DSPA_MINIO_IMAGE,
    DSPA_NAME,
    DSPA_PIPELINE_DEPLOYMENT,
    DSPA_S3_BUCKET,
    DSPA_S3_SECRET,
    MANAGED_PIPELINES_IMAGE,
    MINIO_MC_IMAGE,
    MINIO_UPLOADER_SECURITY_CONTEXT,
)
from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import RHOAI_OPERATOR_NAMESPACE, Timeout
from utilities.general import collect_pod_information
from utilities.infra import create_ns, get_rhods_subscription, wait_for_dsc_status_ready, wait_for_dsci_status_ready

LOGGER = structlog.get_logger(name=__name__)

_SENSITIVE_PATTERN = re.compile(r"(password|login|apikey|api_key|key|token|secret)", re.IGNORECASE)


def _mask_value(key: str, value: str) -> str:
    if _SENSITIVE_PATTERN.search(key):
        return "****"
    return value


_RHOAI_INSTALLED_BY_TESTS = False


def pytest_configure(config: pytest.Config) -> None:
    """Load .env variables and ensure the RHOAI operator is installed.

    This runs before the root conftest's pytest_sessionstart, which tries to
    access the DataScienceCluster CRD.  If the operator is missing, that hook
    crashes with NotImplementedError before any fixture gets a chance to run.
    """
    global _RHOAI_INSTALLED_BY_TESTS

    env_file = Path(__file__).parent / ".env"
    if env_file.is_file():
        loaded = {}
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            loaded[key.strip()] = os.environ.get(key.strip(), value.strip())

        logger = structlog.get_logger(name=__name__)
        logger.info(  # noqa: FCN001
            "Loaded .env file",
            path=str(env_file),
            variables={k: _mask_value(key=k, value=v) for k, v in loaded.items()},
        )

    existing = get_rhods_subscription()
    if existing:
        LOGGER.info(f"RHOAI operator already installed: {existing.name}")
        return

    LOGGER.info("RHOAI operator not found — installing from redhat-operators/stable")
    client = get_client()
    install_operator(
        admin_client=client,
        target_namespaces=None,
        name="rhods-operator",
        channel="stable",
        source="redhat-operators",
        operator_namespace=RHOAI_OPERATOR_NAMESPACE,
        timeout=Timeout.TIMEOUT_15MIN,
    )
    wait_for_dsci_status_ready(dsci_resource=DSCInitialization(client=client, name="default-dsci", ensure_exists=True))
    wait_for_dsc_status_ready(dsc_resource=DataScienceCluster(client=client, name="default-dsc", ensure_exists=True))
    LOGGER.info("RHOAI operator installed and ready")
    _RHOAI_INSTALLED_BY_TESTS = True


@pytest.fixture(scope="session", autouse=True)
def _uninstall_rhoai_if_we_installed_it(admin_client: DynamicClient) -> Generator[None, Any, Any]:
    """Uninstall the RHOAI operator on teardown if it was installed by pytest_configure."""
    yield
    if not _RHOAI_INSTALLED_BY_TESTS:
        return
    LOGGER.info("Uninstalling RHOAI operator (installed by test fixture)")
    uninstall_operator(
        admin_client=admin_client,
        name="rhods-operator",
        operator_namespace=RHOAI_OPERATOR_NAMESPACE,
        clean_up_namespace=False,
    )


@pytest.fixture(scope="session", autouse=True)
def _skip_teardown_if_requested():
    """When SKIP_TEARDOWN=true, prevent all ocp_resources from being deleted on teardown.

    This keeps the namespace, pods, routes, and all resources alive after the test
    so that teams can inspect logs and state.
    Delete the namespace manually when done: oc delete ns <namespace>
    """
    if os.getenv("SKIP_TEARDOWN", "").lower() not in ("true", "1", "yes"):
        yield
        return
    LOGGER.warning("SKIP_TEARDOWN is set — all resources will be kept after test completion")
    from ocp_resources.resource import Resource

    _original_clean_up = Resource.clean_up
    Resource.clean_up = lambda self, *args, **kwargs: True
    yield
    Resource.clean_up = _original_clean_up


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
def automl_train_data(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    dspa_s3_credentials: Secret,
) -> str:
    """Synthetic training CSV uploaded to DSPA MinIO for the AutoML smoke test."""
    minio_endpoint = f"http://minio-{DSPA_NAME}.{pipelines_namespace.name}.svc.cluster.local:9000"

    mc_setup = "export MC_CONFIG_DIR=/work/.mc && mc alias set dspa $MINIO_ENDPOINT $MINIO_ACCESS_KEY $MINIO_SECRET_KEY"
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
                "env": [
                    {"name": "MINIO_ENDPOINT", "value": minio_endpoint},
                    {
                        "name": "MINIO_ACCESS_KEY",
                        "valueFrom": {"secretKeyRef": {"name": DSPA_S3_SECRET, "key": "accesskey"}},
                    },
                    {
                        "name": "MINIO_SECRET_KEY",
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
