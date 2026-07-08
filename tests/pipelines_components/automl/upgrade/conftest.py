import base64
import os
import shlex
import uuid
from collections.abc import Generator
from typing import Any

import pytest
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
from timeout_sampler import TimeoutExpiredError

from tests.pipelines_components.automl.upgrade.utils import (
    load_baseline_from_configmap,
    save_baseline_to_configmap,
)
from tests.pipelines_components.constants import (
    AUTOML_PIPELINE_YAML,
    AUTOML_S3_BUCKET,
    AUTOML_TASK_CONFIGS,
    AUTOML_TRAIN_DATA_FILE_KEY,
    DSPA_MINIO_IMAGE,
    DSPA_NAME,
    DSPA_PIPELINE_DEPLOYMENT,
    DSPA_READY_BUFFER_SECONDS,
    DSPA_S3_BUCKET,
    DSPA_S3_SECRET,
    EXTERNAL_S3_SECRET,
    MANAGED_PIPELINE_AUTOML_TABULAR,
    MANAGED_PIPELINE_POLL_INTERVAL,
    MANAGED_PIPELINE_WAIT_TIMEOUT,
    MANAGED_PIPELINES_IMAGE,
    MINIO_MC_IMAGE,
    MINIO_UPLOADER_SECURITY_CONTEXT,
)
from tests.pipelines_components.utils import (
    create_pipeline_run,
    create_pipeline_run_managed,
    resolve_pipeline_yaml,
    upload_pipeline,
    use_managed_pipelines,
    wait_for_managed_pipeline,
)
from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import DscComponents, Timeout
from utilities.general import collect_pod_information
from utilities.infra import create_ns, wait_for_dsc_status_ready

LOGGER = structlog.get_logger(name=__name__)

UPGRADE_NAMESPACE = "automl-upgrade"
UPGRADE_TASK_TYPE = "regression"
UPGRADE_RUN_DISPLAY_NAME = "automl-upgrade-regression"


# ---------------------------------------------------------------------------
# DSC patch — enable AI Pipelines (non-reverting for upgrade persistence)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pre_upgrade_pipelines_dsc_patch(
    pytestconfig: pytest.Config,
    dsc_resource: DataScienceCluster,
) -> DataScienceCluster:
    """Enable AI Pipelines in DSC before upgrade tests.

    Uses ResourceEditor.update() (non-reverting) so the component stays
    Managed through the upgrade boundary.  No-op during post-upgrade.
    """
    if not pytestconfig.option.pre_upgrade:
        return dsc_resource

    current_state = dsc_resource.instance.spec.components.get("aipipelines", {}).get("managementState")
    if current_state != DscComponents.ManagementState.MANAGED:
        LOGGER.info("Setting AI Pipelines to Managed state")
        editor = ResourceEditor(
            patches={dsc_resource: {"spec": {"components": {"aipipelines": {"managementState": "Managed"}}}}}
        )
        editor.update()
        wait_for_dsc_status_ready(dsc_resource=dsc_resource)

    return dsc_resource


# ---------------------------------------------------------------------------
# Namespace — fixed name, conditional create/reference
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pipelines_namespace(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    pre_upgrade_pipelines_dsc_patch: DataScienceCluster,
) -> Generator[Namespace, Any, Any]:
    """Fixed-name namespace for AutoML upgrade tests."""
    pre = pytestconfig.option.pre_upgrade
    post = pytestconfig.option.post_upgrade
    should_cleanup = not pre or post

    if pre:
        ns = Namespace(client=admin_client, name=UPGRADE_NAMESPACE)
        if ns.exists:
            raise AssertionError(
                f"Namespace {UPGRADE_NAMESPACE} already exists. "
                "This indicates a previous test run did not clean up properly."
            )
        with create_ns(
            admin_client=admin_client,
            name=UPGRADE_NAMESPACE,
            teardown=should_cleanup,
        ) as ns:
            yield ns
    else:
        ns = Namespace(client=admin_client, name=UPGRADE_NAMESPACE)
        yield ns
        if should_cleanup:
            ns.clean_up()


# ---------------------------------------------------------------------------
# DSPA — conditional create/reference
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def dspa(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
) -> Generator[DataSciencePipelinesApplication, Any, Any]:
    """DataSciencePipelinesApplication — created pre-upgrade, referenced post-upgrade."""
    pre = pytestconfig.option.pre_upgrade
    post = pytestconfig.option.post_upgrade
    should_cleanup = not pre or post

    if pre:
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
            teardown=should_cleanup,
        ) as dspa_resource:
            Deployment(
                client=admin_client,
                name=DSPA_PIPELINE_DEPLOYMENT,
                namespace=pipelines_namespace.name,
            ).wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
            yield dspa_resource
    else:
        dspa_resource = DataSciencePipelinesApplication(
            client=admin_client,
            name=DSPA_NAME,
            namespace=pipelines_namespace.name,
        )
        Deployment(
            client=admin_client,
            name=DSPA_PIPELINE_DEPLOYMENT,
            namespace=pipelines_namespace.name,
        ).wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
        yield dspa_resource
        if should_cleanup:
            dspa_resource.clean_up()


# ---------------------------------------------------------------------------
# DSPA API access fixtures (session-scoped overrides of parent class-scoped)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def dspa_route(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    dspa: DataSciencePipelinesApplication,
) -> Route:
    return Route(
        client=admin_client,
        name=DSPA_PIPELINE_DEPLOYMENT,
        namespace=pipelines_namespace.name,
        ensure_exists=True,
    )


@pytest.fixture(scope="session")
def dspa_api_url(dspa_route: Route) -> str:
    return f"https://{dspa_route.host}"


@pytest.fixture(scope="session")
def dspa_auth_headers(current_client_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {current_client_token}"}


@pytest.fixture(scope="session")
def dspa_ca_bundle_file(admin_client: DynamicClient) -> str:
    return create_ca_bundle_file(client=admin_client)


# ---------------------------------------------------------------------------
# S3 credentials (session-scoped overrides)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def dspa_s3_credentials(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    dspa: DataSciencePipelinesApplication,
) -> Secret:
    """Patch DSPA S3 secret with standard AWS credential fields."""
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


@pytest.fixture(scope="session")
def external_s3_secret(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
) -> Generator[Secret, Any, Any]:
    """Transient secret for external AWS S3 credentials (training data download)."""
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    assert aws_access_key_id, "Environment variable 'AWS_ACCESS_KEY_ID' is not set."
    assert aws_secret_access_key, "Environment variable 'AWS_SECRET_ACCESS_KEY' is not set."

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


# ---------------------------------------------------------------------------
# Training data upload — only during pre-upgrade
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def upgrade_train_data(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    dspa_s3_credentials: Secret,
    external_s3_secret: Secret,
) -> str | None:
    """Upload regression training data to DSPA MinIO. No-op when pre-upgrade is not set."""
    if not pytestconfig.option.pre_upgrade:
        return None

    env_var = f"AUTOML_{UPGRADE_TASK_TYPE.upper()}_S3_TRAIN_DATA_KEY"
    src_key_value = os.environ.get(env_var)
    assert src_key_value, (
        f"Environment variable '{env_var}' is not set. "
        f"Set it in .env or shell to provide the S3 key for {UPGRADE_TASK_TYPE} training data."
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

    pod_name = f"automl-upgrade-uploader-{uuid.uuid4().hex[:8]}"
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


# ---------------------------------------------------------------------------
# Managed pipeline discovery
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def automl_managed_pipeline(
    dspa: DataSciencePipelinesApplication,
    dspa_api_url: str,
    dspa_auth_headers: dict[str, str],
    dspa_ca_bundle_file: str,
) -> dict[str, str] | None:
    """Discover managed AutoML tabular pipeline. None in legacy YAML mode."""
    if not use_managed_pipelines(yaml_env_value=AUTOML_PIPELINE_YAML):
        return None
    return wait_for_managed_pipeline(
        api_url=dspa_api_url,
        headers=dspa_auth_headers,
        display_name=MANAGED_PIPELINE_AUTOML_TABULAR,
        ca_bundle=dspa_ca_bundle_file,
        timeout=DSPA_READY_BUFFER_SECONDS + MANAGED_PIPELINE_WAIT_TIMEOUT,
        poll_interval=MANAGED_PIPELINE_POLL_INTERVAL,
    )


# ---------------------------------------------------------------------------
# Pipeline run — created pre-upgrade, loaded from baseline post-upgrade
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def upgrade_run_id(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    dspa_api_url: str,
    dspa_auth_headers: dict[str, str],
    dspa_ca_bundle_file: str,
    automl_managed_pipeline: dict[str, str] | None,
    upgrade_train_data: str | None,
) -> str:
    """Pipeline run ID — created when pre-upgrade is set, loaded from ConfigMap otherwise."""
    if not pytestconfig.option.pre_upgrade:
        baselines = load_baseline_from_configmap(
            client=admin_client,
            namespace=pipelines_namespace.name,
        )
        return baselines["run_id"]

    task_config = AUTOML_TASK_CONFIGS[UPGRADE_TASK_TYPE]
    parameters: dict[str, Any] = {
        "train_data_secret_name": DSPA_S3_SECRET,
        "train_data_bucket_name": DSPA_S3_BUCKET,
        "train_data_file_key": AUTOML_TRAIN_DATA_FILE_KEY,
        "label_column": task_config["label_column"],
        "task_type": task_config["task_type"],
        "top_n": task_config["top_n"],
    }

    if automl_managed_pipeline is not None:
        run_id = create_pipeline_run_managed(
            api_url=dspa_api_url,
            headers=dspa_auth_headers,
            pipeline_id=automl_managed_pipeline["pipeline_id"],
            pipeline_version_id=automl_managed_pipeline["pipeline_version_id"],
            run_name=UPGRADE_RUN_DISPLAY_NAME,
            parameters=parameters,
            ca_bundle=dspa_ca_bundle_file,
        )
    else:
        pipeline_yaml_path = resolve_pipeline_yaml(value=AUTOML_PIPELINE_YAML)
        pipeline_id = upload_pipeline(
            api_url=dspa_api_url,
            headers=dspa_auth_headers,
            pipeline_yaml_path=pipeline_yaml_path,
            pipeline_name=f"automl-upgrade-{pipelines_namespace.name}",
            ca_bundle=dspa_ca_bundle_file,
        )
        run_id = create_pipeline_run(
            api_url=dspa_api_url,
            headers=dspa_auth_headers,
            pipeline_id=pipeline_id,
            run_name=UPGRADE_RUN_DISPLAY_NAME,
            parameters=parameters,
            ca_bundle=dspa_ca_bundle_file,
        )

    return run_id


# ---------------------------------------------------------------------------
# Baseline capture / load
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def automl_capture_upgrade_baseline(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    upgrade_run_id: str,
    automl_managed_pipeline: dict[str, str] | None,
) -> None:
    """Capture baseline after pre-upgrade experiment completes. No-op when pre-upgrade is not set."""
    if not pytestconfig.option.pre_upgrade:
        return

    baselines: dict[str, Any] = {
        "run_id": upgrade_run_id,
        "run_display_name": UPGRADE_RUN_DISPLAY_NAME,
    }
    if automl_managed_pipeline is not None:
        baselines["pipeline_id"] = automl_managed_pipeline["pipeline_id"]
        baselines["pipeline_version_id"] = automl_managed_pipeline["pipeline_version_id"]

    save_baseline_to_configmap(
        client=admin_client,
        namespace=pipelines_namespace.name,
        baselines=baselines,
    )


@pytest.fixture(scope="session")
def automl_upgrade_baseline(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
) -> dict:
    """Load pre-upgrade baseline. Returns empty dict when post-upgrade is not set."""
    if not pytestconfig.option.post_upgrade:
        return {}

    return load_baseline_from_configmap(
        client=admin_client,
        namespace=pipelines_namespace.name,
    )
