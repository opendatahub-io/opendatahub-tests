"""Persistent DSPA and locally generated datasets for AutoML upgrade runs."""

import base64
import shlex
import uuid
from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.data_science_pipelines_application import DataSciencePipelinesApplication
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from timeout_sampler import TimeoutExpiredError

from tests.pipelines_components.automl.data.synthetic import synthetic_automl_csv
from tests.pipelines_components.constants import (
    AUTOML_TASK_CONFIGS,
    AUTOML_TIMESERIES_CONFIG,
    DSPA_MINIO_IMAGE,
    DSPA_NAME,
    DSPA_PIPELINE_DEPLOYMENT,
    DSPA_READY_BUFFER_SECONDS,
    DSPA_S3_BUCKET,
    DSPA_S3_SECRET,
    MANAGED_PIPELINE_AUTOML_TABULAR,
    MANAGED_PIPELINE_AUTOML_TIMESERIES,
    MANAGED_PIPELINE_POLL_INTERVAL,
    MANAGED_PIPELINE_WAIT_TIMEOUT,
    MANAGED_PIPELINES_IMAGE,
    MINIO_MC_IMAGE,
    MINIO_UPLOADER_SECURITY_CONTEXT,
)
from tests.pipelines_components.utils import create_pipeline_run_managed, wait_for_managed_pipeline
from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import DscComponents, Timeout
from utilities.general import collect_pod_information
from utilities.infra import create_ns, wait_for_dsc_status_ready

UPGRADE_NAMESPACE = "automl-aqa-upgrade"


@pytest.fixture(scope="session")
def pre_upgrade_pipelines_dsc_patch(
    pytestconfig: pytest.Config,
    dsc_resource: DataScienceCluster,
) -> DataScienceCluster:
    """Keep AI Pipelines managed through the upgrade boundary."""
    if pytestconfig.option.pre_upgrade:
        current_state = dsc_resource.instance.spec.components.get("aipipelines", {}).get("managementState")
        if current_state != DscComponents.ManagementState.MANAGED:
            assert current_state == DscComponents.ManagementState.REMOVED, (
                f"Unexpected AI Pipelines managementState: {current_state}"
            )
            ResourceEditor(
                patches={dsc_resource: {"spec": {"components": {"aipipelines": {"managementState": "Managed"}}}}}
            ).update()
            wait_for_dsc_status_ready(dsc_resource=dsc_resource)
    return dsc_resource


@pytest.fixture(scope="session")
def post_upgrade_pipelines_dsc_restore(
    pytestconfig: pytest.Config,
    dsc_resource: DataScienceCluster,
) -> Generator[DataScienceCluster, Any, Any]:
    """Restore AI Pipelines after the post-upgrade namespace is removed."""
    if pytestconfig.option.post_upgrade:
        current_state = dsc_resource.instance.spec.components.get("aipipelines", {}).get("managementState")
        assert current_state == DscComponents.ManagementState.MANAGED, (
            f"Expected AI Pipelines to remain Managed through upgrade, got {current_state}"
        )
    yield dsc_resource
    if pytestconfig.option.post_upgrade:
        current_state = dsc_resource.instance.spec.components.get("aipipelines", {}).get("managementState")
        if current_state == DscComponents.ManagementState.MANAGED:
            ResourceEditor(
                patches={dsc_resource: {"spec": {"components": {"aipipelines": {"managementState": "Removed"}}}}}
            ).update()


@pytest.fixture(scope="session")
def upgrade_namespace(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    pre_upgrade_pipelines_dsc_patch: DataScienceCluster,
    post_upgrade_pipelines_dsc_restore: DataScienceCluster,
) -> Generator[Namespace, Any, Any]:
    """Preserve the AutoML namespace pre-upgrade and delete it post-upgrade."""
    if pytestconfig.option.pre_upgrade:
        assert not Namespace(client=admin_client, name=UPGRADE_NAMESPACE).exists, (
            f"Namespace {UPGRADE_NAMESPACE} already exists"
        )
        with create_ns(admin_client=admin_client, name=UPGRADE_NAMESPACE, teardown=False) as namespace:
            assert isinstance(namespace, Namespace)
            yield namespace
    else:
        namespace = Namespace(client=admin_client, name=UPGRADE_NAMESPACE, ensure_exists=True)
        yield namespace
        if pytestconfig.option.post_upgrade:
            namespace.clean_up()


@pytest.fixture(scope="session")
def upgrade_dspa(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    upgrade_namespace: Namespace,
) -> Generator[DataSciencePipelinesApplication, Any, Any]:
    """Create the managed DSPA pre-upgrade and reuse it post-upgrade."""
    if pytestconfig.option.pre_upgrade:
        managed_pipelines_spec: dict[str, Any] = {}
        if MANAGED_PIPELINES_IMAGE:
            managed_pipelines_spec["image"] = MANAGED_PIPELINES_IMAGE
        with DataSciencePipelinesApplication(
            client=admin_client,
            name=DSPA_NAME,
            namespace=upgrade_namespace.name,
            dsp_version="v2",
            api_server={"enableSamplePipeline": False, "managedPipelines": managed_pipelines_spec},
            object_storage={
                "disableHealthCheck": False,
                "enableExternalRoute": False,
                "minio": {"deploy": True, "image": DSPA_MINIO_IMAGE},
            },
            teardown=False,
        ) as dspa_resource:
            Deployment(
                client=admin_client, name=DSPA_PIPELINE_DEPLOYMENT, namespace=upgrade_namespace.name
            ).wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
            yield dspa_resource
    else:
        dspa_resource = DataSciencePipelinesApplication(
            client=admin_client, name=DSPA_NAME, namespace=upgrade_namespace.name, ensure_exists=True
        )
        Deployment(
            client=admin_client, name=DSPA_PIPELINE_DEPLOYMENT, namespace=upgrade_namespace.name
        ).wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
        yield dspa_resource
        if pytestconfig.option.post_upgrade:
            dspa_resource.clean_up()


@pytest.fixture(scope="session")
def upgrade_dspa_api_url(admin_client: DynamicClient, upgrade_dspa: DataSciencePipelinesApplication) -> str:
    """Return the KFP API URL for the persistent DSPA."""
    route = Route(
        client=admin_client, name=DSPA_PIPELINE_DEPLOYMENT, namespace=upgrade_dspa.namespace, ensure_exists=True
    )
    return f"https://{route.host}"


@pytest.fixture(scope="session")
def upgrade_dspa_auth_headers(current_client_token: str) -> dict[str, str]:
    """Return authentication headers for the KFP API."""
    return {"Authorization": f"Bearer {current_client_token}"}


@pytest.fixture(scope="session")
def upgrade_dspa_ca_bundle_file(admin_client: DynamicClient) -> str:
    """Return the cluster CA bundle for the KFP Route."""
    return create_ca_bundle_file(client=admin_client)


@pytest.fixture(scope="session")
def upgrade_dspa_s3_credentials(admin_client: DynamicClient, upgrade_dspa: DataSciencePipelinesApplication) -> Secret:
    """Add the AWS-style credentials required by managed pipeline components."""
    secret = Secret(client=admin_client, name=DSPA_S3_SECRET, namespace=upgrade_dspa.namespace, ensure_exists=True)
    access_key = base64.b64decode(secret.instance.data.get("accesskey", "")).decode()
    secret_key = base64.b64decode(secret.instance.data.get("secretkey", "")).decode()
    endpoint = f"http://minio-{DSPA_NAME}.{upgrade_dspa.namespace}.svc.cluster.local:9000"
    secret.update(
        resource_dict={
            "metadata": {"name": secret.name, "namespace": upgrade_dspa.namespace},
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


@pytest.fixture(scope="function")
def upgrade_managed_pipeline(
    upgrade_dspa_api_url: str,
    upgrade_dspa_auth_headers: dict[str, str],
    upgrade_dspa_ca_bundle_file: str,
    task_type: str,
) -> dict[str, str]:
    """Discover the managed tabular or timeseries pipeline."""
    display_name = MANAGED_PIPELINE_AUTOML_TIMESERIES if task_type == "timeseries" else MANAGED_PIPELINE_AUTOML_TABULAR
    return wait_for_managed_pipeline(
        api_url=upgrade_dspa_api_url,
        headers=upgrade_dspa_auth_headers,
        display_name=display_name,
        ca_bundle=upgrade_dspa_ca_bundle_file,
        timeout=DSPA_READY_BUFFER_SECONDS + MANAGED_PIPELINE_WAIT_TIMEOUT,
        poll_interval=MANAGED_PIPELINE_POLL_INTERVAL,
    )


@pytest.fixture(scope="function")
def upgrade_train_data(
    admin_client: DynamicClient,
    upgrade_namespace: Namespace,
    upgrade_dspa_s3_credentials: Secret,
    task_type: str,
) -> str:
    """Upload 150 generated rows to the persistent DSPA MinIO."""
    destination_key = f"automl-upgrade/{task_type}/train.csv"
    endpoint = f"http://minio-{DSPA_NAME}.{upgrade_namespace.name}.svc.cluster.local:9000"
    bucket = shlex.quote(s=DSPA_S3_BUCKET)
    key = shlex.quote(s=destination_key)
    script = (
        "export MC_CONFIG_DIR=/work/.mc && "
        "mc alias set dspa $DST_ENDPOINT $DST_ACCESS_KEY $DST_SECRET_KEY && "
        f"mc mb --ignore-existing dspa/{bucket} && "
        f"mc cp /data/train.csv dspa/{bucket}/{key}"
    )
    with (
        ConfigMap(
            client=admin_client,
            name=f"automl-upgrade-data-{uuid.uuid4().hex[:8]}",
            namespace=upgrade_namespace.name,
            data={"train.csv": synthetic_automl_csv(task_type=task_type)},
        ) as training_data,
        Pod(
            client=admin_client,
            name=f"automl-upgrade-uploader-{uuid.uuid4().hex[:8]}",
            namespace=upgrade_namespace.name,
            restart_policy="Never",
            volumes=[
                {"name": "work", "emptyDir": {}},
                {"name": "training-data", "configMap": {"name": training_data.name}},
            ],
            containers=[
                {
                    "name": "minio-uploader",
                    "image": MINIO_MC_IMAGE,
                    "command": ["/bin/sh", "-c"],
                    "args": [script],
                    "volumeMounts": [
                        {"name": "work", "mountPath": "/work"},
                        {"name": "training-data", "mountPath": "/data", "readOnly": True},
                    ],
                    "securityContext": MINIO_UPLOADER_SECURITY_CONTEXT,
                    "env": [
                        {"name": "DST_ENDPOINT", "value": endpoint},
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
        ) as upload_pod,
    ):
        try:
            upload_pod.wait_for_status(status="Succeeded", timeout=120)
        except TimeoutExpiredError:
            collect_pod_information(pod=upload_pod)
            raise
    return destination_key


@pytest.fixture(scope="function")
def upgrade_run_id(
    upgrade_dspa_api_url: str,
    upgrade_dspa_auth_headers: dict[str, str],
    upgrade_dspa_ca_bundle_file: str,
    upgrade_managed_pipeline: dict[str, str],
    upgrade_train_data: str,
    task_type: str,
) -> str:
    """Submit the managed AutoML pipeline with its generated dataset."""
    parameters: dict[str, Any] = {
        "train_data_secret_name": DSPA_S3_SECRET,
        "train_data_bucket_name": DSPA_S3_BUCKET,
        "train_data_file_key": upgrade_train_data,
    }
    if task_type == "timeseries":
        parameters.update(AUTOML_TIMESERIES_CONFIG)
    else:
        task_config = AUTOML_TASK_CONFIGS[task_type]
        parameters.update({key: task_config[key] for key in ("label_column", "task_type", "top_n")})
    return create_pipeline_run_managed(
        api_url=upgrade_dspa_api_url,
        headers=upgrade_dspa_auth_headers,
        pipeline_id=upgrade_managed_pipeline["pipeline_id"],
        pipeline_version_id=upgrade_managed_pipeline["pipeline_version_id"],
        run_name=f"automl-upgrade-{task_type}",
        parameters=parameters,
        ca_bundle=upgrade_dspa_ca_bundle_file,
    )
