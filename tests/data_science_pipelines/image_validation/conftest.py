import uuid
from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.data_science_pipelines_application import DataSciencePipelinesApplication
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from pytest_testconfig import config as py_config

from tests.data_science_pipelines.image_validation.constants import (
    DSPA_NAME,
    OBJECT_STORAGE_SECRET_NAME,
    OPERATOR_DEPLOYMENT_NAME,
    OPERATOR_LABEL_SELECTOR,
    PIPELINE_SERVER_DEPLOYMENTS,
)
from tests.data_science_pipelines.image_validation.utils import pod_for_deployment
from utilities.constants import DscComponents, Timeout
from utilities.data_science_cluster_utils import update_components_in_dsc
from utilities.infra import create_ns


@pytest.fixture(scope="class")
def enabled_aipipelines_in_dsc(
    dsc_resource: DataScienceCluster,
) -> Generator[DataScienceCluster, Any, Any]:
    """Set AI Pipelines to Managed and restore the previous management state."""
    with update_components_in_dsc(
        dsc=dsc_resource,
        components={DscComponents.AIPIPELINES: DscComponents.ManagementState.MANAGED},
    ) as dsc:
        yield dsc


@pytest.fixture(scope="class")
def pipelines_image_namespace(admin_client: DynamicClient) -> Generator[Namespace, Any, Any]:
    """Namespace for the pipeline server whose images are checked."""
    with create_ns(
        admin_client=admin_client,
        name=f"dsp-images-{uuid.uuid4().hex[:8]}",
    ) as namespace:
        yield namespace


@pytest.fixture(scope="class")
def pipelines_object_storage_secret(
    admin_client: DynamicClient,
    pipelines_image_namespace: Namespace,
) -> Generator[Secret, Any, Any]:
    """Placeholder object-storage credentials so the DSPA can render Deployments."""
    with Secret(
        client=admin_client,
        name=OBJECT_STORAGE_SECRET_NAME,
        namespace=pipelines_image_namespace.name,
        string_data={
            "accesskey": "placeholder-access-key",
            "secretkey": "placeholder-secret-key",  # pragma: allowlist secret
        },
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def pipelines_server(
    admin_client: DynamicClient,
    pipelines_image_namespace: Namespace,
    pipelines_object_storage_secret: Secret,
    enabled_aipipelines_in_dsc: DataScienceCluster,
) -> Generator[DataSciencePipelinesApplication, Any, Any]:
    """Create one v2 pipeline server with built-in MariaDB and external object storage.

    Health checks are disabled so Deployments are created without a reachable database or
    object store. Pods do not need to become Ready.
    """
    with DataSciencePipelinesApplication(
        client=admin_client,
        name=DSPA_NAME,
        namespace=pipelines_image_namespace.name,
        dsp_version="v2",
        api_server={"deploy": True, "enableSamplePipeline": False},
        persistence_agent={"deploy": True},
        scheduled_workflow={"deploy": True},
        workflow_controller={"deploy": True},
        mlmd={"deploy": True},
        database={"disableHealthCheck": True, "mariaDB": {"deploy": True}},
        object_storage={
            "disableHealthCheck": True,
            "externalStorage": {
                "host": "s3.example.com",
                "bucket": "mlpipeline",
                "scheme": "https",
                "s3CredentialsSecret": {
                    "secretName": pipelines_object_storage_secret.name,
                    "accessKey": "accesskey",
                    "secretKey": "secretkey",  # pragma: allowlist secret
                },
            },
        },
    ) as dspa:
        for deployment_name in PIPELINE_SERVER_DEPLOYMENTS:
            Deployment(
                client=admin_client,
                name=deployment_name,
                namespace=pipelines_image_namespace.name,
            ).wait(timeout=Timeout.TIMEOUT_5MIN)
        yield dspa


@pytest.fixture(scope="class")
def data_science_pipelines_pods(
    admin_client: DynamicClient,
    pipelines_image_namespace: Namespace,
    pipelines_server: DataSciencePipelinesApplication,
    enabled_aipipelines_in_dsc: DataScienceCluster,
) -> list[Pod]:
    """Pods for the operator controller and the pipeline-server Deployments."""
    pods = [
        pod_for_deployment(
            admin_client=admin_client,
            namespace=py_config["applications_namespace"],
            deployment_name=OPERATOR_DEPLOYMENT_NAME,
            label_selector=OPERATOR_LABEL_SELECTOR,
        )
    ]
    pods.extend(
        pod_for_deployment(
            admin_client=admin_client,
            namespace=pipelines_image_namespace.name,
            deployment_name=deployment_name,
            label_selector=f"app={deployment_name}",
        )
        for deployment_name in PIPELINE_SERVER_DEPLOYMENTS
    )
    return pods
