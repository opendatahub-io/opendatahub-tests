from contextlib import contextmanager
from typing import Any, Generator

from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from pyhelper_utils.shell import run_command
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutExpiredError

from utilities.constants import ApiGroups, Labels, MinIo, Protocols, Timeout
from utilities.general import get_s3_secret_dict

LOGGER = get_logger(name=__name__)


def log_minio_diagnostics(namespace: str, name: str = "minio") -> None:
    """Log the resources needed to diagnose a MinIO deployment timeout."""
    commands = (
        ["oc", "get", "deployment", name, "-n", namespace, "-o", "yaml"],
        ["oc", "get", "pods", "-n", namespace, "-l", "app.kubernetes.io/name=minio", "-o", "yaml"],
        ["oc", "get", "pvc", "minio", "-n", namespace, "-o", "yaml"],
        ["oc", "get", "events", "-n", namespace, "--sort-by=.lastTimestamp"],
        ["oc", "logs", f"deployment/{name}", "-n", namespace, "--all-containers=true", "--tail=200"],
    )
    for command in commands:
        result = run_command(command=command, check=False)
        LOGGER.error("MinIO diagnostics (%s):\n%s\n%s", " ".join(command), result[1], result[2])


def wait_for_minio_deployment(deployment: Deployment) -> None:
    """Wait for MinIO and preserve actionable diagnostics when it is not ready."""
    try:
        deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
    except TimeoutExpiredError:
        log_minio_diagnostics(namespace=deployment.namespace, name=deployment.name)
        raise


@contextmanager
def create_minio_data_connection_secret(
    minio_service: Service,
    model_namespace: str,
    aws_s3_bucket: str,
    client: DynamicClient,
) -> Generator[Secret, Any, Any]:
    """
    Create a secret for minio data connection.

    Args:
        minio_service (Service): The service for minio.
        model_namespace (str): The namespace where the model is stored.
        aws_s3_bucket (str): The name of the bucket.
        client (DynamicClient): The client to use for creating the secret.

    Yields:
        Secret: The secret for minio data connection.
    """
    data_dict = get_s3_secret_dict(
        aws_access_key=MinIo.Credentials.ACCESS_KEY_VALUE,
        aws_secret_access_key=MinIo.Credentials.SECRET_KEY_VALUE,  # pragma: allowlist secret
        aws_s3_bucket=aws_s3_bucket,
        aws_s3_endpoint=f"{Protocols.HTTP}://{minio_service.instance.spec.clusterIP}:{str(MinIo.Metadata.DEFAULT_PORT)}",  # noqa: E501
        aws_s3_region="us-south",
    )
    with Secret(
        client=client,
        name="aws-connection-minio-data-connection",
        namespace=model_namespace,
        data_dict=data_dict,
        label={
            Labels.OpenDataHub.DASHBOARD: "true",
            Labels.OpenDataHubIo.MANAGED: "true",
        },
        annotations={
            f"{ApiGroups.OPENDATAHUB_IO}/connection-type": "s3",
            "openshift.io/display-name": "Minio Data Connection",
        },
    ) as minio_secret:
        yield minio_secret
