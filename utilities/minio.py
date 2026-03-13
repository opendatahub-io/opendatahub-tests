from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from kubernetes.dynamic import DynamicClient
from ocp_resources.secret import Secret
from ocp_resources.service import Service

from utilities.constants import ApiGroups, Labels, MinIo, Protocols
from utilities.general import get_s3_secret_dict


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
        aws_s3_endpoint=f"{Protocols.HTTP}://{minio_service.instance.spec.clusterIP}:{MinIo.Metadata.DEFAULT_PORT!s}",
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
