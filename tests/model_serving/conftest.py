from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret


@pytest.fixture(scope="session")
def root_dir(pytestconfig: pytest.Config) -> Any:
    """
    Provides the root directory path of the pytest project for the entire test session.

    Args:
        pytestconfig (pytest.Config): The pytest configuration object.

    Returns:
        Any: The root path of the pytest project.
    """
    return pytestconfig.rootpath


@pytest.fixture(scope="class")
def protocol(request: FixtureRequest) -> str:
    """
    Provides the protocol type parameter for the test class.

    Args:
        request (pytest.FixtureRequest): The pytest fixture request object.

    Returns:
        str: The protocol type specified in the test parameter.
    """
    return request.param["protocol_type"]


@pytest.fixture(scope="session")
def s3_models_storage_uri(request: FixtureRequest, models_s3_bucket_name: str) -> str:
    return f"s3://{models_s3_bucket_name}/{request.param['model-dir']}/"


@pytest.fixture(scope="class")
def kserve_s3_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Secret:
    """
    Creates and yields a Kubernetes Secret configured for S3 access in KServe.

    Args:
        admin_client (DynamicClient): Kubernetes dynamic client.
        model_namespace (Namespace): Namespace where the secret will be created.
        aws_access_key_id (str): AWS access key ID.
        aws_secret_access_key (str): AWS secret access key.
        models_s3_bucket_region (str): AWS S3 bucket region.
        models_s3_bucket_endpoint (str): AWS S3 bucket endpoint URL.

    Yields:
        Secret: A Kubernetes Secret configured with the provided AWS credentials and S3 endpoint.
    """
    with kserve_s3_endpoint_secret(
        admin_client=admin_client,
        name="models-bucket-secret",
        namespace=model_namespace.name,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=models_s3_bucket_region,
        aws_s3_endpoint=models_s3_bucket_endpoint,
    ) as secret:
        yield secret


@contextmanager
def kserve_s3_endpoint_secret(
    admin_client: DynamicClient,
    name: str,
    namespace: str,
    aws_access_key: str,
    aws_secret_access_key: str,
    aws_s3_endpoint: str,
    aws_s3_region: str,
) -> Generator[Secret, Any, Any]:
    """
    Context manager that creates a temporary Kubernetes Secret for KServe
    to access an S3-compatible storage endpoint.

    Args:
        admin_client (DynamicClient): Kubernetes dynamic client for resource operations.
        name (str): Name of the Secret resource.
        namespace (str): Kubernetes namespace in which to create the Secret.
        aws_access_key (str): AWS access key ID for authentication.
        aws_secret_access_key (str): AWS secret access key for authentication.
        aws_s3_endpoint (str): S3 endpoint URL (e.g., https://s3.example.com).
        aws_s3_region (str): AWS region for the S3 service.

    Yields:
        Secret: The created Kubernetes Secret object within the context.
    """
    with Secret(
        client=admin_client,
        name=name,
        namespace=namespace,
        annotations={
            "serving.kserve.io/s3-endpoint": (aws_s3_endpoint.replace("https://", "").replace("http://", "")),
            "serving.kserve.io/s3-region": aws_s3_region,
            "serving.kserve.io/s3-useanoncredential": "false",
            "serving.kserve.io/s3-verifyssl": "0",
            "serving.kserve.io/s3-usehttps": "1",
        },
        string_data={
            "AWS_ACCESS_KEY_ID": aws_access_key,
            "AWS_SECRET_ACCESS_KEY": aws_secret_access_key,
        },
        wait_for_resource=True,
    ) as secret:
        yield secret
