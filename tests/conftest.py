from typing import Tuple, Any, Generator

import pytest
from pytest import FixtureRequest, Config
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.resource import get_client

from utilities.infra import create_ns


@pytest.fixture(scope="session")
def admin_client() -> DynamicClient:
    return get_client()


@pytest.fixture(scope="class")
def model_namespace(request: FixtureRequest, admin_client: DynamicClient) -> Generator[Namespace, Any, Any]:
    with create_ns(admin_client=admin_client, name=request.param["name"]) as ns:
        yield ns


@pytest.fixture(scope="session")
def aws_access_key_id(pytestconfig: Config) -> str:
    access_key = pytestconfig.option.aws_access_key_id
    if not access_key:
        raise ValueError(
            "AWS access key id is not set. "
            "Either pass with `--aws-access-key-id` or set `AWS_ACCESS_KEY_ID` environment variable"
        )
    return access_key


@pytest.fixture(scope="session")
def aws_secret_access_key(pytestconfig: Config) -> str:
    secret_access_key = pytestconfig.option.aws_secret_access_key
    if not secret_access_key:
        raise ValueError(
            "AWS secret access key is not set. "
            "Either pass with `--aws-secret-access-key` or set `AWS_SECRET_ACCESS_KEY` environment variable"
        )
    return secret_access_key


@pytest.fixture(scope="session")
def valid_aws_config(aws_access_key_id: str, aws_secret_access_key: str) -> Tuple[str, str]:
    return aws_access_key_id, aws_secret_access_key


@pytest.fixture(scope="session")
def ci_s3_bucket_name(pytestconfig: Config) -> str:
    bucket_name = pytestconfig.option.ci_s3_bucket_name
    if not bucket_name:
        raise ValueError(
            "CI S3 bucket name is not set. "
            "Either pass with `--ci-s3-bucket-name` or set `CI_S3_BUCKET_NAME` environment variable"
        )
    return bucket_name


@pytest.fixture(scope="session")
def ci_s3_bucket_region(pytestconfig: pytest.Config) -> str:
    ci_bucket_region = pytestconfig.option.ci_s3_bucket_region
    if not ci_bucket_region:
        raise ValueError(
            "Region for the ci s3 bucket is not defined."
            "Either pass with `--ci-s3-bucket-region` or set `CI_S3_BUCKET_REGION` environment variable"
        )
    return ci_bucket_region


@pytest.fixture(scope="session")
def ci_s3_bucket_endpoint(pytestconfig: pytest.Config) -> str:
    ci_bucket_endpoint = pytestconfig.option.ci_s3_bucket_endpoint
    if not ci_bucket_endpoint:
        raise ValueError(
            "Endpoint for the ci s3 bucket is not defined."
            "Either pass with `--ci-s3-bucket-endpoint` or set `CI_S3_BUCKET_ENDPOINT` environment variable"
        )
    return ci_bucket_endpoint


@pytest.fixture(scope="session")
def models_s3_bucket_name(pytestconfig: pytest.Config) -> str:
    models_bucket = pytestconfig.option.models_s3_bucket_name
    if not models_bucket:
        raise ValueError(
            "Bucket name for the models bucket is not defined."
            "Either pass with `--models-s3-bucket-name` or set `MODELS_S3_BUCKET_NAME` environment variable"
        )
    return models_bucket


@pytest.fixture(scope="session")
def models_s3_bucket_region(pytestconfig: pytest.Config) -> str:
    models_bucket_region = pytestconfig.option.models_s3_bucket_region
    if not models_bucket_region:
        raise ValueError(
            "region for the models bucket is not defined."
            "Either pass with `--models-s3-bucket-region` or set `MODELS_S3_BUCKET_REGION` environment variable"
        )
    return models_bucket_region


@pytest.fixture(scope="session")
def models_s3_bucket_endpoint(pytestconfig: pytest.Config) -> str:
    models_bucket_endpoint = pytestconfig.option.models_s3_bucket_endpoint
    if not models_bucket_endpoint:
        raise ValueError(
            "endpoint for the models bucket is not defined."
            "Either pass with `--models-s3-bucket-endpoint` or set `MODELS_S3_BUCKET_ENDPOINT` environment variable"
        )
    return models_bucket_endpoint
