"""Unit tests for tenant model S3 secret alignment."""

import pytest

from tests.ai_gateway.models_as_a_service.multitenancy.utils import s3_secret_config_for_storage_uri
from utilities.constants import ModelStorage

pytestmark = pytest.mark.tier1


def test_s3_secret_config_for_ods_ci_wisdom_uri_uses_us_east_2() -> None:
    """Given an ods-ci-wisdom model URI, when resolving S3 secret fields, then use us-east-2 CI bucket settings.

    Jenkins often passes us-east-1 pytest flags for unrelated buckets; the helper must not apply those
    to TinyLlama/Qwen URIs under ods-ci-wisdom.
    """
    bucket, region, endpoint = s3_secret_config_for_storage_uri(
        storage_uri=ModelStorage.S3.TINYLLAMA,
        aws_s3_bucket="ods-ci-s3",
        aws_s3_region="us-east-1",
        aws_s3_endpoint="https://s3.us-east-1.amazonaws.com",
    )
    assert bucket == ModelStorage.S3.ODS_CI_WISDOM_BUCKET
    assert region == ModelStorage.S3.ODS_CI_WISDOM_REGION
    assert endpoint == ModelStorage.S3.ODS_CI_WISDOM_ENDPOINT


def test_s3_secret_config_for_other_uri_uses_pytest_flags() -> None:
    """Given a non-ods-ci-wisdom URI, when resolving S3 secret fields, then keep pytest-supplied bucket settings."""
    bucket, region, endpoint = s3_secret_config_for_storage_uri(
        storage_uri="s3://my-bucket/some-model/",
        aws_s3_bucket="my-bucket",
        aws_s3_region="eu-central-1",
        aws_s3_endpoint="https://s3.eu-central-1.amazonaws.com",
    )
    assert bucket == "my-bucket"
    assert region == "eu-central-1"
    assert endpoint == "https://s3.eu-central-1.amazonaws.com"
