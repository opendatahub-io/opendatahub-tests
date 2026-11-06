"""Unit tests for tenant model S3 secret alignment."""

from tests.ai_gateway.models_as_a_service.multitenancy.utils import s3_secret_config_for_storage_uri
from utilities.constants import ModelStorage


def test_s3_secret_config_for_ods_ci_wisdom_uri_uses_us_east_2() -> None:
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
    bucket, region, endpoint = s3_secret_config_for_storage_uri(
        storage_uri="s3://my-bucket/some-model/",
        aws_s3_bucket="my-bucket",
        aws_s3_region="eu-central-1",
        aws_s3_endpoint="https://s3.eu-central-1.amazonaws.com",
    )
    assert bucket == "my-bucket"
    assert region == "eu-central-1"
    assert endpoint == "https://s3.eu-central-1.amazonaws.com"
