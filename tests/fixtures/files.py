from typing import Callable, Dict
import pytest
import os
from _pytest.fixtures import FixtureRequest

S3_BUCKET_NAME = os.getenv("LLS_FILES_S3_BUCKET_NAME", "opendatathub-tests-llama-stack")
S3_BUCKET_REGION = os.getenv("LLS_FILES_S3_REGION", "us-east-1")
S3_BUCKET_ENDPOINT_URL = os.getenv("AWS_DEFAULT_ENDPOINT", "https://s3.us-east-1.amazonaws.com")
S3_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minio")
S3_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")
S3_AUTO_CREATE_BUCKET = os.getenv("LLS_FILES_S3_AUTO_CREATE_BUCKET", "true")


@pytest.fixture(scope="class")
def files_provider_config_factory(
    request: FixtureRequest,
) -> Callable[[str], list[Dict[str, str]]]:
    """
    Factory fixture for configuring external files providers and returning their configuration.

    This fixture returns a factory function that can configure additional files storage providers
    (such as S3/minio) and return the necessary environment variables
    for configuring the LlamaStack server to use these providers.

    Args:
        request: Pytest fixture request object for accessing other fixtures

    Returns:
        Callable[[str], list[Dict[str, str]]]: Factory function that takes a provider name
        and returns a list of environment variable dictionaries

    Supported Providers:
        - "local": defaults to using just local filesystem
        - "s3": a remote S3/Minio storage provider

    Environment Variables by Provider:
        - "s3":
          * S3_BUCKET_NAME: Name of the S3/Minio bucket
          * AWS_DEFAULT_REGION: Region of the S3/Minio bucket
          * S3_ENDPOINT_URL: Endpoint URL of the S3/Minio bucket
          * AWS_ACCESS_KEY_ID: Access key ID for the S3/Minio bucket
          * AWS_SECRET_ACCESS_KEY: Secret access key for the S3/Minio bucket
          * S3_AUTO_CREATE_BUCKET: Whether to automatically create the S3/Minio bucket if it doesn't exist

    Example:
        def test_with_s3(files_provider_config_factory):
            env_vars = files_provider_config_factory("s3")
            # env_vars contains S3_BUCKET_NAME, S3_BUCKET_ENDPOINT_URL, etc.
    """

    def _factory(provider_name: str) -> list[Dict[str, str]]:
        env_vars: list[dict[str, str]] = []

        if provider_name == "local" or provider_name is None:
            # Default case - no additional environment variables needed
            pass
        elif provider_name == "s3":
            env_vars.append({"name": "ENABLE_S3", "value": "s3"})
            env_vars.append({"name": "S3_BUCKET_NAME", "value": S3_BUCKET_NAME})
            env_vars.append({"name": "AWS_DEFAULT_REGION", "value": S3_BUCKET_REGION})
            env_vars.append({"name": "S3_ENDPOINT_URL", "value": S3_BUCKET_ENDPOINT_URL})
            env_vars.append({"name": "AWS_ACCESS_KEY_ID", "value": S3_ACCESS_KEY_ID})
            env_vars.append({"name": "AWS_SECRET_ACCESS_KEY", "value": S3_SECRET_ACCESS_KEY})
            env_vars.append({"name": "S3_AUTO_CREATE_BUCKET", "value": S3_AUTO_CREATE_BUCKET})

        return env_vars

    return _factory
