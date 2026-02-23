import os
from collections.abc import Callable

import pytest
from _pytest.fixtures import FixtureRequest

S3_AUTO_CREATE_BUCKET = os.getenv("LLS_FILES_S3_AUTO_CREATE_BUCKET", "true")


@pytest.fixture(scope="class")
def files_provider_config_factory(
    request: FixtureRequest,
) -> Callable[[str], list[dict[str, str]]]:
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
          * ENABLE_S3: Enables S3/Minio storage provider
          * CI_S3_BUCKET_NAME: Name of the S3/Minio bucket
          * CI_S3_BUCKET_REGION: Region of the S3/Minio bucket
          * CI_S3_BUCKET_ENDPOINT: Endpoint URL of the S3/Minio bucket
          * AWS_ACCESS_KEY_ID: Access key ID for the S3/Minio bucket
          * AWS_SECRET_ACCESS_KEY: Secret access key for the S3/Minio bucket
          * S3_AUTO_CREATE_BUCKET: Whether to automatically create the S3/Minio bucket if it doesn't exist

    Example:
        def test_with_s3(files_provider_config_factory):
            env_vars = files_provider_config_factory("s3")
            # env_vars contains S3_BUCKET_NAME, S3_BUCKET_ENDPOINT_URL, etc.
    """

    def _factory(provider_name: str) -> list[dict[str, str]]:
        env_vars: list[dict[str, str]] = []

        if provider_name == "local" or provider_name is None:
            # Default case - no additional environment variables needed
            pass
        elif provider_name == "s3":
            env_vars.append({"name": "ENABLE_S3", "value": "s3"})
            env_vars.append({"name": "S3_BUCKET_NAME", "value": request.getfixturevalue(argname="ci_s3_bucket_name")})
            env_vars.append({
                "name": "AWS_DEFAULT_REGION",
                "value": request.getfixturevalue(argname="ci_s3_bucket_region"),
            })
            env_vars.append({
                "name": "S3_ENDPOINT_URL",
                "value": request.getfixturevalue(argname="ci_s3_bucket_endpoint"),
            })
            env_vars.append({
                "name": "AWS_ACCESS_KEY_ID",
                "value": request.getfixturevalue(argname="aws_access_key_id"),
            })
            env_vars.append({
                "name": "AWS_SECRET_ACCESS_KEY",
                "value": request.getfixturevalue(argname="aws_secret_access_key"),
            })
            env_vars.append({"name": "S3_AUTO_CREATE_BUCKET", "value": S3_AUTO_CREATE_BUCKET})

        else:
            raise ValueError(f"Unsupported files provider: {provider_name}")

        return env_vars

    return _factory
