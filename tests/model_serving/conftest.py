import pytest
from _pytest.fixtures import FixtureRequest


@pytest.fixture(scope="session")
def s3_models_storage_uri(request: FixtureRequest, models_s3_bucket_name: str) -> str:
    return f"s3://{models_s3_bucket_name}/{request.param['model-dir']}/"
