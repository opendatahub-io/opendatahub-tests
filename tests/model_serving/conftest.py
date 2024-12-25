import pytest


@pytest.fixture(scope="session")
def s3_models_storage_uri(request, models_s3_bucket_name) -> str:
    return f"s3://{models_s3_bucket_name}/{request.param['model-dir']}/"
