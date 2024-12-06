from typing import Dict

from tests.model_serving.model_server.utils import b64_encoded_string


def get_s3_secret_dict(
    aws_access_key: str,
    aws_secret_access_key: str,
    aws_s3_bucket: str,
    aws_s3_endpoint: str,
    aws_s3_region: str,
) -> Dict[str, str]:
    return {
        "AWS_ACCESS_KEY_ID": b64_encoded_string(string_to_encode=aws_access_key),
        "AWS_SECRET_ACCESS_KEY": b64_encoded_string(string_to_encode=aws_secret_access_key),
        "AWS_S3_BUCKET": b64_encoded_string(string_to_encode=aws_s3_bucket),
        "AWS_S3_ENDPOINT": b64_encoded_string(string_to_encode=aws_s3_endpoint),
        "AWS_DEFAULT_REGION": b64_encoded_string(string_to_encode=aws_s3_region),
    }
