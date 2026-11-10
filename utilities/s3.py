import base64


def get_s3_secret_dict(
    aws_access_key: str,
    aws_secret_access_key: str,
    aws_s3_bucket: str,
    aws_s3_endpoint: str,
    aws_s3_region: str | None = None,
    aws_default_region: str | None = None,
) -> dict[str, str]:
    """Return base64-encoded S3 connection data for an OpenShift Secret."""
    region = aws_default_region or aws_s3_region or "us-east-1"
    return {
        "AWS_ACCESS_KEY_ID": b64_encoded_string(aws_access_key),
        "AWS_SECRET_ACCESS_KEY": b64_encoded_string(aws_secret_access_key),
        "AWS_S3_BUCKET": b64_encoded_string(aws_s3_bucket),
        "AWS_S3_ENDPOINT": b64_encoded_string(aws_s3_endpoint),
        "AWS_DEFAULT_REGION": b64_encoded_string(region),
    }


def b64_encoded_string(string_to_encode: str) -> str:
    """Return a base64-encoded string suitable for an OpenShift Secret."""
    return base64.b64encode(string_to_encode.encode()).decode()
