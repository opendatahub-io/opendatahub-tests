from utilities.general import get_s3_secret_dict

MINIO: str = "minio"
MINIO_PORT: int = 9000

MINIO_DATA_DICT: dict[str, str] = get_s3_secret_dict(
    aws_access_key="THEACCESSKEY",
    aws_secret_access_key="THESECRETKEY",  # pragma: allowlist secret
    aws_s3_bucket="modelmesh-example-models",
    aws_s3_endpoint=f"http://minio:{str(MINIO_PORT)}",
    aws_s3_region="us-south",
)
