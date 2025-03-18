from utilities.general import get_s3_secret_dict

MINIO: str = "minio"
MINIO_PORT: int = 9000

MINIO_ACCESS_KEY: str = "MINIO_ACCESS_KEY"
MINIO_ACCESS_KEY_VALUE: str = "THEACCESSKEY"
MINIO_SECRET_KEY: str = "MINIO_SECRET_KEY"
MINIO_SECRET_KEY_VALUE: str = "THESECRETKEY"

MINIO_DATA_DICT: dict[str, str] = get_s3_secret_dict(
    aws_access_key=MINIO_ACCESS_KEY_VALUE,
    aws_secret_access_key=MINIO_SECRET_KEY_VALUE,  # pragma: allowlist secret
    aws_s3_bucket="modelmesh-example-models",
    aws_s3_endpoint=f"http://minio:{str(MINIO_PORT)}",
    aws_s3_region="us-south",
)
