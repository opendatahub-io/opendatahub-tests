import base64

MINIO: str = "minio"

MINIO_DATA_DICT: dict[str, str] = {
    "AWS_ACCESS_KEY_ID": base64.b64encode("THEACCESSKEY".encode()).decode(),
    "AWS_DEFAULT_REGION": base64.b64encode("us-south".encode()).decode(),
    "AWS_S3_BUCKET": base64.b64encode("modelmesh-example-models".encode()).decode(),
    "AWS_S3_ENDPOINT": base64.b64encode("http://minio:9000".encode()).decode(),
    "AWS_SECRET_ACCESS_KEY": base64.b64encode("THESECRETKEY".encode()).decode(),  # pragma: allowlist secret
}
