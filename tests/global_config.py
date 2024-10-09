import os
from typing import Optional

global config  # type: ignore[unused-ignore]


# AWS credentials
aws_secret_access_key: Optional[str] = os.environ.get("AWS_SECRET_ACCESS_KEY", "aws_secret_key")
aws_access_key_id: Optional[str] = os.environ.get("AWS_ACCESS_KEY_ID", "aws_access_key")

# S3 buckets
ci_s3_bucket_name: str = "ci-bucket"

for _dir in dir():
    val = locals()[_dir]
    if type(val) not in [bool, list, dict, str, int]:
        continue

    if _dir in ["encoding", "py_file"]:
        continue

    config[_dir] = locals()[_dir]  # type: ignore[name-defined] # noqa: F821
