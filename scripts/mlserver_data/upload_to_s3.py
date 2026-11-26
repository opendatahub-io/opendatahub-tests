#!/usr/bin/env python3
r"""Upload generated MLServer model artifacts to an S3 bucket.

Reads AWS credentials from the environment (AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, AWS_ENDPOINT_URL).

Usage:
    python scripts/mlserver_data/upload_to_s3.py \
        --source-dir /tmp/mlserver_models \
        --bucket my-models-bucket \
        --prefix mlserver

This uploads each scenario directory under the specified prefix:
    s3://my-models-bucket/mlserver/model_repository/
    s3://my-models-bucket/mlserver/multi_version/
    s3://my-models-bucket/mlserver/one_failed_model/
    s3://my-models-bucket/mlserver/malformed_settings/
    s3://my-models-bucket/mlserver/unsupported_impl/
    s3://my-models-bucket/mlserver/missing_artifact/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import boto3


def upload(source_dir: Path, bucket: str, prefix: str, endpoint_url: str | None) -> None:
    session = boto3.Session(
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
    client_kwargs: dict[str, str] = {}
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url
    s3 = session.client(service_name="s3", **client_kwargs)

    uploaded = 0
    for path in sorted(source_dir.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(source_dir)
        key = f"{prefix}/{relative}" if prefix else str(relative)
        s3.upload_file(Filename=str(path), Bucket=bucket, Key=key)
        print(f"  {key}")
        uploaded += 1

    print(f"\nUploaded {uploaded} files to s3://{bucket}/{prefix}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source-dir", type=Path, required=True, help="Local directory produced by generate_artifacts.py."
    )
    parser.add_argument("--bucket", required=True, help="S3 bucket name.")
    parser.add_argument("--prefix", default="mlserver", help="S3 key prefix (default: mlserver).")
    parser.add_argument(
        "--endpoint-url",
        default=os.environ.get("AWS_ENDPOINT_URL"),
        help="S3 endpoint URL (default: AWS_ENDPOINT_URL env var).",
    )
    args = parser.parse_args()

    if not args.source_dir.is_dir():
        parser.error(f"Source directory does not exist: {args.source_dir}")

    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        if var not in os.environ:
            parser.error(f"Required environment variable {var} is not set.")

    print(f"Uploading {args.source_dir} -> s3://{args.bucket}/{args.prefix}/")
    upload(source_dir=args.source_dir, bucket=args.bucket, prefix=args.prefix, endpoint_url=args.endpoint_url)


if __name__ == "__main__":
    main()
