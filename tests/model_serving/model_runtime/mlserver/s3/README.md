# MLServer S3 Model Tests

End-to-end tests for MLServer inference using S3-backed model storage.
Models are mounted from S3 and served through KServe `InferenceService` with MLServer runtime.

Main suite overview:
[`mlserver/README.md`](../README.md)

## Supported Model Formats

- sklearn
- xgboost
- lightgbm
- onnx

## Test Coverage

This suite validates:

- Protocol: REST
- Deployment mode: `RawDeployment`
- Model source: S3 storage
- Validation strategy: JSON snapshot comparison

## Prerequisites

These tests require valid S3 configuration (consumed by the `valid_aws_config` fixture), including:

- S3 bucket name
- S3 bucket region
- S3 endpoint URL
- AWS access key ID
- AWS secret access key

The test fixtures use AWS credentials and S3 settings to create/use the KServe S3 secret for model access.
See `Minimal Required Inputs` for the supported flags and environment variables.

## Minimal Required Inputs

You can pass configuration by flags or environment variables:

```bash
export AWS_ACCESS_KEY_ID=<aws-access-key-id>
export AWS_SECRET_ACCESS_KEY=<aws-secret-access-key>
export MODELS_S3_BUCKET_NAME=<bucket-name>
export MODELS_S3_BUCKET_REGION=<region>
export MODELS_S3_BUCKET_ENDPOINT=<endpoint-url>
```

Equivalent CLI flags:

- `--aws-access-key-id`
- `--aws-secret-access-key`
- `--models-s3-bucket-name`
- `--models-s3-bucket-region`
- `--models-s3-bucket-endpoint`

## Running Tests

Run all MLServer S3 tests:

```bash
OC_BINARY_PATH=/usr/local/bin/oc uv run pytest tests/model_serving/model_runtime/mlserver/s3 \
  --aws-access-key-id=<aws-access-key-id> \
  --aws-secret-access-key=<aws-secret-access-key> \
  --models-s3-bucket-name=<bucket-name> \
  --models-s3-bucket-region=<region> \
  --models-s3-bucket-endpoint=<endpoint-url>
```

Run a specific model format (example: onnx):

```bash
OC_BINARY_PATH=/usr/local/bin/oc uv run pytest tests/model_serving/model_runtime/mlserver/s3 -k "onnx" \
  --aws-access-key-id=<aws-access-key-id> \
  --aws-secret-access-key=<aws-secret-access-key> \
  --models-s3-bucket-name=<bucket-name> \
  --models-s3-bucket-region=<region> \
  --models-s3-bucket-endpoint=<endpoint-url>
```

## Updating Snapshots

If expected model responses change, update snapshots with:

```bash
OC_BINARY_PATH=/usr/local/bin/oc uv run pytest tests/model_serving/model_runtime/mlserver/s3 \
  --snapshot-update \
  --aws-access-key-id=<aws-access-key-id> \
  --aws-secret-access-key=<aws-secret-access-key> \
  --models-s3-bucket-name=<bucket-name> \
  --models-s3-bucket-region=<region> \
  --models-s3-bucket-endpoint=<endpoint-url>
```

## Troubleshooting

- `AWS access key id is not set` / `AWS secret access key is not set`:
  pass `--aws-access-key-id` and `--aws-secret-access-key`, or export the corresponding env vars.
- S3 bucket value errors:
  verify `--models-s3-bucket-name`, `--models-s3-bucket-region`, and `--models-s3-bucket-endpoint`.
- `No pods found for InferenceService ...`:
  check runtime and InferenceService readiness in the target namespace before rerunning.

## Security Note

- Do not hardcode or commit credentials in scripts, docs, or command history files.
- Prefer environment variables or CI-managed secret injection.

## Related Suite

For OCI/model car based MLServer tests, see:
[`mlserver/model_car/README.md`](../model_car/README.md)
