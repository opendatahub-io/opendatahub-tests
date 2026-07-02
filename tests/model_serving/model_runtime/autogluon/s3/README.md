# AutoGluon S3 Model Tests

End-to-end tests for AutoGluon inference using S3-backed model storage and KServe `InferenceService` with a namespace-scoped `ServingRuntime` created from in-repo spec.

Main suite overview: [`../README.md`](../README.md)

## Predictor Variants

| Param id | Protocol | S3 prefix constant |
| -------- | -------- | ------------------ |
| `tabular-v2-s3-Standard` | V2 | `S3_PREFIX_TABULAR_V2` |
| `tabular-v1-s3-Standard` | V1 | `S3_PREFIX_TABULAR_V1` |
| `timeseries-v1-s3-Standard` | V1 | `S3_PREFIX_TIMESERIES_V1` |

Replace `TODO/...` values in [`../constant.py`](../constant.py) before running against real models in MinIO.

## Prerequisites

Valid S3 configuration (via `valid_aws_config` fixture):

- S3 bucket name, region, endpoint
- AWS access key ID and secret access key
- Cluster with KServe and AutoGluon `ClusterServingRuntime` (or `AUTOGLUON_RUNTIME_IMAGE`)

## Minimal Required Inputs

Environment variables:

```bash
export AWS_ACCESS_KEY_ID=<aws-access-key-id>
export AWS_SECRET_ACCESS_KEY=<aws-secret-access-key>
export MODELS_S3_BUCKET_NAME=<bucket-name>
export MODELS_S3_BUCKET_REGION=<region>
export MODELS_S3_BUCKET_ENDPOINT=<endpoint-url>
export CI_S3_BUCKET_ENDPOINT=<endpoint-url> # use the same value as MODELS_S3_BUCKET_ENDPOINT
export AUTOGLUON_RUNTIME_IMAGE=<image@digest> # optional override
export OC_BINARY_PATH=/usr/local/bin/oc # optional if `oc` is already on PATH
```

Equivalent CLI flags: `--aws-access-key-id`, `--aws-secret-access-key`, `--models-s3-bucket-name`, `--models-s3-bucket-region`, `--models-s3-bucket-endpoint`, `--ci-s3-bucket-endpoint`.

If you keep variables in a local `.env` file, load it before running tests:

```bash
set -a
source .env
set +a
```

## Running Tests

```bash
OC_BINARY_PATH=/usr/local/bin/oc uv run pytest tests/model_serving/model_runtime/autogluon/s3 \
  --aws-access-key-id=<aws-access-key-id> \
  --aws-secret-access-key=<aws-secret-access-key> \
  --models-s3-bucket-name=<bucket-name> \
  --models-s3-bucket-region=<region> \
  --models-s3-bucket-endpoint=<endpoint-url>
```

Run one variant:

```bash
OC_BINARY_PATH=/usr/local/bin/oc uv run pytest tests/model_serving/model_runtime/autogluon/s3 -k "tabular-v2" \
  --aws-access-key-id=... --aws-secret-access-key=... \
  --models-s3-bucket-name=... --models-s3-bucket-region=... --models-s3-bucket-endpoint=...
```

## Troubleshooting

- **Skip: AutoGluon runtime image not found** — install RHOAI with autogluonserver or set `AUTOGLUON_RUNTIME_IMAGE`.
- **Model load failures** — confirm `S3_PREFIX_*` paths exist in the bucket and contain the expected `.pkl` layout.
- **Fuzzy validation / HTTP errors** — verify V1/V2 paths and payloads in `constant.py` match autogluonserver API.

## Security Note

Do not commit credentials. Use env vars or CI secret injection.
