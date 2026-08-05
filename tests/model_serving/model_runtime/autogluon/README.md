# AutoGluon KServe Runtime Test Suite

End-to-end tests for AutoGluon models served through the dedicated `autogluonserver` KServe runtime (not MLServer).

## Sub-suites

- S3 model source: [`s3/README.md`](./s3/README.md)

## What This Suite Covers

- Protocols: KServe V1 (tabular, timeseries) and V2 (tabular)
- Deployment mode: `Standard`
- Model source: external S3/MinIO
- Runtime image: resolved from `ClusterServingRuntime` `kserve-autogluonserver`, RHOAI CSV `relatedImages`, or `AUTOGLUON_RUNTIME_IMAGE` override
- Response validation: fuzzy structural checks for deterministic inference responses

## Before First Cluster Run

1. Replace `TODO/...` prefixes in [`constant.py`](./constant.py) (`S3_PREFIX_*`) with real MinIO/S3 paths.
2. Align `TABULAR_*_INPUT` / `TIMESERIES_V1_INPUT` payloads and inference path constants with the autogluonserver API.
3. Run tests and verify deterministic responses with fuzzy checks (no snapshot update step required).

## Run All AutoGluon S3 Tests

```bash
OC_BINARY_PATH=/usr/local/bin/oc uv run pytest tests/model_serving/model_runtime/autogluon/s3 \
  --aws-access-key-id=<aws-access-key-id> \
  --aws-secret-access-key=<aws-secret-access-key> \
  --models-s3-bucket-name=<bucket-name> \
  --models-s3-bucket-region=<region> \
  --models-s3-bucket-endpoint=<endpoint-url>
```

Optional runtime image override:

```bash
export AUTOGLUON_RUNTIME_IMAGE=<image@digest>
```
