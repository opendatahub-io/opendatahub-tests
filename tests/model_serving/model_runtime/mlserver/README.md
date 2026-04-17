# MLServer Runtime Test Suite

End-to-end tests for MLServer model serving on OpenShift AI / OpenDataHub.
This suite validates inference behavior across different model source strategies and formats.

## Sub-suites

- S3 model source: [`s3/README.md`](./s3/README.md)
- OCI model car source: [`model_car/README.md`](./model_car/README.md)

## What This Suite Covers

- Protocol: REST inference
- Deployment mode: `RawDeployment`
- Model formats: sklearn, xgboost, lightgbm, onnx
- Response validation: snapshot-based assertions

## Run All MLServer Tests

```bash
OC_BINARY_PATH=/usr/local/bin/oc uv run pytest tests/model_serving/model_runtime/mlserver \
  --aws-access-key-id=<aws-access-key-id> \
  --aws-secret-access-key=<aws-secret-access-key> \
  --models-s3-bucket-name=<bucket-name> \
  --models-s3-bucket-region=<region> \
  --models-s3-bucket-endpoint=<endpoint-url>
```

Optional runtime image override (use only for custom/private image validation):

```bash
--mlserver-runtime-image=<mlserver-image>
```

## Snapshot Updates

```bash
OC_BINARY_PATH=/usr/local/bin/oc uv run pytest tests/model_serving/model_runtime/mlserver \
  --snapshot-update \
  --aws-access-key-id=<aws-access-key-id> \
  --aws-secret-access-key=<aws-secret-access-key> \
  --models-s3-bucket-name=<bucket-name> \
  --models-s3-bucket-region=<region> \
  --models-s3-bucket-endpoint=<endpoint-url>
```
