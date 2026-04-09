# MLServer Model Car (OCI Image) Tests

End-to-end tests for MLServer inference using model car (OCI image-based) deployments.
Models are packaged as OCI container images and deployed via KServe InferenceService with `storageUri: oci://...`.

Main suite overview:
[`mlserver/README.md`](../README.md)

## Supported Model Formats

- sklearn
- xgboost
- lightgbm
- onnx

## OCI Model Images

The OCI model images used in these tests are built from:
<https://github.com/Jooho/oci-model-images>

If the version of a supported framework (xgboost, lightgbm, sklearn, onnx) changes in MLServer,
the model images must be rebuilt and pushed from that repository.

The framework versions used by MLServer can be found at:
<https://github.com/red-hat-data-services/MLServer/blob/main/requirements/requirements-cpu.txt#L261>

For e2e testing, images should be tagged with the `-e2e` suffix to pin stable versions.

## Running Tests

Run all model car tests:

```bash
OC_BINARY_PATH=/usr/local/bin/oc uv run pytest tests/model_serving/model_runtime/mlserver/model_car
```

Run a specific model format (e.g., onnx only):

```bash
OC_BINARY_PATH=/usr/local/bin/oc uv run pytest tests/model_serving/model_runtime/mlserver/model_car -k "onnx"
```

## Updating Snapshots

If model responses change and snapshots need to be updated, add the `--snapshot-update` flag:

```bash
OC_BINARY_PATH=/usr/local/bin/oc uv run pytest tests/model_serving/model_runtime/mlserver/model_car --snapshot-update
```

## Related Suite

For S3-based MLServer tests, see:
[`mlserver/s3/README.md`](../s3/README.md)
