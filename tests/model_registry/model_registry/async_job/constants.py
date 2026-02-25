# Job identification
ASYNC_UPLOAD_JOB_NAME = "model-sync-async-job"

# Job labels and annotations
ASYNC_JOB_LABELS = {
    "app.kubernetes.io/name": "model-registry-async-job",
    "app.kubernetes.io/component": "async-job",
    "app.kubernetes.io/part-of": "model-registry",
    "component": "model-registry-job",
    "modelregistry.opendatahub.io/job-type": "async-upload",
}
ASYNC_STR: str = "Asynchronous job for uploading models to Model Registry and converting them to ModelCar format"
ASYNC_JOB_ANNOTATIONS = {"modelregistry.opendatahub.io/description": ASYNC_STR}

# Model sync parameters (from sample YAML)
MODEL_SYNC_CONFIG = {
    "MODEL_ID": "1",
    "MODEL_VERSION_ID": "2",
    "MODEL_ARTIFACT_ID": "1",
    "SOURCE_TYPE": "s3",
    "DESTINATION_TYPE": "oci",
    "SOURCE_AWS_KEY": "my-model",
    "DESTINATION_OCI_BASE_IMAGE": "public.ecr.aws/docker/library/busybox:latest",
    "DESTINATION_OCI_ENABLE_TLS_VERIFY": "false",
}

# Volume mount paths (from sample YAML)
VOLUME_MOUNTS = {
    "SOURCE_CREDS_PATH": "/opt/creds/source",
    "DEST_CREDS_PATH": "/opt/creds/destination",
    "DEST_DOCKERCONFIG_PATH": "/opt/creds/destination/.dockerconfigjson",
}

REPO_NAME = "async-job-test/model-artifact"
TAG = "latest"
