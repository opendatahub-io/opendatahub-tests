# Job identification
ASYNC_UPLOAD_JOB_NAME = "model-sync-async-job"

# Placeholder for downstream image (TBD)
ASYNC_UPLOAD_IMAGE = "PLACEHOLDER_DOWNSTREAM_ASYNC_UPLOAD_IMAGE"

# Job labels and annotations
ASYNC_JOB_LABELS = {
    "app.kubernetes.io/name": "model-registry-async-job",
    "app.kubernetes.io/component": "async-job",
    "app.kubernetes.io/part-of": "model-registry",
    "app.kubernetes.io/managed-by": "kubectl",
    "component": "model-registry-job",
    "modelregistry.opendatahub.io/job-type": "async-upload",
}

ASYNC_JOB_ANNOTATIONS = {
    "modelregistry.opendatahub.io/description": "Asynchronous job for uploading models to Model Registry"
}

# Model sync parameters (from sample YAML)
MODEL_SYNC_CONFIG = {
    "MODEL_ID": "1",
    "MODEL_VERSION_ID": "3",
    "MODEL_ARTIFACT_ID": "6",
    "SOURCE_TYPE": "s3",
    "DESTINATION_TYPE": "oci",
    "SOURCE_AWS_KEY": "my-model",
    "DESTINATION_OCI_URI": "PLACEHOLDER_OCI_URI",
    "DESTINATION_OCI_BASE_IMAGE": "busybox:latest",
    "DESTINATION_OCI_ENABLE_TLS_VERIFY": "false",
}

# Placeholder secret names (to be provided by other engineer)
PLACEHOLDER_S3_SECRET_NAME = "PLACEHOLDER_S3_CREDENTIALS_SECRET"  # pragma: allowlist secret
PLACEHOLDER_OCI_SECRET_NAME = "PLACEHOLDER_OCI_CREDENTIALS_SECRET"  # pragma: allowlist secret

# Volume mount paths (from sample YAML)
VOLUME_MOUNTS = {
    "SOURCE_CREDS_PATH": "/opt/creds/source",
    "DEST_CREDS_PATH": "/opt/creds/destination",
    "DEST_DOCKERCONFIG_PATH": "/opt/creds/destination/.dockerconfigjson",
}
