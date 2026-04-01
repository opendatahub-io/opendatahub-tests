"""Constants for Model Registry Python Client Signing Tests."""

import time

from tests.model_registry.constants import MODEL_DICT
from tests.model_registry.model_registry.async_job.constants import MODEL_SYNC_CONFIG

# Securesign instance configuration
SECURESIGN_NAMESPACE = "trusted-artifact-signer"
SECURESIGN_NAME = "securesign-sample"
SECURESIGN_API_VERSION = "rhtas.redhat.com/v1alpha1"
SECURESIGN_ORGANIZATION_NAME = "RHOAI"
SECURESIGN_ORGANIZATION_EMAIL = "admin@example.com"

# TAS Connection Type ConfigMap name
TAS_CONNECTION_TYPE_NAME = "tas-securesign-v1"

# OCI Registry configuration for signed model storage
SIGNING_OCI_REPO_NAME = "signing-test/signed-model"
SIGNING_OCI_TAG = "latest"
SIGNING_ASYNC_REPO = "async-signing-test/signed-model"
SIGNING_ASYNC_TAG = "latest"
NATIVE_SIGNING_REPO = "native-signing-test/signed-model"
NATIVE_SIGNING_TAG = "latest"
MODEL_CONTENT = b"test model content for async signing pipeline validation"
IDENTITY_TOKEN_MOUNT_PATH = "/var/run/secrets/signing"
MINIO_MC_IMAGE = "quay.io/minio/mc@sha256:470f5546b596e16c7816b9c3fa7a78ce4076bb73c2c73f7faeec0c8043923123"
MINIO_UPLOADER_SECURITY_CONTEXT = {
    "allowPrivilegeEscalation": False,
    "capabilities": {"drop": ["ALL"]},
    "runAsNonRoot": True,
    "seccompProfile": {"type": "RuntimeDefault"},
}
SIGNING_MODEL_DATA = {
    **MODEL_DICT,
    "model_name": f"signing-model-{int(time.time())}",
    "model_storage_key": MODEL_SYNC_CONFIG["SOURCE_AWS_KEY"],
    "model_storage_path": "path/to/test/model",
}
