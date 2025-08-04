from typing import Any, Generator

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job

from tests.model_registry.async_job.constants import (
    ASYNC_JOB_ANNOTATIONS,
    ASYNC_JOB_LABELS,
    ASYNC_UPLOAD_IMAGE,
    ASYNC_UPLOAD_JOB_NAME,
    MODEL_SYNC_CONFIG,
    PLACEHOLDER_OCI_SECRET_NAME,
    PLACEHOLDER_S3_SECRET_NAME,
    VOLUME_MOUNTS,
)


@pytest.fixture(scope="class")
def placeholder_s3_secret_name() -> str:
    """Placeholder for S3 credentials secret name - to be replaced by other engineer's fixture"""
    return PLACEHOLDER_S3_SECRET_NAME


@pytest.fixture(scope="class")
def placeholder_oci_secret_name() -> str:
    """Placeholder for OCI credentials secret name - to be replaced by other engineer's fixture"""
    return PLACEHOLDER_OCI_SECRET_NAME


@pytest.fixture(scope="class")
def model_sync_async_job(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    placeholder_s3_secret_name: str,
    placeholder_oci_secret_name: str,
    teardown_resources: bool,
) -> Generator[Job, Any, Any]:
    """Core Job fixture focused on Job deployment and configuration"""
    with Job(
        client=admin_client,
        name=ASYNC_UPLOAD_JOB_NAME,
        namespace=model_registry_namespace,
        labels=ASYNC_JOB_LABELS,
        annotations=ASYNC_JOB_ANNOTATIONS,
        spec={
            "template": {
                "metadata": {
                    "labels": {
                        **ASYNC_JOB_LABELS,
                        # Template pod labels from sample YAML
                        "modelregistry.opendatahub.io/model-sync-model-id": MODEL_SYNC_CONFIG["MODEL_ID"],
                        "modelregistry.opendatahub.io/model-sync-model-version-id": MODEL_SYNC_CONFIG[
                            "MODEL_VERSION_ID"
                        ],
                        "modelregistry.opendatahub.io/model-sync-model-artifact-id": MODEL_SYNC_CONFIG[
                            "MODEL_ARTIFACT_ID"
                        ],
                    }
                },
                "spec": {
                    "restartPolicy": "Never",
                    "volumes": [
                        {"name": "source-credentials", "secret": {"secretName": placeholder_s3_secret_name}},
                        {
                            "name": "destination-credentials",
                            "secret": {
                                "secretName": placeholder_oci_secret_name,
                                "items": [{"key": ".dockerconfigjson", "path": ".dockerconfigjson"}],
                            },
                        },
                    ],
                    "containers": [
                        {
                            "name": "async-upload",
                            "image": ASYNC_UPLOAD_IMAGE,
                            "volumeMounts": [
                                {
                                    "name": "source-credentials",
                                    "readOnly": True,
                                    "mountPath": VOLUME_MOUNTS["SOURCE_CREDS_PATH"],
                                },
                                {
                                    "name": "destination-credentials",
                                    "readOnly": True,
                                    "mountPath": VOLUME_MOUNTS["DEST_CREDS_PATH"],
                                },
                            ],
                            "env": [
                                # Proxy settings
                                {"name": "HTTP_PROXY", "value": ""},
                                {"name": "HTTPS_PROXY", "value": ""},
                                {"name": "NO_PROXY", "value": "*.svc.cluster.local"},
                                # Source configuration
                                {"name": "MODEL_SYNC_SOURCE_TYPE", "value": MODEL_SYNC_CONFIG["SOURCE_TYPE"]},
                                {"name": "MODEL_SYNC_SOURCE_AWS_KEY", "value": MODEL_SYNC_CONFIG["SOURCE_AWS_KEY"]},
                                {
                                    "name": "MODEL_SYNC_SOURCE_S3_CREDENTIALS_PATH",
                                    "value": VOLUME_MOUNTS["SOURCE_CREDS_PATH"],
                                },
                                # Destination configuration
                                {"name": "MODEL_SYNC_DESTINATION_TYPE", "value": MODEL_SYNC_CONFIG["DESTINATION_TYPE"]},
                                {
                                    "name": "MODEL_SYNC_DESTINATION_OCI_URI",
                                    "value": MODEL_SYNC_CONFIG["DESTINATION_OCI_URI"],
                                },
                                {
                                    "name": "MODEL_SYNC_DESTINATION_OCI_CREDENTIALS_PATH",
                                    "value": VOLUME_MOUNTS["DEST_DOCKERCONFIG_PATH"],
                                },
                                {
                                    "name": "MODEL_SYNC_DESTINATION_OCI_BASE_IMAGE",
                                    "value": MODEL_SYNC_CONFIG["DESTINATION_OCI_BASE_IMAGE"],
                                },
                                {
                                    "name": "MODEL_SYNC_DESTINATION_OCI_ENABLE_TLS_VERIFY",
                                    "value": MODEL_SYNC_CONFIG["DESTINATION_OCI_ENABLE_TLS_VERIFY"],
                                },
                                # Model parameters
                                {"name": "MODEL_SYNC_MODEL_ID", "value": MODEL_SYNC_CONFIG["MODEL_ID"]},
                                {"name": "MODEL_SYNC_MODEL_VERSION_ID", "value": MODEL_SYNC_CONFIG["MODEL_VERSION_ID"]},
                                {
                                    "name": "MODEL_SYNC_MODEL_ARTIFACT_ID",
                                    "value": MODEL_SYNC_CONFIG["MODEL_ARTIFACT_ID"],
                                },
                                # Model Registry client params (placeholders)
                                {
                                    "name": "MODEL_SYNC_REGISTRY_SERVER_ADDRESS",
                                    "value": "PLACEHOLDER_MR_SERVER_ADDRESS",
                                },
                                {"name": "MODEL_SYNC_REGISTRY_PORT", "value": "8080"},
                                {"name": "MODEL_SYNC_REGISTRY_AUTHOR", "value": "RHOAI Test"},
                                {"name": "MODEL_SYNC_REGISTRY_USER_TOKEN", "value": "PLACEHOLDER_USER_TOKEN"},
                            ],
                        }
                    ],
                },
            }
        },
        teardown=teardown_resources,
    ) as job:
        yield job
