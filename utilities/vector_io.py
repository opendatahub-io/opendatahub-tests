"""Shared templates and credentials for vector I/O test deployments."""

import os
import secrets
from typing import Any

from utilities.image_constants import SharedImages

MILVUS_IMAGE = os.getenv("OGX_VECTOR_IO_MILVUS_IMAGE", SharedImages.MILVUS)
ETCD_IMAGE = os.getenv("OGX_VECTOR_IO_ETCD_IMAGE", SharedImages.ETCD)
MILVUS_TOKEN = os.getenv("OGX_VECTOR_IO_MILVUS_TOKEN", secrets.token_urlsafe(32))


def get_milvus_deployment_template() -> dict[str, Any]:
    """Return the Kubernetes deployment template for standalone Milvus."""
    return {
        "metadata": {"labels": {"app": "milvus-standalone"}},
        "spec": {
            "containers": [
                {
                    "name": "milvus-standalone",
                    "image": MILVUS_IMAGE,
                    "args": ["milvus", "run", "standalone"],
                    "ports": [{"containerPort": 19530, "protocol": "TCP"}],
                    "volumeMounts": [{"name": "milvus-data", "mountPath": "/var/lib/milvus"}],
                    "env": [
                        {"name": "DEPLOY_MODE", "value": "standalone"},
                        {"name": "ETCD_ENDPOINTS", "value": "vector-io-etcd-service:2379"},
                        {"name": "MINIO_ADDRESS", "value": ""},
                        {"name": "COMMON_STORAGETYPE", "value": "local"},
                    ],
                }
            ],
            "volumes": [{"name": "milvus-data", "emptyDir": {}}],
        },
    }


def get_etcd_deployment_template() -> dict[str, Any]:
    """Return the Kubernetes deployment template for etcd."""
    return {
        "metadata": {"labels": {"app": "etcd"}},
        "spec": {
            "containers": [
                {
                    "name": "etcd",
                    "image": ETCD_IMAGE,
                    "command": [
                        "etcd",
                        "--advertise-client-urls=http://vector-io-etcd-service:2379",
                        "--listen-client-urls=http://0.0.0.0:2379",
                        "--data-dir=/etcd",
                    ],
                    "ports": [{"containerPort": 2379}],
                    "volumeMounts": [{"name": "etcd-data", "mountPath": "/etcd"}],
                    "env": [
                        {"name": "ETCD_AUTO_COMPACTION_MODE", "value": "revision"},
                        {"name": "ETCD_AUTO_COMPACTION_RETENTION", "value": "1000"},
                        {"name": "ETCD_QUOTA_BACKEND_BYTES", "value": "4294967296"},
                        {"name": "ETCD_SNAPSHOT_COUNT", "value": "50000"},
                    ],
                }
            ],
            "volumes": [{"name": "etcd-data", "emptyDir": {}}],
        },
    }
