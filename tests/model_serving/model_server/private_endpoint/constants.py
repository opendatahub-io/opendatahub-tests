AWS_REGION: str = "us-east-2"
AWS_BUCKET: str = "ods-ci-wisdom"
AWS_ENDPOINT: str = "https://s3.us-east-2.amazonaws.com/"

SR_CONTAINERS_KSERVE_CAIKIT: list = [
            {
                "args": ["--model-name=/mnt/models/artifacts/"],
                "command": ["text-generation-launcher"],
                "env": [{"name": "HF_HOME", "value": "/tmp/hf_home"}],
                "image": "quay.io/modh/text-generation-inference@sha256:294f07b2a94a223a18e559d497a79cac53bf7893f36cfc6c995475b6e431bcfe",
                "name": "kserve-container",
                "volumeMounts": [{"mountPath": "/dev/shm", "name": "shm"}],
            },
            {
                "env": [
                    {"name": "RUNTIME_LOCAL_MODELS_DIR", "value": "/mnt/models"},
                    {"name": "HF_HOME", "value": "/tmp/hf_home"},
                    {"name": "RUNTIME_GRPC_ENABLED", "value": "false"},
                    {"name": "RUNTIME_HTTP_ENABLED", "value": "true"},
                ],
                "image": "quay.io/modh/caikit-tgis-serving@sha256:4e907ce35a3767f5be2f3175a1854e8d4456c43b78cf3df4305bceabcbf0d6e2",
                "name": "transformer-container",
                "ports": [{"containerPort": 8080, "protocol": "TCP"}],
                "volumeMounts": [{"mountPath": "/dev/shm", "name": "shm"}],
            },
    ]
SR_SUPPORTED_FORMATS_CAIKIT: list = [{"autoSelect": True, "name": "caikit"},]
SR_VOLUMES: list = [{"emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}, "name": "shm"}]
SR_ANNOTATIONS: dict = {"prometheus.io/path": "/metrics", "prometheus.io/port": "3000"}