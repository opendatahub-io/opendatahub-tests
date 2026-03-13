from typing import Any

SERVING_RUNTIME_TEMPLATE_NAME: str = "kserve-ovms-serving-runtime-template"
SERVING_RUNTIME_INSTANCE_NAME: str = "kserve-ovms-serving-runtime-instance"

OVMS_SERVING_RUNTIME_IMAGE: str = (
    "quay.io/modh/openvino_model_server@sha256:53b7fcf95de9b81e4c8652d0bf4e84e22d5b696827a5d951d863420c68b9cfe8"
)

OVMS_TEMPLATE_LABELS: dict[str, str] = {
    "opendatahub.io/dashboard": "true",
    "opendatahub.io/ootb": "true",
}

OVMS_TEMPLATE_ANNOTATIONS: dict[str, str] = {
    "tags": "kserve-ovms,servingruntime",
    "description": "OpenVino Model Serving Definition",
    "opendatahub.io/modelServingSupport": '["single"]',
    "opendatahub.io/apiProtocol": "REST",
}

OVMS_RUNTIME_LABELS: dict[str, str] = {
    "opendatahub.io/dashboard": "true",
}

OVMS_RUNTIME_ANNOTATIONS: dict[str, str] = {
    "openshift.io/display-name": "OpenVINO Model Server",
    "opendatahub.io/recommended-accelerators": '["nvidia.com/gpu"]',
    "opendatahub.io/runtime-version": "v2025.1",
}

OVMS_RUNTIME_PROMETHEUS_ANNOTATIONS: dict[str, str] = {
    "prometheus.io/port": "8888",
    "prometheus.io/path": "/metrics",
}

OVMS_SUPPORTED_MODEL_FORMATS: list[dict[str, str | bool]] = [
    {"name": "openvino_ir", "version": "opset13", "autoSelect": True},
    {"name": "onnx", "version": "1"},
    {"name": "tensorflow", "version": "1", "autoSelect": True},
    {"name": "tensorflow", "version": "2", "autoSelect": True},
    {"name": "paddle", "version": "2", "autoSelect": True},
    {"name": "pytorch", "version": "2", "autoSelect": True},
]

OVMS_CONTAINER_ARGS: list[str] = [
    "--model_name={{.Name}}",
    "--port=8001",
    "--rest_port=8888",
    "--model_path=/mnt/models",
    "--file_system_poll_wait_seconds=0",
    "--grpc_bind_address=0.0.0.0",
    "--rest_bind_address=0.0.0.0",
    "--target_device=AUTO",
    "--metrics_enable",
]

OVMS_SERVING_RUNTIME_TEMPLATE_DICT: dict[str, Any] = {
    "metadata": {
        "name": SERVING_RUNTIME_TEMPLATE_NAME,
        "labels": OVMS_TEMPLATE_LABELS,
        "annotations": OVMS_TEMPLATE_ANNOTATIONS,
    },
    "objects": [
        {
            "apiVersion": "serving.kserve.io/v1alpha1",
            "kind": "ServingRuntime",
            "metadata": {
                "name": SERVING_RUNTIME_INSTANCE_NAME,
                "labels": OVMS_RUNTIME_LABELS,
                "annotations": OVMS_RUNTIME_ANNOTATIONS,
            },
            "spec": {
                "multiModel": False,
                "annotations": OVMS_RUNTIME_PROMETHEUS_ANNOTATIONS,
                "supportedModelFormats": OVMS_SUPPORTED_MODEL_FORMATS,
                "protocolVersions": ["v2", "grpc-v2"],
                "containers": [
                    {
                        "name": "kserve-container",
                        "image": OVMS_SERVING_RUNTIME_IMAGE,
                        "args": OVMS_CONTAINER_ARGS,
                        "ports": [{"containerPort": 8888, "protocol": "TCP"}],
                    }
                ],
            },
        }
    ],
}
