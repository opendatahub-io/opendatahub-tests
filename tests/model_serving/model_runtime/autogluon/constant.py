"""
Constants for AutoGluon KServe model serving tests.

Defines ServingRuntime spec helpers, S3 model prefixes, inference payloads,
and per-predictor configuration used across the AutoGluon test suite.
"""

from typing import Any

from utilities.constants import KServeDeploymentType, ModelFormat, ModelVersion

CLUSTER_SERVING_RUNTIME_NAME: str = "kserve-autogluonserver"


S3_PREFIX_TABULAR_V2: str = "autogluon/tabular-predictor"

S3_PREFIX_TABULAR_V1: str = "autogluon/tabular-predictor"
S3_PREFIX_TIMESERIES_V1: str = "autogluon/timeseries-predictor"

# autogluonserver paths (model_name = InferenceService metadata.name, e.g. tabular-v2).
V2_INFER_PATH_TEMPLATE: str = "/v2/models/{model_name}/infer"
V1_PREDICT_PATH_TEMPLATE: str = "/v1/models/{model_name}:predict"

PREDICT_RESOURCES: dict[str, dict[str, str]] = {
    "resources": {"requests": {"cpu": "1", "memory": "2Gi"}, "limits": {"cpu": "1", "memory": "2Gi"}},
}

BASE_RAW_DEPLOYMENT_CONFIG: dict[str, Any] = {
    "deployment_mode": KServeDeploymentType.STANDARD,
    "min-replicas": 1,
    "enable_external_route": True,
}


class OutputType:
    """Model output types for response validation."""

    DETERMINISTIC: str = "deterministic"
    NON_DETERMINISTIC: str = "non_deterministic"


class PredictorType:
    """AutoGluon predictor variants exercised by the S3 suite."""

    TABULAR_V2: str = "tabular-v2"
    TABULAR_V1: str = "tabular-v1"
    TIMESERIES_V1: str = "timeseries-v1"


class ProtocolVersion:
    """KServe protocol versions used for inference."""

    V1: str = "v1"
    V2: str = "v2"


# Tabular inference payloads (one row) for Telco Churn model schema.
# V2: one tensor per feature (input.name must match required feature names).
TABULAR_V2_INPUT: dict[str, Any] = {
    "inputs": [
        {"name": "gender", "shape": [1], "datatype": "BYTES", "data": ["Female"]},
        {"name": "SeniorCitizen", "shape": [1], "datatype": "INT64", "data": [0]},
        {"name": "Partner", "shape": [1], "datatype": "BYTES", "data": ["Yes"]},
        {"name": "Dependents", "shape": [1], "datatype": "BYTES", "data": ["No"]},
        {"name": "tenure", "shape": [1], "datatype": "INT64", "data": [1]},
        {"name": "PhoneService", "shape": [1], "datatype": "BYTES", "data": ["No"]},
        {"name": "MultipleLines", "shape": [1], "datatype": "BYTES", "data": ["No phone service"]},
        {"name": "InternetService", "shape": [1], "datatype": "BYTES", "data": ["DSL"]},
        {"name": "OnlineSecurity", "shape": [1], "datatype": "BYTES", "data": ["No"]},
        {"name": "OnlineBackup", "shape": [1], "datatype": "BYTES", "data": ["Yes"]},
        {"name": "DeviceProtection", "shape": [1], "datatype": "BYTES", "data": ["No"]},
        {"name": "TechSupport", "shape": [1], "datatype": "BYTES", "data": ["No"]},
        {"name": "StreamingTV", "shape": [1], "datatype": "BYTES", "data": ["No"]},
        {"name": "StreamingMovies", "shape": [1], "datatype": "BYTES", "data": ["No"]},
        {"name": "Contract", "shape": [1], "datatype": "BYTES", "data": ["Month-to-month"]},
        {"name": "PaperlessBilling", "shape": [1], "datatype": "BYTES", "data": ["Yes"]},
        {"name": "PaymentMethod", "shape": [1], "datatype": "BYTES", "data": ["Electronic check"]},
        {"name": "MonthlyCharges", "shape": [1], "datatype": "FP64", "data": [29.85]},
        {"name": "TotalCharges", "shape": [1], "datatype": "FP64", "data": [29.85]},
    ]
}

# V1: row-wise values in the same feature order as required by the runtime.
TABULAR_V1_INPUT: dict[str, Any] = {
    "instances": [
        [
            "Female",
            0,
            "Yes",
            "No",
            1,
            "No",
            "No phone service",
            "DSL",
            "No",
            "Yes",
            "No",
            "No",
            "No",
            "No",
            "Month-to-month",
            "Yes",
            "Electronic check",
            29.85,
            29.85,
        ]
    ]
}

TIMESERIES_V1_INPUT: dict[str, Any] = {
    "instances": [
        {"item_id": "industry_a", "timestamp": "2021-08-18", "target": 47.2642021},
        {"item_id": "industry_a", "timestamp": "2021-08-19", "target": 47.88765448},
        {"item_id": "industry_a", "timestamp": "2021-08-20", "target": 48.6093501},
        {"item_id": "industry_a", "timestamp": "2021-08-21", "target": 48.48047505},
        {"item_id": "industry_a", "timestamp": "2021-08-22", "target": 47.76307524},
    ],
}

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    PredictorType.TABULAR_V2: {
        "model_version": ModelVersion.AUTOGLUON_1,
        "protocol_version": ProtocolVersion.V2,
        "input_payload": TABULAR_V2_INPUT,
        "output_type": OutputType.DETERMINISTIC,
        "s3_prefix": S3_PREFIX_TABULAR_V2,
    },
    PredictorType.TABULAR_V1: {
        "model_version": ModelVersion.AUTOGLUON_1,
        "protocol_version": ProtocolVersion.V1,
        "input_payload": TABULAR_V1_INPUT,
        "output_type": OutputType.DETERMINISTIC,
        "s3_prefix": S3_PREFIX_TABULAR_V1,
    },
    PredictorType.TIMESERIES_V1: {
        "model_version": ModelVersion.AUTOGLUON_1,
        "protocol_version": ProtocolVersion.V1,
        "input_payload": TIMESERIES_V1_INPUT,
        "output_type": OutputType.DETERMINISTIC,
        "s3_prefix": S3_PREFIX_TIMESERIES_V1,
    },
}


def build_serving_runtime_kwargs(
    namespace: str,
    image: str,
    name: str,
) -> dict[str, Any]:
    """
    Build keyword arguments for a namespace-scoped AutoGluon ServingRuntime.

    Args:
        namespace: Target namespace for the ServingRuntime.
        image: Container image resolved from the cluster or an override.
        name: ServingRuntime resource name (must match create_isvc runtime=...).

    Returns:
        dict[str, Any]: Arguments for ocp_resources.serving_runtime.ServingRuntime.
    """
    return {
        "name": name,
        "namespace": namespace,
        "annotations": {
            "opendatahub.io/dashboard": "true",
            "opendatahub.io/kserve-runtime": ModelFormat.AUTOGLUON,
            "openshift.io/display-name": "AutoGluon Runtime",
        },
        "spec_annotations": {
            "prometheus.io/path": "/metrics",
            "prometheus.io/port": "8080",
        },
        "multi_model": False,
        "protocol_versions": [ProtocolVersion.V1, ProtocolVersion.V2],
        "supported_model_formats": [
            {"name": ModelFormat.AUTOGLUON, "version": ModelVersion.AUTOGLUON_1, "autoSelect": True},
        ],
        "containers": [
            {
                "name": "kserve-container",
                "image": image,
                "args": [
                    "--model_name={{.Name}}",
                    "--model_dir=/mnt/models",
                    "--http_port=8080",
                ],
                "ports": [{"containerPort": 8080, "protocol": "TCP"}],
                "resources": PREDICT_RESOURCES["resources"],
            },
        ],
    }
