from typing import Any, Dict

from utilities.manifests.caikit_tgis import CAIKIT_TGIS_INFERENCE_CONFIG
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG
from utilities.manifests.openvino import (
    OPENVINO_INFERENCE_CONFIG,
    OPENVINO_KSERVE_INFERENCE_CONFIG,
)
from utilities.manifests.tgis_grpc import TGIS_INFERENCE_CONFIG


class KServeDeploymentType:
    SERVERLESS: str = "Serverless"
    RAW_DEPLOYMENT: str = "RawDeployment"
    MODEL_MESH: str = "ModelMesh"


class ModelFormat:
    CAIKIT: str = "caikit"
    ONNX: str = "onnx"
    OPENVINO: str = "openvino"
    OVMS: str = "ovms"


class ModelName:
    FLAN_T5_SMALL: str = "flan-t5-small"
    FLAN_T5_SMALL_HF: str = f"{FLAN_T5_SMALL}-hf"


class ModelAndFormat:
    FLAN_T5_SMALL_CAIKIT: str = f"{ModelName.FLAN_T5_SMALL}-{ModelFormat.CAIKIT}"
    OPENVINO_IR: str = f"{ModelFormat.OPENVINO}_ir"
    KSERVE_OPENVINO_IR: str = f"{OPENVINO_IR}_kserve"


class ModelStoragePath:
    FLAN_T5_SMALL: str = f"{ModelName.FLAN_T5_SMALL}/{ModelAndFormat.FLAN_T5_SMALL_CAIKIT}"
    OPENVINO_EXAMPLE_MODEL: str = f"{ModelFormat.OPENVINO}-example-model"
    KSERVE_OPENVINO_EXAMPLE_MODEL: str = f"kserve-openvino-test/{OPENVINO_EXAMPLE_MODEL}"


class CurlOutput:
    HEALTH_OK: str = "OK"


class ModelEndpoint:
    HEALTH: str = "health"


class ModelVersion:
    OPSET1: str = "opset1"
    OPSET13: str = "opset13"


class RuntimeTemplates:
    CAIKIT_TGIS_SERVING: str = "caikit-tgis-serving-template"
    OVMS_MODEL_MESH: str = ModelFormat.OVMS
    OVMS_KSERVE: str = f"kserve-{ModelFormat.OVMS}"


class ModelInferenceRuntime:
    TGIS_RUNTIME: str = "tgis-runtime"
    CAIKIT_TGIS_RUNTIME: str = f"{ModelFormat.CAIKIT}-{TGIS_RUNTIME}"
    OPENVINO_RUNTIME: str = f"{ModelFormat.OPENVINO}-runtime"
    OPENVINO_KSERVE_RUNTIME: str = f"{ModelFormat.OPENVINO}-kserve-runtime"
    ONNX_RUNTIME: str = f"{ModelFormat.ONNX}-runtime"

    MAPPING: Dict[str, Any] = {
        CAIKIT_TGIS_RUNTIME: CAIKIT_TGIS_INFERENCE_CONFIG,
        OPENVINO_RUNTIME: OPENVINO_INFERENCE_CONFIG,
        OPENVINO_KSERVE_RUNTIME: OPENVINO_KSERVE_INFERENCE_CONFIG,
        TGIS_RUNTIME: TGIS_INFERENCE_CONFIG,
        ONNX_RUNTIME: ONNX_INFERENCE_CONFIG,
    }


class Protocols:
    HTTP: str = "http"
    HTTPS: str = "https"
    GRPC: str = "grpc"
    TCP_PROTOCOLS: set[str] = {HTTP, HTTPS}
    ALL_SUPPORTED_PROTOCOLS: set[str] = TCP_PROTOCOLS.union({GRPC})


class AcceleratorType:
    NVIDIA: str = "nvidia"
    AMD: str = "amd"
    GAUDI: str = "gaudi"
    SUPPORTED_LISTS: list[str] = [NVIDIA, AMD, GAUDI]


MODELMESH_SERVING: str = "modelmesh-serving"
