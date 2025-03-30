from typing import Any

from ocp_resources.resource import Resource


class KServeDeploymentType:
    SERVERLESS: str = "Serverless"
    RAW_DEPLOYMENT: str = "RawDeployment"
    MODEL_MESH: str = "ModelMesh"


class ModelFormat:
    CAIKIT: str = "caikit"
    ONNX: str = "onnx"
    OPENVINO: str = "openvino"
    OVMS: str = "ovms"
    VLLM: str = "vllm"
    TENSORFLOW: str = "tensorflow"
    PYTORCH: str = "pytorch"


class ModelName:
    FLAN_T5_SMALL: str = "flan-t5-small"
    FLAN_T5_SMALL_HF: str = f"{FLAN_T5_SMALL}-hf"
    CAIKIT_BGE_LARGE_EN: str = f"bge-large-en-v1.5-{ModelFormat.CAIKIT}"
    BLOOM_560M: str = "bloom-560m"
    MNIST: str = "mnist"


class ModelAndFormat:
    FLAN_T5_SMALL_CAIKIT: str = f"{ModelName.FLAN_T5_SMALL}-{ModelFormat.CAIKIT}"
    OPENVINO_IR: str = f"{ModelFormat.OPENVINO}_ir"
    KSERVE_OPENVINO_IR: str = f"{OPENVINO_IR}_kserve"
    ONNX_1: str = f"{ModelFormat.ONNX}-1"
    BLOOM_560M_CAIKIT: str = f"bloom-560m-{ModelFormat.CAIKIT}"


class ModelStoragePath:
    FLAN_T5_SMALL_CAIKIT: str = f"{ModelName.FLAN_T5_SMALL}/{ModelAndFormat.FLAN_T5_SMALL_CAIKIT}"
    OPENVINO_EXAMPLE_MODEL: str = f"{ModelFormat.OPENVINO}-example-model"
    KSERVE_OPENVINO_EXAMPLE_MODEL: str = f"kserve-openvino-test/{OPENVINO_EXAMPLE_MODEL}"
    EMBEDDING_MODEL: str = "embeddingsmodel"
    TENSORFLOW_MODEL: str = "inception_resnet_v2.pb"
    OPENVINO_VEHICLE_DETECTION: str = "vehicle-detection"
    FLAN_T5_SMALL_HF: str = f"{ModelName.FLAN_T5_SMALL}/{ModelName.FLAN_T5_SMALL_HF}"
    BLOOM_560M_CAIKIT: str = f"{ModelName.BLOOM_560M}/{ModelAndFormat.BLOOM_560M_CAIKIT}"
    MNIST_8_ONNX: str = f"{ModelName.MNIST}-8.onnx"


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
    CAIKIT_STANDALONE_SERVING: str = "caikit-standalone-serving-template"
    TGIS_GRPC_SERVING: str = "tgis-grpc-serving-template"
    VLLM_CUDA: str = "vllm-cuda-runtime-template"
    VLLM_ROCM: str = "vllm-rocm-runtime-template"
    VLLM_GAUDUI: str = "vllm-gaudi-runtime-template"


class ModelInferenceRuntime:
    TGIS_RUNTIME: str = "tgis-runtime"
    CAIKIT_TGIS_RUNTIME: str = f"{ModelFormat.CAIKIT}-{TGIS_RUNTIME}"
    OPENVINO_RUNTIME: str = f"{ModelFormat.OPENVINO}-runtime"
    OPENVINO_KSERVE_RUNTIME: str = f"{ModelFormat.OPENVINO}-kserve-runtime"
    ONNX_RUNTIME: str = f"{ModelFormat.ONNX}-runtime"
    CAIKIT_STANDALONE_RUNTIME: str = f"{ModelFormat.CAIKIT}-standalone-runtime"
    VLLM_RUNTIME: str = f"{ModelFormat.VLLM}-runtime"
    TENSORFLOW_RUNTIME: str = f"{ModelFormat.TENSORFLOW}-runtime"


class Protocols:
    HTTP: str = "http"
    HTTPS: str = "https"
    GRPC: str = "grpc"
    REST: str = "rest"
    TCP: str = "TCP"
    TCP_PROTOCOLS: set[str] = {HTTP, HTTPS}
    ALL_SUPPORTED_PROTOCOLS: set[str] = TCP_PROTOCOLS.union({GRPC})


class Ports:
    GRPC_PORT: int = 8033
    REST_PORT: int = 8080


class PortNames:
    REST_PORT_NAME: str = "http1"
    GRPC_PORT_NAME: str = "h2c"


class HTTPRequest:
    # Use string formatting to set the token value when using this constant
    AUTH_HEADER: str = "-H 'Authorization: Bearer {token}'"
    CONTENT_JSON: str = "-H 'Content-Type: application/json'"


class AcceleratorType:
    NVIDIA: str = "nvidia"
    AMD: str = "amd"
    GAUDI: str = "gaudi"
    SUPPORTED_LISTS: list[str] = [NVIDIA, AMD, GAUDI]


class ApiGroups:
    OPENDATAHUB_IO: str = "opendatahub.io"


class Annotations:
    class KubernetesIo:
        NAME: str = f"{Resource.ApiGroup.APP_KUBERNETES_IO}/name"
        INSTANCE: str = f"{Resource.ApiGroup.APP_KUBERNETES_IO}/instance"
        PART_OF: str = f"{Resource.ApiGroup.APP_KUBERNETES_IO}/part-of"
        CREATED_BY: str = f"{Resource.ApiGroup.APP_KUBERNETES_IO}/created-by"

    class KserveIo:
        DEPLOYMENT_MODE: str = "serving.kserve.io/deploymentMode"

    class KserveAuth:
        SECURITY: str = f"security.{ApiGroups.OPENDATAHUB_IO}/enable-auth"

    class OpenDataHubIo:
        MANAGED: str = f"{ApiGroups.OPENDATAHUB_IO}/managed"
        SERVICE_MESH: str = f"{ApiGroups.OPENDATAHUB_IO}/service-mesh"


class StorageClassName:
    NFS: str = "nfs"


class DscComponents:
    MODELMESHSERVING: str = "modelmeshserving"
    KSERVE: str = "kserve"
    MODELREGISTRY: str = "modelregistry"

    class ManagementState:
        MANAGED: str = "Managed"
        REMOVED: str = "Removed"

    class ConditionType:
        MODEL_REGISTRY_READY: str = "ModelRegistryReady"
        KSERVE_READY: str = "KserveReady"
        MODEL_MESH_SERVING_READY: str = "ModelMeshServingReady"

    COMPONENT_MAPPING: dict[str, str] = {
        MODELMESHSERVING: ConditionType.MODEL_MESH_SERVING_READY,
        KSERVE: ConditionType.KSERVE_READY,
        MODELREGISTRY: ConditionType.MODEL_REGISTRY_READY,
    }


class Labels:
    class OpenDataHub:
        DASHBOARD: str = f"{ApiGroups.OPENDATAHUB_IO}/dashboard"

    class KserveAuth:
        SECURITY: str = f"security.{ApiGroups.OPENDATAHUB_IO}/enable-auth"

    class Notebook:
        INJECT_OAUTH: str = f"notebooks.{ApiGroups.OPENDATAHUB_IO}/inject-oauth"

    class OpenDataHubIo:
        MANAGED: str = Annotations.OpenDataHubIo.MANAGED


class Timeout:
    TIMEOUT_1MIN: int = 60
    TIMEOUT_2MIN: int = 2 * TIMEOUT_1MIN
    TIMEOUT_4MIN: int = 4 * TIMEOUT_1MIN
    TIMEOUT_5MIN: int = 5 * TIMEOUT_1MIN
    TIMEOUT_10MIN: int = 10 * TIMEOUT_1MIN
    TIMEOUT_15MIN: int = 15 * TIMEOUT_1MIN


class Containers:
    KSERVE_CONTAINER_NAME: str = "kserve-container"


class RunTimeConfigs:
    ONNX_OPSET13_RUNTIME_CONFIG: dict[str, Any] = {
        "runtime-name": ModelInferenceRuntime.ONNX_RUNTIME,
        "model-format": {ModelFormat.ONNX: ModelVersion.OPSET13},
    }


MODEL_REGISTRY: str = "model-registry"
MODELMESH_SERVING: str = "modelmesh-serving"
ISTIO_CA_BUNDLE_FILENAME: str = "istio_knative.crt"
OPENSHIFT_CA_BUNDLE_FILENAME: str = "openshift_ca.crt"
INTERNAL_IMAGE_REGISTRY_PATH: str = "image-registry.openshift-image-registry.svc:5000"

vLLM_CONFIG: dict[str, dict[str, Any]] = {
    "port_configurations": {
        "grpc": [{"containerPort": Ports.GRPC_PORT, "name": PortNames.GRPC_PORT_NAME, "protocol": Protocols.TCP}],
        "raw": [
            {"containerPort": Ports.REST_PORT, "name": PortNames.REST_PORT_NAME, "protocol": Protocols.TCP},
            {"containerPort": Ports.GRPC_PORT, "name": PortNames.GRPC_PORT_NAME, "protocol": Protocols.TCP},
        ],
    },
    "commands": {"GRPC": "vllm_tgis_adapter"},
}
