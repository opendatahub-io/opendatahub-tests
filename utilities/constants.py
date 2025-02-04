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


class ModelName:
    FLAN_T5_SMALL: str = "flan-t5-small"
    FLAN_T5_SMALL_HF: str = f"{FLAN_T5_SMALL}-hf"
    CAIKIT_BGE_LARGE_EN: str = f"bge-large-en-v1.5-{ModelFormat.CAIKIT}"


class ModelAndFormat:
    FLAN_T5_SMALL_CAIKIT: str = f"{ModelName.FLAN_T5_SMALL}-{ModelFormat.CAIKIT}"
    OPENVINO_IR: str = f"{ModelFormat.OPENVINO}_ir"
    KSERVE_OPENVINO_IR: str = f"{OPENVINO_IR}_kserve"
    ONNX_1: str = f"{ModelFormat.ONNX}-1"


class ModelStoragePath:
    FLAN_T5_SMALL: str = f"{ModelName.FLAN_T5_SMALL}/{ModelAndFormat.FLAN_T5_SMALL_CAIKIT}"
    OPENVINO_EXAMPLE_MODEL: str = f"{ModelFormat.OPENVINO}-example-model"
    KSERVE_OPENVINO_EXAMPLE_MODEL: str = f"kserve-openvino-test/{OPENVINO_EXAMPLE_MODEL}"
    EMBEDDING_MODEL: str = "embeddingsmodel"
    TENSORFLOW_MODEL: str = "inception_resnet_v2.pb"
    OPENVINO_VEHICLE_DETECTION: str = "vehicle-detection"


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
    TCP_PROTOCOLS: set[str] = {HTTP, HTTPS}
    ALL_SUPPORTED_PROTOCOLS: set[str] = TCP_PROTOCOLS.union({GRPC})


class HTTPRequest:
    # Use string formatting to set the token value when using this constant
    AUTH_HEADER: str = "-H 'Authorization: Bearer {token}'"
    CONTENT_JSON: str = "-H 'Content-Type: application/json'"


class AcceleratorType:
    NVIDIA: str = "nvidia"
    AMD: str = "amd"
    GAUDI: str = "gaudi"
    SUPPORTED_LISTS: list[str] = [NVIDIA, AMD, GAUDI]


class Annotations:
    class KubernetesIo:
        NAME: str = f"{Resource.ApiGroup.APP_KUBERNETES_IO}/name"
        INSTANCE: str = f"{Resource.ApiGroup.APP_KUBERNETES_IO}/instance"
        PART_OF: str = f"{Resource.ApiGroup.APP_KUBERNETES_IO}/part-of"
        CREATED_BY: str = f"{Resource.ApiGroup.APP_KUBERNETES_IO}/created-by"

    class KserveIo:
        DEPLOYMENT_MODE: str = "serving.kserve.io/deploymentMode"

    class KserveAuth:
        SECURITY: str = "security.opendatahub.io/enable-auth"


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
        DASHBOARD: str = "opendatahub.io/dashboard"

    class KserveAuth:
        SECURITY: str = "security.opendatahub.io/enable-auth"


MODEL_REGISTRY: str = "model-registry"
MODELMESH_SERVING: str = "modelmesh-serving"
ISTIO_CA_BUNDLE_FILENAME: str = "istio_knative.crt"
OPENSHIFT_CA_BUNDLE_FILENAME: str = "openshift_ca.crt"
