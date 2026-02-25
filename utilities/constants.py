from dataclasses import dataclass
from typing import Any, Dict

from ocp_resources.resource import Resource


class KServeDeploymentType:
    SERVERLESS: str = "Serverless"
    RAW_DEPLOYMENT: str = "RawDeployment"
    MODEL_MESH: str = "ModelMesh"


class ModelFormat:
    CAIKIT: str = "caikit"
    LIGHTGBM: str = "lightgbm"
    MLSERVER: str = "mlserver"
    ONNX: str = "onnx"
    OPENVINO: str = "openvino"
    OVMS: str = "ovms"
    PYTORCH: str = "pytorch"
    SKLEARN: str = "sklearn"
    TENSORFLOW: str = "tensorflow"
    VLLM: str = "vllm"
    XGBOOST: str = "xgboost"


class ModelName:
    FLAN_T5_SMALL: str = "flan-t5-small"
    FLAN_T5_SMALL_HF: str = f"{FLAN_T5_SMALL}-hf"
    CAIKIT_BGE_LARGE_EN: str = f"bge-large-en-v1.5-{ModelFormat.CAIKIT}"
    BLOOM_560M: str = "bloom-560m"
    MNIST: str = "mnist"
    # LLM Models
    QWEN: str = "Qwen"
    TINYLLAMA: str = "TinyLlama"


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
    DOG_BREED_ONNX: str = "dog_breed_classification"
    CAT_DOG_ONNX: str = "cat_dog_classification"


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
    VLLM_GAUDI: str = "vllm-gaudi-runtime-template"
    VLLM_SPYRE: str = "vllm-spyre-x86-runtime-template"
    VLLM_CPU_x86: str = "vllm-cpu-x86-runtime-template"
    MLSERVER: str = f"{ModelFormat.MLSERVER}-runtime-template"
    TRITON_REST: str = "triton-rest-runtime-template"
    TRITON_GRPC: str = "triton-grpc-runtime-template"
    GUARDRAILS_DETECTOR_HUGGINGFACE: str = "guardrails-detector-huggingface-serving-template"


class ModelInferenceRuntime:
    TGIS_RUNTIME: str = "tgis-runtime"
    CAIKIT_TGIS_RUNTIME: str = f"{ModelFormat.CAIKIT}-{TGIS_RUNTIME}"
    OPENVINO_RUNTIME: str = f"{ModelFormat.OPENVINO}-runtime"
    OPENVINO_KSERVE_RUNTIME: str = f"{ModelFormat.OPENVINO}-kserve-runtime"
    ONNX_RUNTIME: str = f"{ModelFormat.ONNX}-runtime"
    CAIKIT_STANDALONE_RUNTIME: str = f"{ModelFormat.CAIKIT}-standalone-runtime"
    VLLM_RUNTIME: str = f"{ModelFormat.VLLM}-runtime"
    TENSORFLOW_RUNTIME: str = f"{ModelFormat.TENSORFLOW}-runtime"
    MLSERVER_RUNTIME: str = f"{ModelFormat.MLSERVER}-runtime"


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
    SPYRE: str = "spyre"
    CPU_x86: str = "cpu_x86"
    SUPPORTED_LISTS: list[str] = [NVIDIA, AMD, GAUDI, SPYRE, CPU_x86]


class ApiGroups:
    HAPROXY_ROUTER_OPENSHIFT_IO: str = "haproxy.router.openshift.io"
    OPENDATAHUB_IO: str = "opendatahub.io"
    KSERVE: str = "serving.kserve.io"
    KUADRANT_IO: str = "kuadrant.io"
    MAAS_IO: str = "maas.opendatahub.io"


class Annotations:
    class KubernetesIo:
        NAME: str = f"{Resource.ApiGroup.APP_KUBERNETES_IO}/name"
        INSTANCE: str = f"{Resource.ApiGroup.APP_KUBERNETES_IO}/instance"
        PART_OF: str = f"{Resource.ApiGroup.APP_KUBERNETES_IO}/part-of"
        CREATED_BY: str = f"{Resource.ApiGroup.APP_KUBERNETES_IO}/created-by"

    class KserveIo:
        DEPLOYMENT_MODE: str = f"{ApiGroups.KSERVE}/deploymentMode"
        FORCE_STOP_RUNTIME: str = f"{ApiGroups.KSERVE}/stop"

    class KserveAuth:
        SECURITY: str = f"security.{ApiGroups.OPENDATAHUB_IO}/enable-auth"

    class OpenDataHubIo:
        MANAGED: str = f"{ApiGroups.OPENDATAHUB_IO}/managed"
        SERVICE_MESH: str = f"{ApiGroups.OPENDATAHUB_IO}/service-mesh"

    class HaproxyRouterOpenshiftIo:
        TIMEOUT: str = f"{ApiGroups.HAPROXY_ROUTER_OPENSHIFT_IO}/timeout"


class StorageClassName:
    NFS: str = "nfs"


class DscComponents:
    MODELMESHSERVING: str = "modelmeshserving"
    KSERVE: str = "kserve"
    MODELREGISTRY: str = "modelregistry"
    LLAMASTACKOPERATOR: str = "llamastackoperator"
    KUEUE: str = "kueue"

    class ManagementState:
        MANAGED: str = "Managed"
        REMOVED: str = "Removed"
        UNMANAGED: str = "Unmanaged"

    class ConditionType:
        MODEL_REGISTRY_READY: str = "ModelRegistryReady"
        KSERVE_READY: str = "KserveReady"
        MODEL_MESH_SERVING_READY: str = "ModelMeshServingReady"
        LLAMA_STACK_OPERATOR_READY: str = "LlamaStackOperatorReady"

    COMPONENT_MAPPING: dict[str, str] = {
        MODELMESHSERVING: ConditionType.MODEL_MESH_SERVING_READY,
        KSERVE: ConditionType.KSERVE_READY,
        MODELREGISTRY: ConditionType.MODEL_REGISTRY_READY,
        LLAMASTACKOPERATOR: ConditionType.LLAMA_STACK_OPERATOR_READY,
    }


class Labels:
    class OpenDataHub:
        DASHBOARD: str = f"{ApiGroups.OPENDATAHUB_IO}/dashboard"

    class KserveAuth:
        SECURITY: str = f"security.{ApiGroups.OPENDATAHUB_IO}/enable-auth"

    class Notebook:
        INJECT_AUTH: str = f"notebooks.{ApiGroups.OPENDATAHUB_IO}/inject-auth"

    class OpenDataHubIo:
        MANAGED: str = Annotations.OpenDataHubIo.MANAGED
        NAME: str = f"component.{ApiGroups.OPENDATAHUB_IO}/name"

    class Openshift:
        APP: str = "app"

    class Kserve:
        NETWORKING_KSERVE_IO: str = "networking.kserve.io/visibility"
        NETWORKING_KNATIVE_IO: str = "networking.knative.dev/visibility"
        EXPOSED: str = "exposed"
        # Gateway labels
        GATEWAY_LABEL: str = "serving.kserve.io/gateway"

    class Nvidia:
        NVIDIA_COM_GPU: str = "nvidia.com/gpu"

    class ROCm:
        ROCM_GPU: str = "amd.com/gpu"

    class Spyre:
        SPYRE_COM_GPU: str = "ibm.com/spyre_pf"

    class CPU:
        CPU_x86: str = "cpu"

    class Kueue:
        MANAGED: str = "kueue.openshift.io/managed"
        QUEUE_NAME: str = "kueue.x-k8s.io/queue-name"


class Timeout:
    TIMEOUT_15_SEC: int = 15
    TIMEOUT_30SEC: int = 30
    TIMEOUT_1MIN: int = 60
    TIMEOUT_2MIN: int = 2 * TIMEOUT_1MIN
    TIMEOUT_4MIN: int = 4 * TIMEOUT_1MIN
    TIMEOUT_5MIN: int = 5 * TIMEOUT_1MIN
    TIMEOUT_10MIN: int = 10 * TIMEOUT_1MIN
    TIMEOUT_15MIN: int = 15 * TIMEOUT_1MIN
    TIMEOUT_20MIN: int = 20 * TIMEOUT_1MIN
    TIMEOUT_30MIN: int = 30 * TIMEOUT_1MIN
    TIMEOUT_40MIN: int = 40 * TIMEOUT_1MIN


class ResourceLimits:
    """Standard resource limits for different workload types."""

    class CPU:
        LIMIT: str = "1"
        REQUEST: str = "100m"

    class Memory:
        LIMIT: str = "10Gi"
        REQUEST: str = "8Gi"

    class GPU:
        # GPU resource limits
        LIMIT: str = "1"
        REQUEST: str = "1"

        # CPU limits for GPU workloads (higher requirements)
        CPU_LIMIT: str = "4"
        CPU_REQUEST: str = "2"

        # Memory limits for GPU workloads (higher requirements)
        MEMORY_LIMIT: str = "32Gi"
        MEMORY_REQUEST: str = "16Gi"


class OpenshiftRouteTimeout:
    TIMEOUT_1MICROSEC: str = "1us"


class Containers:
    KSERVE_CONTAINER_NAME: str = "kserve-container"


class RunTimeConfigs:
    ONNX_OPSET13_RUNTIME_CONFIG: dict[str, Any] = {
        "runtime-name": ModelInferenceRuntime.ONNX_RUNTIME,
        "model-format": {ModelFormat.ONNX: ModelVersion.OPSET13},
    }


class ModelCarImage:
    MNIST_8_1: str = (
        "oci://quay.io/mwaykole/test@sha256:cb7d25c43e52c755e85f5b59199346f30e03b7112ef38b74ed4597aec8748743"
        # noqa: E501
    )
    GRANITE_8B_CODE_INSTRUCT: str = "oci://registry.redhat.io/rhelai1/modelcar-granite-8b-code-instruct:1.4"


class ModelStorage:
    """Model storage URIs for different storage backends."""

    class OCI:
        TINYLLAMA: str = (
            "oci://quay.io/mwaykole/test@sha256:8bfd02132b03977ebbca93789e81c4549d8f724ee78fa378616d9ae4387717c8"
        )
        MNIST_8_1: str = ModelCarImage.MNIST_8_1
        GRANITE_8B_CODE_INSTRUCT: str = ModelCarImage.GRANITE_8B_CODE_INSTRUCT

    class S3:
        QWEN_7B_INSTRUCT: str = "s3://ods-ci-wisdom/Qwen2.5-7B-Instruct/"
        TINYLLAMA: str = "s3://ods-ci-wisdom/TinyLlama-1.1B-Chat-v1.0/"
        OPT_125M: str = "s3://ods-ci-wisdom/opt-125m/"

    class HuggingFace:
        TINYLLAMA: str = "hf://TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        OPT125M: str = "hf://facebook/opt-125m"


class OCIRegistry:
    class Metadata:
        NAME: str = "oci-registry"
        DEFAULT_PORT: int = 5000
        DEFAULT_HTTP_ADDRESS: str = "0.0.0.0"

    class PodConfig:
        REGISTRY_IMAGE: str = "ghcr.io/project-zot/zot:v2.1.8"
        REGISTRY_BASE_CONFIG: dict[str, Any] = {
            "args": None,
            "labels": {
                "maistra.io/expose-route": "true",
            },
        }

    class Storage:
        STORAGE_DRIVER: str = "s3"
        STORAGE_DRIVER_ROOT_DIRECTORY: str = "/registry"
        STORAGE_DRIVER_REGION: str = "us-east-1"
        STORAGE_STORAGEDRIVER_SECURE: str = "false"
        STORAGE_STORAGEDRIVER_FORCEPATHSTYLE: str = "true"


class MinIo:
    class Metadata:
        NAME: str = "minio"
        DEFAULT_PORT: int = 9000
        DEFAULT_ENDPOINT: str = f"{Protocols.HTTP}://{NAME}:{DEFAULT_PORT}"

    class Credentials:
        ACCESS_KEY_NAME: str = "MINIO_ROOT_USER"
        ACCESS_KEY_VALUE: str = "THEACCESSKEY"
        SECRET_KEY_NAME: str = "MINIO_ROOT_PASSWORD"
        SECRET_KEY_VALUE: str = "THESECRETKEY"

    class Buckets:
        EXAMPLE_MODELS: str = "example-models"
        MODELMESH_EXAMPLE_MODELS: str = f"modelmesh-{EXAMPLE_MODELS}"

    class PodConfig:
        KSERVE_MINIO_IMAGE: str = (
            "quay.io/jooholee/model-minio@sha256:b9554be19a223830cf792d5de984ccc57fc140b954949f5ffc6560fab977ca7a"
            # noqa: E501
        )
        MINIO_BASE_LABELS_ANNOTATIONS: dict[str, Any] = {
            "labels": {
                "maistra.io/expose-route": "true",
            },
            "annotations": {
                "sidecar.istio.io/inject": "true",
            },
        }

        MINIO_BASE_CONFIG: dict[str, Any] = {
            "args": ["server", "/data1"],
            **MINIO_BASE_LABELS_ANNOTATIONS,
        }

        MODEL_MESH_MINIO_CONFIG: dict[str, Any] = {
            "image": "quay.io/trustyai_testing/modelmesh-minio-examples@sha256:d2ccbe92abf9aa5085b594b2cae6c65de2bf06306c30ff5207956eb949bb49da",  # noqa: E501
            **MINIO_BASE_CONFIG,
        }

        QWEN_MINIO_CONFIG: dict[str, Any] = {
            "image": "quay.io/trustyai_testing/hf-llm-minio@sha256:2404a37d578f2a9c7adb3971e26a7438fedbe7e2e59814f396bfa47cd5fe93bb",  # noqa: E501
            **MINIO_BASE_CONFIG,
        }

        QWEN_HAP_BPIV2_MINIO_CONFIG: dict[str, Any] = {
            "image": "quay.io/trustyai_testing/qwen2.5-0.5b-instruct-hap-bpiv2-minio@sha256:eac1ca56f62606e887c80b4a358b3061c8d67f0b071c367c0aa12163967d5b2b",  # noqa: E501
            **MINIO_BASE_CONFIG,
        }

        KSERVE_MINIO_CONFIG: dict[str, Any] = {
            "image": KSERVE_MINIO_IMAGE,
            **MINIO_BASE_CONFIG,
        }

        MODEL_REGISTRY_MINIO_CONFIG: dict[str, Any] = {
            "image": "quay.io/minio/minio@sha256:14cea493d9a34af32f524e538b8346cf79f3321eff8e708c1e2960462bd8936e",
            "args": ["server", "/data"],
            **MINIO_BASE_LABELS_ANNOTATIONS,
        }

    class RunTimeConfig:
        # TODO: Remove runtime_image once ovms/loan_model_alpha model works with latest ovms
        IMAGE = "quay.io/opendatahub/openvino_model_server@sha256:564664371d3a21b9e732a5c1b4b40bacad714a5144c0a9aaf675baec4a04b148"  # noqa: E501


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

RHOAI_OPERATOR_NAMESPACE = "redhat-ods-operator"
OPENSHIFT_OPERATORS: str = "openshift-operators"

MAAS_GATEWAY_NAME: str = "maas-default-gateway"
MAAS_GATEWAY_NAMESPACE: str = "openshift-ingress"
MAAS_RATE_LIMIT_POLICY_NAME: str = "gateway-rate-limits"
MAAS_TOKEN_RATE_LIMIT_POLICY_NAME: str = "gateway-token-rate-limits"

MARIADB: str = "mariadb"
MODEL_REGISTRY_CUSTOM_NAMESPACE: str = "model-registry-custom-ns"
THANOS_QUERIER_ADDRESS = "https://thanos-querier.openshift-monitoring.svc:9092"
BUILTIN_DETECTOR_CONFIG: Dict[str, Any] = {
    "regex": {
        "type": "text_contents",
        "service": {
            "hostname": "127.0.0.1",
            "port": 8080,
        },
        "chunker_id": "whole_doc_chunker",
        "default_threshold": 0.5,
    }
}

QWEN_ISVC_NAME = "qwen-isvc"
QWEN_MODEL_NAME: str = "qwen25-05b-instruct"


class ContainerImages:
    """Centralized container images for various runtimes and models."""

    class VLLM:
        CPU: str = "quay.io/pierdipi/vllm-cpu@sha256:ce3a0c057394b2c332498f9742a17fd31b5cc2ef07db882d579fd157fe2c9a98"

    class MinIO:
        KSERVE: str = (
            "quay.io/jooholee/model-minio@sha256:b9554be19a223830cf792d5de984ccc57fc140b954949f5ffc6560fab977ca7a"
        )
        MODEL_MESH: str = "quay.io/trustyai_testing/modelmesh-minio-examples@sha256:d2ccbe92abf9aa5085b594b2cae6c65de2bf06306c30ff5207956eb949bb49da"  # noqa: E501
        QWEN: str = "quay.io/trustyai_testing/hf-llm-minio@sha256:2404a37d578f2a9c7adb3971e26a7438fedbe7e2e59814f396bfa47cd5fe93bb"  # noqa: E501
        QWEN_HAP_BPIV2: str = "quay.io/trustyai_testing/qwen2.5-0.5b-instruct-hap-bpiv2-minio@sha256:eac1ca56f62606e887c80b4a358b3061c8d67f0b071c367c0aa12163967d5b2b"  # noqa: E501
        MODEL_REGISTRY: str = (
            "quay.io/minio/minio@sha256:14cea493d9a34af32f524e538b8346cf79f3321eff8e708c1e2960462bd8936e"
        )

    class OCI:
        REGISTRY: str = "ghcr.io/project-zot/zot:v2.1.8"

    class OpenVINO:
        MODEL_SERVER: str = "quay.io/opendatahub/openvino_model_server@sha256:564664371d3a21b9e732a5c1b4b40bacad714a5144c0a9aaf675baec4a04b148"  # noqa: E501


CHAT_GENERATION_CONFIG: Dict[str, Any] = {
    "service": {
        "hostname": f"{QWEN_MODEL_NAME}-predictor",
        "port": 80,
        "request_timeout": 600,
    }
}
TRUSTYAI_SERVICE_NAME: str = "trustyai-service"

LLM_D_INFERENCE_SIM_NAME = "llm-d-inference-sim"


@dataclass
class LLMdInferenceSimConfig:
    name: str = LLM_D_INFERENCE_SIM_NAME
    port: int = 8032
    model_name: str = "Qwen2.5-1.5B-Instruct"
    serving_runtime_name: str = f"{LLM_D_INFERENCE_SIM_NAME}-serving-runtime"
    isvc_name: str = f"{LLM_D_INFERENCE_SIM_NAME}-isvc"


LLM_D_CHAT_GENERATION_CONFIG: Dict[str, Any] = {
    "service": {"hostname": f"{LLMdInferenceSimConfig.isvc_name}-predictor", "port": 80}
}


class PodNotFound(Exception):
    """Pod not found"""

    pass
