APPLICATIONS_NAMESPACE: str = "redhat-ods-applications"


class KServeDeploymentType:
    SERVERLESS: str = "Serverless"
    RAW_DEPLOYMENT: str = "RawDeployment"


class ModelFormat:
    CAIKIT: str = "caikit"


class ModelName:
    FLAN_T5_SMALL: str = "flan-t5-small"


class ModelAndFormat:
    FLAN_T5_SMALL_CAIKIT: str = f"{ModelName.FLAN_T5_SMALL}-{ModelFormat.CAIKIT}"


class ModelStoragePath:
    FLAN_T5_SMALL: str = f"{ModelName.FLAN_T5_SMALL}/{ModelAndFormat.FLAN_T5_SMALL_CAIKIT}"


class CurlOutput:
    HEALTH_OK: str = "OK"


class ModelEndpoint:
    HEALTH: str = "health"


class RuntimeTemplates:
    CAIKIT_TGIS_SERVING: str = "caikit-tgis-serving-template"


class RuntimeQueryKeys:
    CAIKIT_TGIS_RUNTIME: str = f"{ModelFormat.CAIKIT}-tgis-runtime"


class Protocols:
    HTTP: str = "http"
    HTTPS: str = "https"
    GRPC: str = "grpc"
    REST: str = "rest"


class HTTPRequest:
    # You will need to `.format(token='foo')`
    # See e.g. tests/model_registry/utils.py#L51
    AUTH_HEADER: str = "-H 'Authorization: Bearer {token}'"
    CONTENT_JSON: str = "-H 'Content-Type: application/json'"


class AcceleratorType:
    NVIDIA: str = "nvidia"
    AMD: str = "amd"
    GAUDI: str = "gaudi"
    SUPPORTED_LISTS: list[str] = [NVIDIA, AMD, GAUDI]


class KubernetesAnnotations:
    NAME: str = "app.kubernetes.io/name"
    INSTANCE: str = "app.kubernetes.io/instance"
    PART_OF: str = "app.kubernetes.io/part-of"
    CREATED_BY: str = "app.kubernetes.io/created-by"
