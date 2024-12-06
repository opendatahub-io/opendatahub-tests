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
