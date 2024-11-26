from enum import Enum

APPLICATIONS_NAMESPACE: str = "redhat-ods-applications"


class KServeDeploymentType(Enum):
    SERVERLESS: str = "Serverless"
    RAW_DEPLOYMENT: str = "RawDeployment"


class ModelFormat(Enum):
    CAIKIT: str = "caikit"


class ModelStoragePath(Enum):
    FLAN_T5_SMALL: str = f"flan-t5-small/flan-t5-small-{ModelFormat.CAIKIT.value}"


class CurlOutput(Enum):
    HEALTH_OK: str = "OK"


class ModelEndpoint(Enum):
    HEALTH: str = "health"
