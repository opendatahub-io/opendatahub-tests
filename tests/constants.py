from enum import Enum


class KServeDeploymentType(Enum):
    SERVERLESS: str = "Serverless"
    RAW_DEPLOYMENT: str = "RawDeployment"


class ModelStoragePath(Enum):
    FLAN_T5_SMALL: str = "flan-t5-small/flan-t5-small-caikit"


class ModelFormat(Enum):
    CAIKIT: str = "caikit"


class CurlOutput(Enum):
    HEALTH_OK: str = "OK"


class ModelEndpoint(Enum):
    HEALTH: str = "health"
