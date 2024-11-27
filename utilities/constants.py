APPLICATIONS_NAMESPACE: str = "redhat-ods-applications"


class KServeDeploymentType:
    SERVERLESS: str = "Serverless"
    RAW_DEPLOYMENT: str = "RawDeployment"


class ModelFormat:
    CAIKIT: str = "caikit"


class ModelStoragePath:
    FLAN_T5_SMALL: str = f"flan-t5-small/flan-t5-small-{ModelFormat.CAIKIT}"


class CurlOutput:
    HEALTH_OK: str = "OK"


class ModelEndpoint:
    HEALTH: str = "health"
