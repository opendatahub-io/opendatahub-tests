from __future__ import annotations

from ocp_resources.service import Service


class ProtocolNotSupportedError(Exception):
    def __init__(self, protocol: str):
        self.protocol = protocol

    def __str__(self) -> str:
        return f"Protocol {self.protocol} is not supported"


class TooManyServicesError(Exception):
    def __init__(self, services: list[Service]):
        self.services = services

    def __str__(self) -> str:
        return f"The Model Registry instance has too many Services, there should be only 1. List: {self.services}"


class InferenceResponseError(Exception):
    pass


class InvalidStorageArgumentError(Exception):
    def __init__(
        self,
        storage_uri: str | None,
        storage_key: str | None,
        storage_path: str | None,
    ):
        self.storage_uri = storage_uri
        self.storage_key = storage_key
        self.storage_path = storage_path

    def __str__(self) -> str:
        msg = f"""
            You've passed the following parameters:
            "storage_uri": {self.storage_uri}
            "storage_key": {self.storage_key}
            "storage_path: {self.storage_path}
            In order to create a valid ISVC you need to specify either a storage_uri value
            or both a storage key and a storage path.
        """
        return msg


class MetricValidationError(Exception):
    pass


class FailedPodsError(Exception):
    def __init__(self, pods: dict[str, str]):
        self.pods = pods

    def __str__(self) -> str:
        return f"The following pods are not running: {self.pods}"


class NotSupportedError(Exception):
    """
    Custom exception message for not supported types
    """

    def __init__(self, type: str):
        self.type = type

    def __str__(self) -> str:
        return f"The {self.type} is not supported"


class DeploymentValidationError(Exception):
    pass


class InferenceCanaryTrafficError(Exception):
    pass


class PodLogMissMatchError(Exception):
    pass


class ResourceMismatchError(Exception):
    pass


class ClusterLoginError(Exception):
    def __init__(self, user: str):
        self.user = user

    def __str__(self) -> str:
        return f"Failed to log in as user {self.user}."


class InvalidArgumentsError(Exception):
    """Raised when mutually exclusive or invalid argument combinations are passed."""


class ResourceNotReadyError(Exception):
    pass


class PodContainersRestartError(Exception):
    pass


class TooManyPodsError(Exception):
    pass


class UnexpectedFailureError(Exception):
    pass


class UnexpectedResourceCountError(Exception):
    """Unexpected number of API resources found"""


class ResourceValueMismatch(Exception):
    """Resource value mismatch"""


class MissingParameter(Exception):
    """Raised required argument is not passed."""


class ExceptionUserLogin(Exception):
    pass


class UnexpectedValueError(Exception):
    """Unexpected value found"""
