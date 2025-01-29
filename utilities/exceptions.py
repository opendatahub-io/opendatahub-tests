from typing import Optional

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
        storage_uri: Optional[str],
        storage_key: Optional[str],
        storage_path: Optional[str],
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
