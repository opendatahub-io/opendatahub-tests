from typing import Optional

from ocp_resources.service import Service


class ProtocolNotSupported(Exception):
    def __init__(self, protocol: str):
        self.protocol = protocol

    def __str__(self) -> str:
        return f"Protocol {self.protocol} is not supported"


class TooManyServices(Exception):
    def __init__(self, services: list[Service]):
        self.services = services

    def __str__(self) -> str:
        return f"The Model Registry instance has too many Services, there should be only 1. List: {self.services}"


class InferenceResponseError(Exception):
    pass


class InvalidStorageArgument(Exception):
    def __init__(
        self,
        storageUri: Optional[str],
        storage_key: Optional[str],
        storage_path: Optional[str],
    ):
        self.storageUri = storageUri
        self.storage_key = storage_key
        self.storage_path = storage_path

    def __str__(self) -> str:
        msg = f"""
            You've passed the following parameters:
            "storageUri": {self.storageUri}
            "storage_key": {self.storage_key}
            "storage_path: {self.storage_path}
            In order to create a valid ISVC you need to specify either a storageUri value
            or both a storage key and a storage path.
        """
        return msg


class MetricValidationError(Exception):
    pass
