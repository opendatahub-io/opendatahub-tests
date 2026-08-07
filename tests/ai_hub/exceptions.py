class ModelRegistryResourceNotCreated(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ModelRegistryResourceNotFoundError(Exception):
    pass


class ModelRegistryResourceNotUpdated(Exception):
    pass
