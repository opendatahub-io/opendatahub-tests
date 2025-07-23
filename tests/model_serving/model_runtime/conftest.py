from typing import Any

import pytest
from syrupy.extensions.json import JSONSnapshotExtension


@pytest.fixture(scope="session")
def skip_if_no_supported_accelerator_type(supported_accelerator_type: str) -> None:
    if not supported_accelerator_type:
        pytest.skip("Accelerator type is not provided,vLLM test cannot be run on CPU")


@pytest.fixture
def response_snapshot(snapshot: Any) -> Any:
    return snapshot.use_extension(extension_class=JSONSnapshotExtension)
