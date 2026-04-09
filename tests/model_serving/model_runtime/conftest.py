from typing import Any

import pytest
from syrupy.extensions.json import JSONSnapshotExtension


@pytest.fixture
def response_snapshot(snapshot: Any) -> Any:
    return snapshot.use_extension(extension_class=JSONSnapshotExtension)
