from collections.abc import Generator
from typing import Any

import pytest
from ocp_resources.inference_service import InferenceService
from ocp_resources.resource import ResourceEditor

from utilities.constants import Annotations


@pytest.fixture(scope="function")
def patched_raw_inference_service_stop_annotation(
    request: pytest.FixtureRequest,
    ovms_raw_inference_service: InferenceService,
) -> Generator[InferenceService, Any, Any]:
    with ResourceEditor(
        patches={
            ovms_raw_inference_service: {
                "metadata": {
                    "annotations": {Annotations.KserveIo.FORCE_STOP_RUNTIME: request.param.get("stop", "false")}
                },
            }
        }
    ):
        yield ovms_raw_inference_service
