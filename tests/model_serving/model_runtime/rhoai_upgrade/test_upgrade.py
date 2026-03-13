from collections.abc import Generator
from typing import Any

import pytest
from simple_logger.logger import get_logger

from tests.model_serving.model_runtime.rhoai_upgrade.constant import (
    SERVING_RUNTIME_INSTANCE_NAME,
    SERVING_RUNTIME_TEMPLATE_NAME,
)

LOGGER = get_logger(name=__name__)


class TestServingRuntimeLifecycle:
    """
    Tests to validate the lifecycle of ServingRuntime resources
    including creation, verification, and deletion.
    """

    @pytest.mark.post_upgrade
    @pytest.mark.order(1)
    def test_serving_runtime_template_lifecycle(self, serving_runtime_template: Generator[Any, Any, Any]) -> None:
        assert serving_runtime_template.exists, (
            f"Failed to validate Serving Runtime template lifecycle'{SERVING_RUNTIME_TEMPLATE_NAME}'"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.order(2)
    def test_serving_runtime_instance_lifecycle(self, serving_runtime_instance: Generator[Any, Any, Any]) -> None:
        assert serving_runtime_instance.exists, (
            f"Failed to validate Serving Runtime instance lifecycle'{SERVING_RUNTIME_INSTANCE_NAME}'"
        )
