import pytest
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="session", autouse=True)
def vllm_skip_if_no_supported_accelerator_type(supported_accelerator_type: str) -> None:  # noqa: UFN001
    if not supported_accelerator_type:
        pytest.skip("Accelerator type is not provided,vLLM test can not be run on CPU")
