import pytest
from huggingface_hub import HfApi


@pytest.fixture()
def huggingface_api():
    return HfApi()
