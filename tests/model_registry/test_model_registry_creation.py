from typing import Self
from simple_logger.logger import get_logger
from ocp_resources.model_registry import ModelRegistry

LOGGER = get_logger(name=__name__)


class TestModelRegistryCreation:
    """Tests the creation of a model registry"""

    #Enable Model Registry in DSC


    #Deploy mysql DB in rhoai-model-registries
    #Create MR instance
    def test_model_registry_instance_creation(self: Self, model_registry_instance: ModelRegistry):
        assert model_registry_instance.name == "model-registry"


    #Register a Model
    #Query Model
