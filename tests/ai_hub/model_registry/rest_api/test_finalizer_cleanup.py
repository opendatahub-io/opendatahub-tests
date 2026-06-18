from typing import Self

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from timeout_sampler import TimeoutSampler

from tests.ai_hub.model_registry.rest_api.constants import MODEL_REGISTER_DATA
from utilities.resources.model_registry_modelregistry_opendatahub_io import ModelRegistry

LOGGER = structlog.get_logger(name=__name__)

MODEL_REGISTRY_FINALIZER: str = "modelregistry.opendatahub.io/finalizer"
FINALIZER_CLEANUP_TIMEOUT: int = 120


@pytest.mark.parametrize(
    "model_registry_metadata_db_resources, model_registry_instance, registered_model_rest_api",
    [
        pytest.param(
            {"db_name": "postgres"},
            {"db_name": "postgres"},
            MODEL_REGISTER_DATA,
            id="test_finalizer_cleanup_after_model_registry_deleted",
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures(
    "updated_dsc_component_state_scope_session",
    "model_registry_namespace",
    "model_registry_metadata_db_resources",
    "model_registry_instance",
    "registered_model_rest_api",
    "model_registry_deployment_ns",
    "model_registry_connection_secret",
    "model_registry_serving_runtime",
    "model_registry_linked_inference_service",
)
class TestModelRegistryFinalizerCleanup:
    """Tests for ModelRegistry finalizer behavior during out-of-order resource deletion."""

    @pytest.mark.tier1
    def test_inference_service_deletion_after_model_registry_deleted(
        self: Self,
        admin_client: DynamicClient,
        model_registry_instance: list[ModelRegistry],
        model_registry_linked_inference_service: InferenceService,
        model_registry_deployment_ns: Namespace,
    ) -> None:
        """
        Given a ModelRegistry with a deployed InferenceService referencing it
        When the ModelRegistry CR is deleted before the InferenceService
        Then the InferenceService finalizer is removed and deletion completes
        """
        model_registry = model_registry_instance[0]
        inference_service_name = model_registry_linked_inference_service.name
        inference_service_namespace = model_registry_deployment_ns.name

        finalizers = model_registry_linked_inference_service.instance.metadata.get("finalizers", [])
        assert MODEL_REGISTRY_FINALIZER in finalizers, (
            f"InferenceService '{inference_service_name}' missing finalizer '{MODEL_REGISTRY_FINALIZER}', "
            f"found: {finalizers}"
        )

        LOGGER.info(f"Deleting ModelRegistry '{model_registry.name}' while InferenceService exists")
        model_registry.delete(wait=True)
        LOGGER.info(f"Deleting InferenceService '{inference_service_name}' after ModelRegistry is gone")
        model_registry_linked_inference_service.delete()

        for sample in TimeoutSampler(
            wait_timeout=FINALIZER_CLEANUP_TIMEOUT,
            sleep=5,
            func=lambda: (
                not InferenceService(
                    client=admin_client,
                    name=inference_service_name,
                    namespace=inference_service_namespace,
                ).exists
            ),
        ):
            if sample:
                LOGGER.info(f"InferenceService '{inference_service_name}' deleted successfully")
                break
