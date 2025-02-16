import json
from typing import Any

from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.trustyai_service import TrustyAIService
from requests import Response
from simple_logger.logger import get_logger

from tests.model_explainability.trustyai_service.utils import TrustyAIServiceClient

LOGGER = get_logger(name=__name__)


def verify_shap_explanation(
    client: DynamicClient, token: str, trustyai_service: TrustyAIService, inference_service: InferenceService
) -> None:
    tas_client: TrustyAIServiceClient = TrustyAIServiceClient(token=token, service=trustyai_service, client=client)

    response: Response = tas_client.get_inference_ids(model_name=inference_service.name)
    response_data: Any = json.loads(response.text)

    if len(response_data) < 2:
        raise ValueError(f"Not enough inferences available. Found only {len(response_data)}")

    inference_id: str = response_data[-1]["id"]

    response = tas_client.request_shap_explanation(
        model_name=inference_service.name,
        inference_id=inference_id,
        target="knative-local-gateway.istio-system.svc.cluster.local:80",
        n_samples=75,
    )
    LOGGER.error(response.text)
