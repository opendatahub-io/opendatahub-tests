import json

import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import Protocols, RuntimeTemplates
from utilities.inference_utils import Inference

# Sentiment transformer v1 predict inference config.
# The transformer accepts text via v1 predict endpoint and returns
# {"predictions": [{"sentiment": "...", "confidence": ..., ...}]}.
# NOTE: query_input is pre-serialised to JSON because the inference
# framework only calls json.dumps() on lists, not dicts.
SENTIMENT_INFERENCE_CONFIG = {
    "default_query_model": {
        "query_input": json.dumps({
            "texts": [
                "This product is amazing! I love it!",
                "Terrible experience, very disappointed.",
            ]
        }),
        "query_output": r'"predictions":\s*\[.*"sentiment"',
        "use_regex": True,
    },
    "infer": {
        "http": {
            "endpoint": "v1/models/$model_name:predict",
            "header": "Content-type:application/json",
            "body": "$query_input",
            "response_fields_map": {
                "response_output": "output",
            },
        },
    },
}

ISVC_NAME = "sentiment-analysis"


@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "unprivileged_model_namespace, transformer_auth_inference_service",
    [
        pytest.param(
            {"name": "test-kserve-transformer-auth"},
            {
                "name": "onnx",
                "template-name": RuntimeTemplates.MLSERVER,
                "multi-model": False,
                "storage-uri": "hf://optimum/distilbert-base-uncased-finetuned-sst-2-english",
            },
        )
    ],
    indirect=True,
)
class TestTransformerAuthEnforcement:
    """Verify kube-rbac-proxy auth enforcement on ISVC with transformer.

    Steps:
        1. Deploy a sentiment model with a custom transformer and authentication enabled.
        2. Query the model without a token and verify the request is rejected (403).
        3. Query the model with a valid token and verify successful inference (200).
    """

    def test_unauthenticated_request_rejected(self, transformer_auth_inference_service):
        """Request without token is rejected by kube-rbac-proxy."""
        verify_inference_response(
            inference_service=transformer_auth_inference_service,
            inference_config=SENTIMENT_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            model_name=ISVC_NAME,
            use_default_query=True,
            authorized_user=False,
        )

    def test_authenticated_request_succeeds(self, transformer_auth_inference_service, transformer_inference_token):
        """Request with valid token succeeds through kube-rbac-proxy to transformer."""
        verify_inference_response(
            inference_service=transformer_auth_inference_service,
            inference_config=SENTIMENT_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            model_name=ISVC_NAME,
            use_default_query=True,
            token=transformer_inference_token,
        )
