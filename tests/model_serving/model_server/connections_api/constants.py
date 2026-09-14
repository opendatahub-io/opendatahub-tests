"""Constants for the ConnectionsAPI end-to-end test suite.

Holds the ConnectionsAPI annotation keys, the odh-model-controller webhook names used by the
platform guard, and the model reference strings the tests reuse (all point at the repo's
existing smallest model artifacts — see the "Test Data Requirements" section of
``ai_specs/20260907_connections_api_e2e_test.md``).
"""

from typing import Any

from utilities.constants import ApiGroups, ModelFormat, ModelStorage
from utilities.image_constants import SharedImages

# ---------------------------------------------------------------------------
# ConnectionsAPI annotation keys
# ---------------------------------------------------------------------------
CONNECTIONS_ANNOTATION: str = f"{ApiGroups.OPENDATAHUB_IO}/connections"
CONNECTION_PATH_ANNOTATION: str = f"{ApiGroups.OPENDATAHUB_IO}/connection-path"
CONNECTION_TYPE_PROTOCOL_ANNOTATION: str = f"{ApiGroups.OPENDATAHUB_IO}/connection-type-protocol"

# ---------------------------------------------------------------------------
# odh-model-controller ConnectionsAPI webhook names (post RHOAIENG-62537 migration)
# ---------------------------------------------------------------------------
ISVC_CONNECTIONS_WEBHOOK: str = "minferenceservice-v1beta1.odh-model-controller.opendatahub.io"
# LLMInferenceService is versioned (v1alpha1/v1alpha2); the wrapper resolves to the CRD's storage
# version (v1alpha2), so that is the webhook entry actually exercised by this suite's CREATE/UPDATE
# calls.
LLMISVC_CONNECTIONS_WEBHOOK: str = "connection-llmisvc-v1alpha2.odh-model-controller.opendatahub.io"

# Old opendatahub-operator ConnectionsAPI webhooks — must no longer be present.
STALE_ISVC_CONNECTIONS_WEBHOOK: str = "platform-connection-isvc"
STALE_LLMISVC_CONNECTIONS_WEBHOOK: str = "platform-connection-llmisvc"

# ---------------------------------------------------------------------------
# Namespaces (one per module, per AGENTS.md isolation convention)
# ---------------------------------------------------------------------------
ISVC_NAMESPACE: str = "connections-api-isvc"
LLMISVC_NAMESPACE: str = "connections-api-llmisvc"
SMOKE_ISVC_NAMESPACE: str = "connections-api-smoke-isvc"
SMOKE_LLMISVC_NAMESPACE: str = "connections-api-smoke-llmisvc"

# ---------------------------------------------------------------------------
# ISVC test data — MLServer sklearn/onnx iris-scale models (see mlserver runtime tests)
# ---------------------------------------------------------------------------
ISVC_S3_CONNECTION_PATH: str = "mlserver/model_repository/sklearn"
ISVC_OCI_STORAGE_URI: str = SharedImages.MLSERVER_SKLEARN
ISVC_URI_MODEL_URI: str = "hf://dbasunag/onnx-test-model"
# Input schema for the `dbasunag/onnx-test-model` model above (confirmed via its v2 metadata
# endpoint: {"inputs": [{"name": "dense_input", "shape": [-1, 5], "datatype": "FP32"}]}). This is
# a different model than the sibling mlserver suite's own onnx test model, so it does not share
# that suite's `ONNX_REST_INPUT_QUERY` (a different input name/shape) — reusing it 400s.
ISVC_URI_ONNX_REST_INPUT_QUERY: dict[str, Any] = {
    "id": "onnx-test-model",
    "inputs": [
        {
            "name": "dense_input",
            "shape": [1, 5],
            "datatype": "FP32",
            "data": [[0.1, 0.2, 0.3, 0.4, 0.5]],
        }
    ],
}
ISVC_S3_MODEL_FORMAT: str = ModelFormat.SKLEARN
ISVC_OCI_MODEL_FORMAT: str = ModelFormat.SKLEARN
ISVC_URI_MODEL_FORMAT: str = ModelFormat.ONNX

# ---------------------------------------------------------------------------
# LLMISVC test data — TinyLlama-1.1B, uniform across S3/OCI/hf:// (CPU vLLM, no GPU)
# ---------------------------------------------------------------------------
LLMISVC_S3_CONNECTION_PATH: str = "TinyLlama-1.1B-Chat-v1.0"
LLMISVC_OCI_MODEL_URI: str = SharedImages.OCI_TINYLLAMA
LLMISVC_URI_MODEL_URI: str = ModelStorage.HuggingFace.TINYLLAMA
LLMISVC_PLACEHOLDER_MODEL_URI: str = "placeholder"

# ---------------------------------------------------------------------------
# Secret / resource names
# ---------------------------------------------------------------------------
S3_CONNECTION_SECRET_NAME: str = "connections-api-s3-secret"
URI_CONNECTION_SECRET_NAME: str = "connections-api-uri-secret"
OCI_CONNECTION_SECRET_NAME: str = "connections-api-oci-secret"

# Chat prompt reused for LLMISVC functional inference checks.
LLMISVC_CHAT_PROMPT: str = "What is the capital of France?"
