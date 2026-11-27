import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_serving.model_runtime.mlserver.utils import run_mlserver_inference
from tests.model_serving.model_server.kserve.storage.constants import (
    ISVC_URI_MODEL_URI,
    ISVC_URI_ONNX_REST_INPUT_QUERY,
)
from tests.model_serving.model_server.kserve.storage.utils import assert_isvc_uri_injected
from utilities.constants import ModelFormat, Protocols, Timeout
from utilities.inference_utils import create_isvc

pytestmark = [pytest.mark.tier1, pytest.mark.skip_on_disconnected]


@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [pytest.param({"name": "storage-uri-connection"})],
    indirect=True,
)
class TestUriConnection:
    """Manual 1.2: a `uri`-typed ConnectionsAPI connection Secret injects `storageUri` on CREATE."""

    def test_uri_create_injects_storage_uri(
        self,
        admin_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        kserve_storage_mlserver_runtime: ServingRuntime,
        uri_connection_secret: Secret,
    ) -> None:
        """Test steps:

        1. Create an InferenceService with `opendatahub.io/connections` set to the URI secret.
        2. Assert the webhook injected `predictor.model.storageUri`.
        3. Send a v2 REST inference request and assert a successful, non-empty response — proving
           the model actually loaded from the injected source, not merely that Ready was reached.
        """
        with create_isvc(
            client=admin_client,
            name="isvc-uri-connection",
            namespace=unprivileged_model_namespace.name,
            model_format=ModelFormat.ONNX,
            runtime=kserve_storage_mlserver_runtime.name,
            connections=uri_connection_secret.name,
            timeout=Timeout.TIMEOUT_10MIN,
        ) as isvc:
            assert_isvc_uri_injected(isvc=isvc, expected_uri=ISVC_URI_MODEL_URI)

            response = run_mlserver_inference(
                isvc=isvc, input_data=ISVC_URI_ONNX_REST_INPUT_QUERY, model_version="", protocol=Protocols.REST
            )
            outputs = response.get("outputs") if isinstance(response, dict) else None
            assert outputs, f"Expected non-empty 'outputs' in inference response, got: {response}"
