import pytest

from utilities.inference_utils import INFERENCE_QUERIES, Inference

pytestmark = pytest.mark.usefixtures("skip_if_no_authorino_operator", "valid_aws_config")


CAIKIT_STR: str = "caikit"
CAIKIT_TGIS_RUNTIME_STR: str = f"{CAIKIT_STR}-tgis-runtime"
INFERENCE_QUERY = INFERENCE_QUERIES["nitrogen-boil-temp"]


@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, s3_serving_runtime, s3_inference_service",
    [
        pytest.param(
            {"name": "kserve-token-authentication"},
            {"model-dir": f"flan-t5-small/flan-t5-small-{CAIKIT_STR}"},
            {
                "name": CAIKIT_TGIS_RUNTIME_STR,
                "model-name": CAIKIT_STR,
                "template-name": "caikit-tgis-serving-template",
                "multi-model": False,
            },
            {
                "name": CAIKIT_STR,
                "deployment-mode": "Serverless",
                "enable-model-auth": True,
            },
        )
    ],
    indirect=True,
)
class TestKserveTokenAuthentication:
    def test_model_authentication_using_rest(self, s3_inference_service, inference_token):
        inference = Inference(
            inference_service=s3_inference_service,
            runtime=CAIKIT_TGIS_RUNTIME_STR,
            inference_type="all-tokens",
            protocol="http",
        )

        res = inference.run_inference(
            model_name=CAIKIT_STR,
            text=INFERENCE_QUERY["text"],
            token=inference_token,
            insecure=True,
        )

        assert res[inference.inference_response_text_key_name] == INFERENCE_QUERY["response_text"]

    # def test_model_authentication_using_grpc(self):
    #     pass
    #
    # def test_block_cross_model_authentication(self):
    #     pass
    #
    # def test_disabled_model_authentication(self):
    #     pass
    #
    # def test_re_enabled_model_authentication(self):
    #     pass
    #
    # def test_model_authentication_using_invalid_token(self):
    #   pass
