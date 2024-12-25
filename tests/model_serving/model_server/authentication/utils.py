import json
import re
from typing import Optional

from ocp_resources.inference_service import InferenceService
from simple_logger.logger import get_logger

from utilities.inference_utils import LlmInference

LOGGER = get_logger(name=__name__)


class InferenceResponseError(Exception):
    pass


def verify_inference_response(
    inference_service: InferenceService,
    runtime: str,
    inference_type: str,
    protocol: str,
    model_name: str,
    text: Optional[str] = None,
    use_default_query: bool = False,
    expected_response_text: Optional[str] = None,
    insecure: bool = True,
    token: Optional[str] = None,
    authorized_user: Optional[bool] = None,
) -> None:
    inference = LlmInference(
        inference_service=inference_service,
        runtime=runtime,
        inference_type=inference_type,
        protocol=protocol,
    )

    res = inference.run_inference(
        model_name=model_name,
        text=text,
        use_default_query=use_default_query,
        token=token,
        insecure=insecure,
    )

    if authorized_user is False:
        auth_header = "x-ext-auth-reason"

        if auth_reason := re.search(rf"{auth_header}: (.*)", res["output"], re.MULTILINE):
            reason = auth_reason.group(1).lower()

            if token:
                assert re.search(r"not (?:authenticated|authorized)", reason)

            else:
                assert "credential not found" in reason

        else:
            raise ValueError(f"Auth header {auth_header} not found in response. Response: {res['output']}")

    else:
        if use_default_query:
            expected_response_text = inference.inference_config["default_query_model"]["model"].get("response_output")
            if not expected_response_text:
                raise ValueError(f"Missing response text key for inference {runtime}")

        if inference.inference_response_text_key_name:
            if inference_type == inference.STREAMING:
                if output := re.findall(
                    rf"{inference.inference_response_text_key_name}\": \"(.*)\"",
                    res["output"],
                    re.MULTILINE,
                ):
                    assert "".join(output) == expected_response_text

            elif inference_type == inference.INFER:
                assert json.dumps(res["output"]).replace(" ", "") == expected_response_text

            else:
                assert res["output"][inference.inference_response_text_key_name] == expected_response_text

        else:
            raise InferenceResponseError(f"Inference response output not found in response. Response: {res}")
