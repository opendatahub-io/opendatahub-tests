from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from string import Template
from typing import Any, Optional

from ocp_resources.inference_service import InferenceService
from simple_logger.logger import get_logger

from utilities.constants import KServeDeploymentType
from utilities.exceptions import (
    InferenceResponseError,
)
from utilities.inference_utils import UserInference

LOGGER = get_logger(name=__name__)


def verify_inference_response(
    inference_service: InferenceService,
    inference_config: dict[str, Any],
    inference_type: str,
    protocol: str,
    model_name: Optional[str] = None,
    inference_input: Optional[Any] = None,
    use_default_query: bool = False,
    expected_response_text: Optional[str] = None,
    insecure: bool = False,
    token: Optional[str] = None,
    authorized_user: Optional[bool] = None,
) -> None:
    """
    Verify the inference response.

    Args:
        inference_service (InferenceService): Inference service.
        inference_config (dict[str, Any]): Inference config.
        inference_type (str): Inference type.
        protocol (str): Protocol.
        model_name (str): Model name.
        inference_input (Any): Inference input.
        use_default_query (bool): Use default query or not.
        expected_response_text (str): Expected response text.
        insecure (bool): Insecure mode.
        token (str): Token.
        authorized_user (bool): Authorized user.

    Raises:
        InvalidInferenceResponseError: If inference response is invalid.
        ValidationError: If inference response is invalid.

    """
    model_name = model_name or inference_service.name

    inference = UserInference(
        inference_service=inference_service,
        inference_config=inference_config,
        inference_type=inference_type,
        protocol=protocol,
    )

    res = inference.run_inference_flow(
        model_name=model_name,
        inference_input=inference_input,
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

        elif inference.deployment_mode == KServeDeploymentType.MODEL_MESH:
            reason = "Forbidden"
            assert reason in res["output"], f"{reason} not found in output:\n{res['output']}"

        else:
            raise ValueError(f"Auth header {auth_header} not found in response. Response: {res['output']}")

    else:
        use_regex = False

        if use_default_query:
            expected_response_text_config: dict[str, Any] = inference.inference_config.get("default_query_model", {})
            use_regex = expected_response_text_config.get("use_regex", False)

            if not expected_response_text_config:
                raise ValueError(
                    f"Missing default_query_model config for inference {inference_config}. "
                    f"Config: {expected_response_text_config}"
                )

            if inference.inference_config.get("support_multi_default_queries"):
                query_config = expected_response_text_config.get(inference_type)
                if not query_config:
                    raise ValueError(
                        f"Missing default_query_model config for inference {inference_config}. "
                        f"Config: {expected_response_text_config}"
                    )
                expected_response_text = query_config.get("query_output", "")
                use_regex = query_config.get("use_regex", False)

            else:
                expected_response_text = expected_response_text_config.get("query_output")

            if not expected_response_text:
                raise ValueError(f"Missing response text key for inference {inference_config}")

            if isinstance(expected_response_text, str):
                expected_response_text = Template(expected_response_text).safe_substitute(model_name=model_name)

            elif isinstance(expected_response_text, dict):
                expected_response_text = Template(expected_response_text.get("response_output")).safe_substitute(
                    model_name=model_name
                )

        if inference.inference_response_text_key_name:
            if inference_type == inference.STREAMING:
                if output := re.findall(
                    rf"{inference.inference_response_text_key_name}\": \"(.*)\"",
                    res[inference.inference_response_key_name],
                    re.MULTILINE,
                ):
                    assert "".join(output) == expected_response_text

            elif inference_type == inference.INFER or use_regex:
                formatted_res = json.dumps(res[inference.inference_response_text_key_name]).replace(" ", "")
                if use_regex:
                    assert re.search(expected_response_text, formatted_res)  # type: ignore[arg-type]  # noqa: E501

                else:
                    assert (
                        json.dumps(res[inference.inference_response_key_name]).replace(" ", "")
                        == expected_response_text
                    )

            else:
                response = res[inference.inference_response_key_name]
                if isinstance(response, list):
                    response = response[0]

                assert response[inference.inference_response_text_key_name] == expected_response_text

        else:
            raise InferenceResponseError(f"Inference response output not found in response. Response: {res}")


def run_inference_multiple_times(
    isvc: InferenceService,
    inference_config: dict[str, Any],
    inference_type: str,
    protocol: str,
    model_name: str,
    iterations: int,
    run_in_parallel: bool = False,
) -> None:
    """
    Run inference multiple times.

    Args:
        isvc (InferenceService): Inference service.
        inference_config (dict[str, Any]): Inference config.
        inference_type (str): Inference type.
        protocol (str): Protocol.
        model_name (str): Model name.
        iterations (int): Number of iterations.
        run_in_parallel (bool, optional): Run inference in parallel.

    """
    futures = []

    with ThreadPoolExecutor() as executor:
        for iteration in range(iterations):
            infer_kwargs = {
                "inference_service": isvc,
                "inference_config": inference_config,
                "inference_type": inference_type,
                "protocol": protocol,
                "model_name": model_name,
                "use_default_query": True,
            }

            if run_in_parallel:
                futures.append(executor.submit(verify_inference_response, **infer_kwargs))
            else:
                verify_inference_response(**infer_kwargs)

        if futures:
            exceptions = []
            for result in as_completed(futures):
                if _exception := result.exception():
                    exceptions.append(_exception)

            if exceptions:
                raise InferenceResponseError(f"Failed to run inference. Error: {exceptions}")
