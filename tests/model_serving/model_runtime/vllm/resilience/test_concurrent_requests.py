"""TC-RES-003: Concurrent inference requests to a vLLM ISVC.

Validates that a vLLM model server handles N concurrent requests without
returning 502 Bad Gateway or other proxy errors.
"""

import concurrent.futures
from typing import Any

import pytest
import requests
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.vllm.constant import BASE_RAW_DEPLOYMENT_CONFIG
from utilities.constants import Timeout
from utilities.inference_utils import get_exposed_isvc_url

pytestmark = pytest.mark.usefixtures("valid_aws_config")

CONCURRENT_REQUEST_COUNT = 5

SIMPLE_CHAT_QUERY: dict[str, Any] = {
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 10,
}


def _send_single_request(url: str, model_name: str) -> tuple[int, str]:
    query = {**SIMPLE_CHAT_QUERY, "model": model_name}
    try:
        resp = requests.post(f"{url}/v1/chat/completions", json=query, verify=False, timeout=Timeout.TIMEOUT_2MIN)
        return resp.status_code, resp.text[:200]
    except Exception as exc:
        return 0, str(exc)[:200]


@pytest.mark.tier3
@pytest.mark.resilience
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, resilience_serving_runtime, resilience_inference_service",
    [
        pytest.param(
            {"name": "vllm-res-concurrent"},
            {"model-dir": "vllm/granite-7b-lab"},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {
                "name": "vllm-concurrent",
                **BASE_RAW_DEPLOYMENT_CONFIG,
                "gpu_count": 1,
                "runtime_argument": [
                    "--model=/mnt/models",
                    "--dtype=float16",
                ],
            },
            id="vllm-concurrent-requests",
        ),
    ],
    indirect=True,
)
class TestConcurrentRequests:
    def test_no_502_under_concurrent_load(
        self,
        resilience_inference_service: InferenceService,
    ) -> None:
        """Given a running vLLM ISVC,
        When N concurrent chat completion requests are sent simultaneously,
        Then none of them should return 502 Bad Gateway.
        """
        url = get_exposed_isvc_url(isvc=resilience_inference_service)
        model_name = resilience_inference_service.name

        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_REQUEST_COUNT) as executor:
            futures = [executor.submit(_send_single_request, url, model_name) for _ in range(CONCURRENT_REQUEST_COUNT)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        bad_gateway_responses = [(code, body) for code, body in results if code == 502]
        failed_responses = [(code, body) for code, body in results if code == 0]

        assert not bad_gateway_responses, (
            f"{len(bad_gateway_responses)} of {CONCURRENT_REQUEST_COUNT} requests got 502 Bad Gateway"
        )
        assert not failed_responses, (
            f"{len(failed_responses)} of {CONCURRENT_REQUEST_COUNT} requests failed with connection errors"
        )

        success_count = sum(1 for code, _ in results if 200 <= code < 300)
        assert success_count > 0, (
            f"No successful responses out of {CONCURRENT_REQUEST_COUNT}. Status codes: {[code for code, _ in results]}"
        )
