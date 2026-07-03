import re

import pytest
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.llm_inference_service import LLMInferenceService
from packaging.version import Version

from tests.model_serving.model_server.llmd.llmd_configs import Fast1Config, Fast2Config
from tests.model_serving.model_server.llmd.utils import (
    get_vllm_version,
    ns_from_file,
    parse_completion_text,
    send_chat_completions,
    workaround_503_no_healthy_upstream,
)

pytestmark = [pytest.mark.tier1, pytest.mark.llmd_gpu]

NAMESPACE = ns_from_file(file=__file__)

# Fast image LLMInferenceServiceConfig CRs are available from RHOAI 3.5.1 onwards
FAST_IMAGE_MIN_VERSION = Version(version="3.5.1")


@pytest.fixture(scope="class")
def skip_if_fast_images_unsupported(admin_client) -> None:
    """Skip on RHOAI versions that don't ship fast image LLMInferenceServiceConfig CRs."""
    for csv in ClusterServiceVersion.get(client=admin_client, namespace="redhat-ods-operator"):
        if csv.name.startswith("rhods-operator") and csv.status == csv.Status.SUCCEEDED:
            raw_version = csv.instance.spec.version
            # strip EA suffix for comparison: "3.5.0-ea.2" → "3.5.0"
            rhoai_version = Version(version=raw_version.split("-ea")[0])
            if rhoai_version < FAST_IMAGE_MIN_VERSION:
                pytest.skip(f"Fast image CRs require RHOAI >= {FAST_IMAGE_MIN_VERSION}, found {raw_version}")
            return
    pytest.skip("RHOAI CSV not found")


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [
        pytest.param({"name": NAMESPACE}, Fast1Config, id="fast_1"),
        pytest.param({"name": NAMESPACE}, Fast2Config, id="fast_2"),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("skip_if_disconnected", "skip_if_fast_images_unsupported")
class TestLlmdFastImage:
    """Deploy TinyLlama using fast-1 and fast-2 LLMInferenceServiceConfig and verify inference."""

    def test_inference(self, llmisvc: LLMInferenceService) -> None:
        prompt = "What is the capital of Italy?"
        expected = "rome"

        workaround_503_no_healthy_upstream(llmisvc=llmisvc, prompt=prompt)

        status, body = send_chat_completions(llmisvc=llmisvc, prompt=prompt)
        assert status == 200, f"Expected 200, got {status}: {body}"
        completion = parse_completion_text(response_body=body)
        assert expected in completion.lower(), f"Expected '{expected}' in response, got: {completion}"

    def test_vllm_version(self, llmisvc: LLMInferenceService) -> None:
        version = get_vllm_version(llmisvc=llmisvc)
        assert version, "Expected a non-empty vLLM version string from /version endpoint"
        assert re.fullmatch(r"\d+\.\d+\.\d+(?:-[a-zA-Z0-9.]+)?(?:\+[a-zA-Z0-9.]+)?", version), (
            f"vLLM version '{version}' does not match expected semver format (e.g. '0.8.5')"
        )
