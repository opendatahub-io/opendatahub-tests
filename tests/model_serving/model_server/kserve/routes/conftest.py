from collections.abc import Generator
from typing import Any

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.resource import ResourceEditor
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from utilities.constants import Labels

LOGGER = get_logger(name=__name__)


@pytest.fixture()
def patched_s3_caikit_kserve_isvc_visibility_label(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    s3_models_inference_service: InferenceService,
) -> Generator[InferenceService, Any, Any]:
    visibility = request.param["visibility"]

    labels = s3_models_inference_service.instance.metadata.labels

    # If no label is applied, visibility is "local-cluster"
    if (not labels and visibility == "local-cluster") or (
        labels and labels.get(Labels.Kserve.NETWORKING_KSERVE_IO) == visibility
    ):
        LOGGER.info(f"Inference service visibility is set to {visibility}. Skipping update.")
        yield s3_models_inference_service

    else:
        isvc_orig_url = s3_models_inference_service.instance.status.url

        with ResourceEditor(
            patches={
                s3_models_inference_service: {
                    "metadata": {
                        "labels": {Labels.Kserve.NETWORKING_KSERVE_IO: visibility},
                    }
                }
            }
        ):
            LOGGER.info(f"Wait for inference service {s3_models_inference_service.name} url update")
            for sample in TimeoutSampler(
                wait_timeout=2 * 60,
                sleep=1,
                func=lambda: s3_models_inference_service.instance.status.url,
            ):
                if sample:  # noqa: SIM102
                    if visibility == Labels.Kserve.EXPOSED and isvc_orig_url == sample or sample != isvc_orig_url:
                        break

            yield s3_models_inference_service

        LOGGER.info(f"Wait for inference service {s3_models_inference_service.name} url restore to original one")
        for sample in TimeoutSampler(
            wait_timeout=2 * 60,
            sleep=1,
            func=lambda: s3_models_inference_service.instance.status.url,
        ):
            if sample and sample == isvc_orig_url:
                break
