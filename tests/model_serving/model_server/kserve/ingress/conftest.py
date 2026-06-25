from collections.abc import Generator
from typing import Any

import pytest
import structlog
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from timeout_sampler import TimeoutSampler

from tests.model_serving.model_server.kserve.ingress.utils import create_curl_pod
from utilities.constants import KServeDeploymentType, Labels, ModelStoragePath
from utilities.inference_utils import create_isvc
from utilities.infra import create_ns

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture()
def patched_kserve_isvc_visibility_label(
    request: FixtureRequest,
    ovms_kserve_inference_service: InferenceService,
) -> Generator[InferenceService, Any, Any]:
    param = getattr(request, "param", None)
    if not isinstance(param, dict) or "visibility" not in param:
        raise pytest.UsageError(
            "Fixture 'patched_kserve_isvc_visibility_label' requires indirect "
            "parametrization with request.param['visibility']."
        )

    visibility = param["visibility"]

    labels = ovms_kserve_inference_service.instance.metadata.labels
    current_visibility = labels.get(Labels.Kserve.NETWORKING_KSERVE_IO) if labels else None

    if (current_visibility is None and visibility == "local-cluster") or current_visibility == visibility:
        LOGGER.info(f"Inference service visibility is set to {visibility}. Skipping update.")
        yield ovms_kserve_inference_service

    else:
        isvc_orig_url = ovms_kserve_inference_service.instance.status.url

        with ResourceEditor(
            patches={
                ovms_kserve_inference_service: {
                    "metadata": {
                        "labels": {Labels.Kserve.NETWORKING_KSERVE_IO: visibility},
                    }
                }
            }
        ):
            LOGGER.info(f"Wait for inference service {ovms_kserve_inference_service.name} url update")
            for sample in TimeoutSampler(
                wait_timeout=2 * 60,
                sleep=1,
                func=lambda: ovms_kserve_inference_service.instance.status.url,
            ):
                if sample and sample != isvc_orig_url:
                    break

            yield ovms_kserve_inference_service

        LOGGER.info(f"Wait for inference service {ovms_kserve_inference_service.name} url restore to original one")
        for sample in TimeoutSampler(
            wait_timeout=2 * 60,
            sleep=1,
            func=lambda: ovms_kserve_inference_service.instance.status.url,
        ):
            if sample and sample == isvc_orig_url:
                break

        LOGGER.info(f"Wait for inference service {ovms_kserve_inference_service.name} ready after label restore")
        ovms_kserve_inference_service.wait_for_condition(
            condition=ovms_kserve_inference_service.Condition.READY,
            status=ovms_kserve_inference_service.Condition.Status.TRUE,
            timeout=2 * 60,
        )


@pytest.fixture()
def ovms_raw_isvc_patched_annotations(
    request: FixtureRequest,
    ovms_raw_inference_service: InferenceService,
) -> Generator[InferenceService, Any, Any]:
    """Patch ISVC annotations and restore after the test."""
    param = getattr(request, "param", None)
    if param is not None and not isinstance(param, dict):
        raise pytest.UsageError(
            "Fixture 'ovms_raw_isvc_patched_annotations' requires indirect "
            "parametrization with request.param['annotations']."
        )

    annotations = param.get("annotations") if param else None
    if annotations is not None:
        with ResourceEditor(
            patches={
                ovms_raw_inference_service: {
                    "metadata": {"annotations": annotations},
                }
            }
        ):
            yield ovms_raw_inference_service
    else:
        yield ovms_raw_inference_service


@pytest.fixture(scope="class")
def diff_namespace(admin_client: DynamicClient, unprivileged_client: DynamicClient) -> Generator[Namespace, Any, Any]:
    with create_ns(admin_client=admin_client, unprivileged_client=unprivileged_client, name="diff-namespace") as ns:
        yield ns


@pytest.fixture(scope="class")
def endpoint_isvc(
    unprivileged_client: DynamicClient,
    serving_runtime_from_template: ServingRuntime,
    ci_endpoint_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=unprivileged_client,
        name="endpoint-isvc",
        namespace=serving_runtime_from_template.namespace,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        storage_key=ci_endpoint_s3_secret.name,
        storage_path=ModelStoragePath.OPENVINO_EXAMPLE_MODEL,
        model_format=serving_runtime_from_template.instance.spec.supportedModelFormats[0].name,
        runtime=serving_runtime_from_template.name,
        wait_for_predictor_pods=False,
    ) as isvc:
        yield isvc


@pytest.fixture()
def same_namespace_pod(
    unprivileged_client: DynamicClient, unprivileged_model_namespace: Namespace
) -> Generator[Pod, Any, Any]:
    with create_curl_pod(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        pod_name="curl-same-ns",
    ) as pod:
        yield pod


@pytest.fixture()
def diff_namespace_pod(
    unprivileged_client: DynamicClient,
    diff_namespace: Namespace,
) -> Generator[Pod, Any, Any]:
    with create_curl_pod(
        client=unprivileged_client,
        namespace=diff_namespace.name,
        pod_name="curl-diff-ns",
    ) as pod:
        yield pod
