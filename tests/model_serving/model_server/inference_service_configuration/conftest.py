from typing import Any, Generator

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod

from tests.model_serving.model_server.inference_service_configuration.constants import (
    ISVC_ENV_VARS,
)
from tests.model_serving.model_server.inference_service_configuration.utils import (
    update_inference_service,
)
from utilities.infra import get_pods_by_isvc_label


@pytest.fixture(scope="class")
def removed_isvc_env_vars(
    request: pytest.FixtureRequest,
    unprivileged_client: DynamicClient,
    ovms_kserve_inference_service: InferenceService,
) -> Generator[InferenceService, Any, Any]:
    if isvc_predictor_spec_model_env := ovms_kserve_inference_service.instance.spec.predictor.model.get("env"):
        isvc_predictor_spec_model_env = [
            env_var for env_var in isvc_predictor_spec_model_env if env_var.to_dict() not in ISVC_ENV_VARS
        ]

        with update_inference_service(
            client=unprivileged_client,
            isvc=ovms_kserve_inference_service,
            isvc_updated_dict={"spec": {"predictor": {"model": {"env": isvc_predictor_spec_model_env}}}},
        ):
            yield ovms_kserve_inference_service

    else:
        raise ValueError(
            f"Inference service {ovms_kserve_inference_service.name} does not have env vars in predictor spec model."
        )


@pytest.fixture
def isvc_pods(
    unprivileged_client: DynamicClient, ovms_kserve_inference_service: InferenceService
) -> Generator[list[Pod], Any, Any]:
    yield get_pods_by_isvc_label(client=unprivileged_client, isvc=ovms_kserve_inference_service)


@pytest.fixture(scope="class")
def patched_isvc_replicas(
    request: pytest.FixtureRequest,
    unprivileged_client: DynamicClient,
    ovms_kserve_inference_service: InferenceService,
) -> Generator[InferenceService, Any, Any]:
    with update_inference_service(
        client=unprivileged_client,
        isvc=ovms_kserve_inference_service,
        isvc_updated_dict={
            "spec": {
                "predictor": {
                    "maxReplicas": request.param["max-replicas"],
                    "minReplicas": request.param["min-replicas"],
                }
            }
        },
        wait_for_new_pods=request.param["wait-for-new-pods"],
    ):
        yield ovms_kserve_inference_service
