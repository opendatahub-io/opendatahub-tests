import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService

from utilities.infra import get_pods_by_isvc_label

pytestmark = pytest.mark.usefixtures("valid_aws_config")

MANAGED_BY_LABEL_KEY = "app.kubernetes.io/managed-by"
MANAGED_BY_LABEL_VALUE = "kserve-controller"


@pytest.mark.smoke
@pytest.mark.tier2
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "unprivileged_model_namespace, dummy_ovms_raw_inference_service",
    [
        pytest.param(
            {"name": "test-kserve-managed-by-label"},
            {"model-dir": "test-dir"},
        )
    ],
    indirect=True,
)
class TestManagedByLabel:
    """Validate that KServe controller sets the managed-by label on ISVC pods.

    Preconditions:
        - InferenceService deployed and ready in raw deployment mode

    Test Steps:
        1. Deploy an OVMS inference service using raw deployment mode.
        2. Wait for the InferenceService to reach Ready state.
        3. List all pods associated with the InferenceService.
        4. Check each pod for the app.kubernetes.io/managed-by label.

    Expected Results:
        - All ISVC pods carry the label app.kubernetes.io/managed-by=kserve-controller.
    """

    @pytest.mark.smoke
    def test_pods_have_managed_by_label(
        self,
        admin_client: DynamicClient,
        dummy_ovms_raw_inference_service: InferenceService,
    ) -> None:
        """Verify that all ISVC predictor pods carry the managed-by label.

        Given an InferenceService is deployed and ready
        When listing the pods managed by the InferenceService
        Then every pod should have label app.kubernetes.io/managed-by=kserve-controller
        """
        pods = get_pods_by_isvc_label(
            client=admin_client,
            isvc=dummy_ovms_raw_inference_service,
        )

        assert pods, f"No pods found for InferenceService {dummy_ovms_raw_inference_service.name}"

        for pod in pods:
            pod_labels = pod.instance.metadata.labels or {}
            assert MANAGED_BY_LABEL_KEY in pod_labels, (
                f"Pod {pod.name} is missing label '{MANAGED_BY_LABEL_KEY}'. Existing labels: {dict(pod_labels)}"
            )
            assert pod_labels[MANAGED_BY_LABEL_KEY] == MANAGED_BY_LABEL_VALUE, (
                f"Pod {pod.name} has '{MANAGED_BY_LABEL_KEY}={pod_labels[MANAGED_BY_LABEL_KEY]}', "
                f"expected '{MANAGED_BY_LABEL_VALUE}'"
            )
