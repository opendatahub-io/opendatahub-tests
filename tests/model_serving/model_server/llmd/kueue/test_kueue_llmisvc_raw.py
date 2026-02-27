import pytest
from ocp_resources.deployment import Deployment
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_serving.model_server.llmd.utils import (
    verify_gateway_status,
    verify_llm_service_status,
)
from utilities.constants import Labels, Protocols
from utilities.exceptions import UnexpectedResourceCountError
from utilities.kueue_utils import check_gated_pods_and_running_pods
from utilities.llmd_utils import verify_inference_response_llmd
from utilities.manifests.tinyllama import TINYLLAMA_INFERENCE_CONFIG

pytestmark = [
    pytest.mark.rawdeployment,
    pytest.mark.llmd_cpu,
    pytest.mark.kueue,
    pytest.mark.smoke,
]

# --- Test Configuration ---
NAMESPACE_NAME = "test-kueue-llmd-raw"
LOCAL_QUEUE_NAME = "llmd-local-queue-raw"
CLUSTER_QUEUE_NAME = "llmd-cluster-queue-raw"
RESOURCE_FLAVOR_NAME = "llmd-flavor-raw"

# Set a quota sufficient for only ONE model to run
CPU_QUOTA = "3"
MEMORY_QUOTA = "20Gi"
LLMISVC_RESOURCES = {
    "requests": {"cpu": "2", "memory": "6Gi"},
    "limits": {"cpu": CPU_QUOTA, "memory": MEMORY_QUOTA},
}

# INITIAL_REPLICAS needs to be 1 or you need to change the test to check for the number of
# available replicas
INITIAL_REPLICAS = 1
EXPECTED_UPDATED_REPLICAS = 2
EXPECTED_DEPLOYMENTS = 1

# We will create two replicas, so we expect 1 to be admitted (running) and 1 to be gated (pending)
EXPECTED_RUNNING_PODS = 1
EXPECTED_GATED_PODS = 1


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmd_inference_service, "
    "kueue_cluster_queue_from_template, kueue_resource_flavor_from_template, kueue_local_queue_from_template",
    [
        (
            {"name": NAMESPACE_NAME, "add-kueue-label": True},
            {
                "name": "llmd-kueue-scaleup-test",
                "replicas": INITIAL_REPLICAS,
                "labels": {Labels.Kueue.QUEUE_NAME: LOCAL_QUEUE_NAME},
                "container_resources": LLMISVC_RESOURCES,
            },
            {
                "name": CLUSTER_QUEUE_NAME,
                "resource_flavor_name": RESOURCE_FLAVOR_NAME,
                "cpu_quota": CPU_QUOTA,
                "memory_quota": MEMORY_QUOTA,
            },
            {"name": RESOURCE_FLAVOR_NAME},
            {"name": LOCAL_QUEUE_NAME, "cluster_queue": CLUSTER_QUEUE_NAME},
        )
    ],
    indirect=True,
)
class TestKueueLLMDScaleUp:
    """
    Test Kueue admission control for a single LLMInferenceService that scales up
    to exceed the available resource quota.
    """

    def _get_deployment_status_replicas(self, deployment: Deployment) -> int:
        deployment.get()
        return deployment.instance.status.replicas

    def test_kueue_llmd_scaleup(
        self,
        unprivileged_client,
        kueue_resource_flavor_from_template,
        kueue_cluster_queue_from_template,
        kueue_local_queue_from_template,
        llmd_inference_service,
        llmd_gateway,
    ):
        """
        Verify that Kueue admits the first replica of an LLMInferenceService and
        gates the second replica when the service is scaled up beyond the queue's quota.
        """
        # The llmd_inference_service is created with 1 replica at first to ensure the LLMISVC is ready
        # Wait for the service and its single pod to become ready.
        assert verify_gateway_status(llmd_gateway), "Gateway should be ready"
        assert verify_llm_service_status(llmd_inference_service), "LLMInferenceService should be ready"

        selector_labels = [f"app.kubernetes.io/name={llmd_inference_service.name}", "kserve.io/component=workload"]
        deployments = list(
            Deployment.get(
                label_selector=",".join(selector_labels),
                namespace=llmd_inference_service.namespace,
                client=unprivileged_client,
            )
        )
        assert len(deployments) == EXPECTED_DEPLOYMENTS, (
            f"Expected {EXPECTED_DEPLOYMENTS} deployment, got {len(deployments)}"
        )

        deployment = deployments[0]
        deployment.wait_for_replicas(deployed=True)
        replicas = deployment.instance.spec.replicas
        assert replicas == INITIAL_REPLICAS, f"Deployment should have {INITIAL_REPLICAS} replica, got {replicas}"

        # Update the LLMInferenceService to request 2 replicas, which exceeds the quota.
        isvc_to_update = llmd_inference_service.instance.to_dict()
        isvc_to_update["spec"]["replicas"] = EXPECTED_UPDATED_REPLICAS
        llmd_inference_service.update(isvc_to_update)

        # Check the deployment until it has 2 replicas, which means it's been updated
        try:
            for replicas in TimeoutSampler(
                wait_timeout=60,
                sleep=2,
                func=lambda: self._get_deployment_status_replicas(deployment),
            ):
                if replicas == EXPECTED_UPDATED_REPLICAS:
                    break
        except TimeoutExpiredError:
            raise UnexpectedResourceCountError(
                f"Timeout waiting for deployment to update. "
                f"Expected {EXPECTED_UPDATED_REPLICAS} replicas, found {replicas}."
            ) from None

        # Verify that Kueue correctly gates the second pod.
        try:
            for running_pods, gated_pods in TimeoutSampler(
                wait_timeout=120,
                sleep=5,
                func=lambda: check_gated_pods_and_running_pods(
                    selector_labels, llmd_inference_service.namespace, unprivileged_client
                ),
            ):
                if running_pods == EXPECTED_RUNNING_PODS and gated_pods == EXPECTED_GATED_PODS:
                    break
        except TimeoutExpiredError:
            raise UnexpectedResourceCountError(
                "Timeout: "
                f"Expected {EXPECTED_RUNNING_PODS} running and {EXPECTED_GATED_PODS} gated pods. "
                f"Found {running_pods} running and {gated_pods} gated."
            ) from None

        # Verify that inference still works on the single running pod
        verify_inference_response_llmd(
            llm_service=llmd_inference_service,
            inference_config=TINYLLAMA_INFERENCE_CONFIG,
            inference_type="chat_completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
            insecure=True,
            model_name=llmd_inference_service.name,
        )
