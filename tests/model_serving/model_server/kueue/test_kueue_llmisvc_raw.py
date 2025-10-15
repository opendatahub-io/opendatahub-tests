import pytest
from timeout_sampler import TimeoutExpiredError, TimeoutSampler
from ocp_resources.deployment import Deployment

from tests.model_serving.model_server.llmd.utils import (
    verify_gateway_status,
    verify_llm_service_status,
)
from utilities.kueue_utils import check_gated_pods_and_running_pods
from utilities.llmd_utils import verify_inference_response_llmd
from utilities.manifests.tinyllama import TINYLLAMA_INFERENCE_CONFIG
from utilities.constants import Protocols

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
CPU_QUOTA = "2"
MEMORY_QUOTA = "6Gi"
LLMISVC_RESOURCES = {
    "requests": {"cpu": "1", "memory": "4Gi"},
    "limits": {"cpu": CPU_QUOTA, "memory": MEMORY_QUOTA},
}

# min_replicas needs to be 1 or you need to change the test to check for the number of
# available replicas
MIN_REPLICAS = 1
MAX_REPLICAS = 2
EXPECTED_RUNNING_PODS = 1
EXPECTED_GATED_PODS = 1
EXPECTED_DEPLOYMENTS = 1
EXPECTED_INITIAL_REPLICAS = 1
EXPECTED_UPDATED_REPLICAS = 2

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
                "min-replicas": MIN_REPLICAS,
                "max-replicas": MAX_REPLICAS,
                "labels": {"kueue.x-k8s.io/queue-name": LOCAL_QUEUE_NAME},
                "resources": LLMISVC_RESOURCES,
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

    def test_kueue_llmd_scaleup(
        self,
        unprivileged_client,
        unprivileged_model_namespace,
        llmd_gateway,
        llmd_inference_service,
        kueue_local_queue_from_template,
    ):
        """
        Verify that Kueue admits the first replica of an LLMInferenceService and
        gates the second replica when the service is scaled up beyond the queue's quota.
        """
        # The llmd_inference_service is already created by the fixture with 1 replica.
        # Wait for the service and its single pod to become ready.
        assert verify_gateway_status(llmd_gateway), "Gateway should be ready"
        assert verify_llm_service_status(
            llmd_inference_service, timeout=600
        ), "LLMInferenceService should be ready"
        
        selector_labels = [f"app.kubernetes.io/name={llmd_inference_service.name}"]
        deployments = list(
            Deployment.get(
                label_selector=",".join(selector_labels),
                namespace=llmd_inference_service.namespace,
                dyn_client=unprivileged_client,
            )
        )
        assert len(deployments) == EXPECTED_DEPLOYMENTS, (
            f"Expected {EXPECTED_DEPLOYMENTS} deployment, got {len(deployments)}"
        )

        deployment = deployments[0]
        deployment.wait_for_replicas(deployed=True)
        replicas = deployment.instance.spec.replicas
        assert replicas == EXPECTED_INITIAL_REPLICAS, (
            f"Deployment should have {EXPECTED_INITIAL_REPLICAS} replica, got {replicas}"
        )

        # Update the LLMInferenceService to request 2 replicas, which exceeds the quota.
        isvc_to_update = llmd_inference_service.instance.to_dict()
        isvc_to_update["spec"]["replicas"] = EXPECTED_UPDATED_REPLICAS
        llmd_inference_service.update(isvc_to_update)

        # Check the deployment until it has 2 replicas, which means it's been updated
        for replicas in TimeoutSampler(
            wait_timeout=30,
            sleep=2,
            func=lambda: self._get_deployment_status_replicas(deployment),
        ):
            if replicas == EXPECTED_UPDATED_REPLICAS:
                break

        # Verify that Kueue correctly gates the second pod.
        
        try:
            for running_pods, gated_pods in TimeoutSampler(
                wait_timeout=120,
                sleep=5,
                func=lambda: check_gated_pods_and_running_pods(
                    selector_labels, unprivileged_model_namespace.name, unprivileged_client
                ),
            ):
                if (
                    running_pods == EXPECTED_RUNNING_PODS
                    and gated_pods == EXPECTED_GATED_PODS
                ):
                    break
        except TimeoutExpiredError:
            running_pods, gated_pods = check_gated_pods_and_running_pods(
                selector_labels, unprivileged_model_namespace.name, unprivileged_client
            )
            assert False, (
                f"Timeout: Expected {EXPECTED_RUNNING_PODS} running and {EXPECTED_GATED_PODS} gated pods. "
                f"Found {running_pods} running and {gated_pods} gated."
            )

        # Refresh the llmisvc instance to get latest status
        llmd_inference_service.get()
        llmisvc = llmd_inference_service.instance
        total_copies = llmisvc.status.modelStatus.copies.totalCopies
        assert total_copies == EXPECTED_RUNNING_PODS, (
            f"InferenceService should have {EXPECTED_RUNNING_PODS} total model copy, got {total_copies}"
        )

        # Verify that inference still works on the single running pod.
        verify_inference_response_llmd(
            llm_service=llmd_inference_service,
            inference_config=TINYLLAMA_INFERENCE_CONFIG,
            inference_type="chat_completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
            insecure=True,
            model_name=llmd_inference_service.name,
        )
