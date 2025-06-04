"""
Integration test for Kueue and InferenceService admission control.
This test imports the reusable test logic from utilities.kueue_utils.
"""

import pytest
from ocp_resources.deployment import Deployment
from ocp_resources.pod import Pod
from timeout_sampler import TimeoutExpiredError
from utilities.exceptions import DeploymentValidationError
from utilities.constants import RunTimeConfigs, KServeDeploymentType, Timeout
from tests.model_serving.model_server.serverless.constants import ONNX_SERVERLESS_INFERENCE_SERVICE_CONFIG
from utilities.general import create_isvc_label_selector_str
import time

pytestmark = [
    pytest.mark.serverless,
    pytest.mark.sanity,
    pytest.mark.usefixtures("valid_aws_config"),
]

NAMESPACE_NAME = "kueue-isvc-serverless-test"
LOCAL_QUEUE_NAME = "local-queue-serverless"
CLUSTER_QUEUE_NAME = "cluster-queue-serverless"
RESOURCE_FLAVOR_NAME = "default-flavor-serverless"
CPU_QUOTA = 2
MEMORY_QUOTA = "10Gi"
ISVC_RESOURCES = {"requests": {"cpu": "1", "memory": "8Gi"}, "limits": {"cpu": CPU_QUOTA, "memory": MEMORY_QUOTA}}
MIN_REPLICAS = (
    1  # min_replicas needs to be 1 or you need to change the test to check for the number of available replicas
)
MAX_REPLICAS = 2


@pytest.mark.serverless
@pytest.mark.parametrize(
    "unprivileged_model_namespace, kueue_kserve_serving_runtime, kueue_kserve_inference_service, "
    "kueue_cluster_queue_from_template, kueue_resource_flavor_from_template, kueue_local_queue_from_template",
    [
        pytest.param(
            {"name": NAMESPACE_NAME, "add-kueue-label": True},
            RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
            {
                **ONNX_SERVERLESS_INFERENCE_SERVICE_CONFIG,
                "name": "kueue",
                "min-replicas": MIN_REPLICAS,
                "max-replicas": MAX_REPLICAS,
                "labels": {"kueue.x-k8s.io/queue-name": LOCAL_QUEUE_NAME},
                "deployment-mode": KServeDeploymentType.SERVERLESS,
                "resources": ISVC_RESOURCES,
            },
            {
                "name": CLUSTER_QUEUE_NAME,
                "resource_flavor_name": RESOURCE_FLAVOR_NAME,
                "cpu_quota": CPU_QUOTA,
                "memory_quota": MEMORY_QUOTA,
                "namespace_selector": {"matchLabels": {"kubernetes.io/metadata.name": NAMESPACE_NAME}},
            },
            {"name": RESOURCE_FLAVOR_NAME},
            {"name": LOCAL_QUEUE_NAME, "cluster_queue": CLUSTER_QUEUE_NAME},
        )
    ],
    indirect=True,
)
class TestKueueInferenceServiceServerless:
    """Test inference service with serverless deployment"""

    def test_kueue_inference_service_serverless(
        self,
        admin_client,
        kueue_resource_flavor_from_template,
        kueue_cluster_queue_from_template,
        kueue_local_queue_from_template,
        kueue_kserve_inference_service,
        kueue_kserve_serving_runtime,
    ):
        """Test inference service with serverless deployment"""
        # Verify initial deployment has 1 pod
        labels = [
            create_isvc_label_selector_str(
                isvc=kueue_kserve_inference_service,
                resource_type="deployment",
                runtime_name=kueue_kserve_serving_runtime.name,
            )
        ]
        deployments = list(
            Deployment.get(
                label_selector=",".join(labels),
                namespace=kueue_kserve_inference_service.namespace,
                dyn_client=admin_client,
            )
        )
        if len(deployments) != 1:
            deployment_names = [deployment.instance.metadata.name for deployment in deployments]
            raise DeploymentValidationError(f"Expected 1 deployment, got {len(deployments)}: {deployment_names}")

        deployment = deployments[0]
        deployment.wait_for_replicas(deployed=True)
        replicas = deployment.instance.spec.replicas
        if replicas != 1:
            raise DeploymentValidationError(f"Deployment should have 1 replica, got {replicas}")

        # Update inference service to request 2 replicas
        isvc_to_update = kueue_kserve_inference_service.instance.to_dict()
        isvc_to_update["spec"]["predictor"]["minReplicas"] = 2
        kueue_kserve_inference_service.update(isvc_to_update)

        # Give time for updated deployment
        time.sleep(10)  # noqa: FCN001
        # Verify deployment still has 1 pod due to Kueue admission control
        deployments = list(
            Deployment.get(
                label_selector=",".join(labels),
                namespace=kueue_kserve_inference_service.namespace,
                dyn_client=admin_client,
            )
        )
        try:
            total_available_replicas = 0
            for deployment in deployments:
                # This will raise TimeoutExpiredError if the deployment is not available
                # If it is available, we will add the number of replicas to the total, since the deployment is available
                # it means spec.replicas == status.replicas == status.updatedReplicas ==
                # status.availableReplicas == status.readyReplicas
                deployment.wait_for_replicas(deployed=True, timeout=Timeout.TIMEOUT_30SEC)
                total_available_replicas += deployment.instance.status.availableReplicas
        except TimeoutExpiredError:
            pass
        if total_available_replicas != 1:
            raise DeploymentValidationError(
                f"Total available replicas across all deployments should be 1, got {total_available_replicas}"
            )
        # Get pods that match isvc labels and verify their status
        pods = list(
            Pod.get(
                label_selector=",".join(labels),
                namespace=kueue_kserve_inference_service.namespace,
                dyn_client=admin_client,
            )
        )

        if len(pods) != 3:
            pod_names = [pod.instance.metadata.name for pod in pods]
            raise DeploymentValidationError(f"Expected 3 pods, got {len(pods)}: {pod_names}")

        running_pods = 0
        gated_pods = 0
        for pod in pods:
            pod_phase = pod.instance.status.phase
            if pod_phase == "Running":
                running_pods += 1
            elif pod_phase == "Pending" and all(
                condition.type == "PodScheduled"
                and condition.status == "False"
                and condition.reason == "SchedulingGated"
                for condition in pod.instance.status.conditions
            ):
                gated_pods += 1

        if running_pods != 1 or gated_pods != 2:
            raise DeploymentValidationError(
                f"Expected 1 Running pod and 2 SchedulingGated pods, "
                f"got {running_pods} Running and {gated_pods} SchedulingGated"
            )
            # Refresh the isvc instance to get latest status
        kueue_kserve_inference_service.get()
        isvc = kueue_kserve_inference_service.instance
        if isvc.status.modelStatus.copies.totalCopies != 1:
            raise DeploymentValidationError(
                f"InferenceService should have 1 total model copy, got {isvc.status.modelStatus.copies.totalCopies}"
            )
