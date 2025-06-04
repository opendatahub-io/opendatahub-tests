"""
Integration test for Kueue and InferenceService admission control.
This test imports the reusable test logic from utilities.kueue_utils.
"""

import time
import pytest
from ocp_resources.deployment import Deployment
from timeout_sampler import TimeoutExpiredError
from utilities.exceptions import DeploymentValidationError
from utilities.constants import RunTimeConfigs, KServeDeploymentType, ModelVersion, Timeout
from utilities.general import create_isvc_label_selector_str
from ocp_resources.pod import Pod


pytestmark = [
    pytest.mark.rawdeployment,
    pytest.mark.sanity,
    pytest.mark.usefixtures("valid_aws_config"),
]

NAMESPACE_NAME = "kueue-isvc-raw-test"
LOCAL_QUEUE_NAME = "local-queue-raw"
CLUSTER_QUEUE_NAME = "cluster-queue-raw"
RESOURCE_FLAVOR_NAME = "default-flavor-raw"
CPU_QUOTA = 2
MEMORY_QUOTA = "10Gi"
ISVC_RESOURCES = {"requests": {"cpu": "1", "memory": "8Gi"}, "limits": {"cpu": CPU_QUOTA, "memory": MEMORY_QUOTA}}
MIN_REPLICAS = (
    1  # min_replicas needs to be 1 or you need to change the test to check for the number of available replicas
)
MAX_REPLICAS = 2


@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "unprivileged_model_namespace, kueue_kserve_serving_runtime, kueue_raw_inference_service, "
    "kueue_cluster_queue_from_template, kueue_resource_flavor_from_template, kueue_local_queue_from_template",
    [
        pytest.param(
            {"name": NAMESPACE_NAME, "add-kueue-label": True},
            RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
            {
                "name": "kueue-isvc-raw",
                "min-replicas": MIN_REPLICAS,
                "max-replicas": MAX_REPLICAS,
                "labels": {"kueue.x-k8s.io/queue-name": LOCAL_QUEUE_NAME},
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "model-dir": "test-dir",
                "model-version": ModelVersion.OPSET13,
                "resources": ISVC_RESOURCES,
            },
            {
                "name": CLUSTER_QUEUE_NAME,
                "resource_flavor_name": RESOURCE_FLAVOR_NAME,
                "cpu_quota": CPU_QUOTA,
                "memory_quota": MEMORY_QUOTA,
                # "namespace_selector": {"matchLabels": {"kubernetes.io/metadata.name": NAMESPACE_NAME}},
                "namespace_selector": {},
            },
            {"name": RESOURCE_FLAVOR_NAME},
            {"name": LOCAL_QUEUE_NAME, "cluster_queue": CLUSTER_QUEUE_NAME},
        )
    ],
    indirect=True,
)
class TestKueueInferenceServiceRaw:
    """Test inference service with raw deployment"""

    def test_kueue_inference_service_raw(
        self,
        admin_client,
        kueue_resource_flavor_from_template,
        kueue_cluster_queue_from_template,
        kueue_local_queue_from_template,
        kueue_raw_inference_service,
        kueue_kserve_serving_runtime,
    ):
        """Test inference service with raw deployment"""
        labels = [
            create_isvc_label_selector_str(
                isvc=kueue_raw_inference_service,
                resource_type="deployment",
                runtime_name=kueue_kserve_serving_runtime.name,
            )
        ]
        deployments = list(
            Deployment.get(
                label_selector=",".join(labels),
                namespace=kueue_raw_inference_service.namespace,
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
        isvc_to_update = kueue_raw_inference_service.instance.to_dict()
        isvc_to_update["spec"]["predictor"]["minReplicas"] = 2
        kueue_raw_inference_service.update(isvc_to_update)

        # Give time for updated deployment
        time.sleep(10)  # noqa: FCN001

        # Verify deployment still has 1 pod due to Kueue admission control
        deployments = list(
            Deployment.get(
                label_selector=",".join(labels),
                namespace=kueue_raw_inference_service.namespace,
                dyn_client=admin_client,
            )
        )
        if len(deployments) != 1:
            deployment_names = [deployment.instance.metadata.name for deployment in deployments]
            raise DeploymentValidationError(f"Expected 1 deployment, got {len(deployments)}: {deployment_names}")

        deployment = deployments[0]
        try:
            deployment.wait_for_replicas(deployed=True, timeout=Timeout.TIMEOUT_30SEC)
        except TimeoutExpiredError as e:
            available_replicas = deployment.instance.status.availableReplicas
            if available_replicas != 1:
                raise DeploymentValidationError(
                    f"Deployment should have 1 available replica, got {available_replicas}"
                ) from None
            # Get pods that match isvc labels and verify their status
            pods = list(
                Pod.get(
                    label_selector=",".join(labels),
                    namespace=kueue_raw_inference_service.namespace,
                    dyn_client=admin_client,
                )
            )

            if len(pods) != 2:
                pod_names = [pod.instance.metadata.name for pod in pods]
                raise DeploymentValidationError(f"Expected 2 pods, got {len(pods)}: {pod_names}") from e

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

            if running_pods != 1 or gated_pods != 1:
                raise DeploymentValidationError(
                    f"Expected 1 Running pod and 1 SchedulingGated pod, "
                    f"got {running_pods} Running and {gated_pods} SchedulingGated"
                ) from e
                # Check InferenceService status for total model copies
            # Refresh the isvc instance to get latest status
            kueue_raw_inference_service.get()
            isvc = kueue_raw_inference_service.instance
            if isvc.status.modelStatus.copies.totalCopies != 1:
                raise DeploymentValidationError(
                    f"InferenceService should have 1 total model copy, got {isvc.status.modelStatus.copies.totalCopies}"
                ) from e
