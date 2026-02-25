"""
Integration test for Kueue and InferenceService admission control.
This test imports the reusable test logic from utilities.kueue_utils.
"""

import pytest
from ocp_resources.deployment import Deployment
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from utilities.constants import KServeDeploymentType, Labels, ModelVersion, RunTimeConfigs
from utilities.general import create_isvc_label_selector_str
from utilities.kueue_utils import check_gated_pods_and_running_pods

pytestmark = [
    pytest.mark.rawdeployment,
    pytest.mark.sanity,
    pytest.mark.usefixtures("valid_aws_config"),
    pytest.mark.kueue,
    pytest.mark.smoke,
]

NAMESPACE_NAME = "test-kueue-isvc-raw"
LOCAL_QUEUE_NAME = "local-queue-raw"
CLUSTER_QUEUE_NAME = "cluster-queue-raw"
RESOURCE_FLAVOR_NAME = "default-flavor-raw"
CPU_QUOTA = 2
MEMORY_QUOTA = "10Gi"
ISVC_RESOURCES = {"requests": {"cpu": "1", "memory": "8Gi"}, "limits": {"cpu": CPU_QUOTA, "memory": MEMORY_QUOTA}}
# min_replicas needs to be 1 or you need to change the test to check for the number of
# available replicas
MIN_REPLICAS = 1
MAX_REPLICAS = 2
EXPECTED_RUNNING_PODS = 1
EXPECTED_GATED_PODS = 1
EXPECTED_DEPLOYMENTS = 1
EXPECTED_INITIAL_REPLICAS = 1
EXPECTED_UPDATED_REPLICAS = 2


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
                "labels": {Labels.Kueue.QUEUE_NAME: LOCAL_QUEUE_NAME},
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

    def _get_deployment_status_replicas(self, deployment: Deployment) -> int:
        deployment.get()
        return deployment.instance.status.replicas

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
        deployment_labels = [
            create_isvc_label_selector_str(
                isvc=kueue_raw_inference_service,
                resource_type="deployment",
                runtime_name=kueue_kserve_serving_runtime.name,
            )
        ]
        pod_labels = [
            create_isvc_label_selector_str(
                isvc=kueue_raw_inference_service,
                resource_type="pod",
                runtime_name=kueue_kserve_serving_runtime.name,
            )
        ]
        deployments = list(
            Deployment.get(
                label_selector=",".join(deployment_labels),
                namespace=kueue_raw_inference_service.namespace,
                client=admin_client,
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

        # Update inference service to request 2 replicas
        isvc_to_update = kueue_raw_inference_service.instance.to_dict()
        isvc_to_update["spec"]["predictor"]["minReplicas"] = EXPECTED_UPDATED_REPLICAS
        kueue_raw_inference_service.update(isvc_to_update)

        # Check the deployment until it has 2 replicas, which means it's been updated
        for replicas in TimeoutSampler(
            wait_timeout=30,
            sleep=2,
            func=lambda: self._get_deployment_status_replicas(deployment),
        ):
            if replicas == EXPECTED_UPDATED_REPLICAS:
                break

        # Verify only 1 pod is running due to Kueue admission control, 1 pod is pending due to Kueue admission control
        try:
            for running_pods, gated_pods in TimeoutSampler(
                wait_timeout=30,
                sleep=2,
                func=lambda: check_gated_pods_and_running_pods(
                    pod_labels, kueue_raw_inference_service.namespace, admin_client
                ),
            ):
                if running_pods == EXPECTED_RUNNING_PODS and gated_pods == EXPECTED_GATED_PODS:
                    break
        except TimeoutExpiredError:
            assert False, (
                f"Timeout waiting for {EXPECTED_RUNNING_PODS} running pods and "
                f"{EXPECTED_GATED_PODS} gated pods, got {running_pods} running pods and {gated_pods} gated pods"
            )

        # Refresh the isvc instance to get latest status
        kueue_raw_inference_service.get()
        isvc = kueue_raw_inference_service.instance
        total_copies = isvc.status.modelStatus.copies.totalCopies
        assert total_copies == EXPECTED_RUNNING_PODS, (
            f"InferenceService should have {EXPECTED_RUNNING_PODS} total model copy, got {total_copies}"
        )
