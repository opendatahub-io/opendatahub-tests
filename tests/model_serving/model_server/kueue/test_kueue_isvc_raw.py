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


pytestmark = [
    pytest.mark.rawdeployment,
    pytest.mark.sanity,
    pytest.mark.usefixtures("valid_aws_config"),
]

local_queue_name = "local-queue-raw"
cluster_queue_name = "cluster-queue-raw"
resource_flavor_name = "default-flavor-raw"
cpu_quota = 2
memory_quota = "10Gi"
isvc_resources = {"requests": {"cpu": "1", "memory": "8Gi"}, "limits": {"cpu": cpu_quota, "memory": memory_quota}}
min_replicas = (
    1  # min_replicas needs to be 1 or you need to change the test to check for the number of available replicas
)
max_replicas = 2


@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "unprivileged_model_namespace, kueue_kserve_serving_runtime, kueue_raw_inference_service, "
    "kueue_cluster_queue_from_template, kueue_resource_flavor_from_template, kueue_local_queue_from_template",
    [
        pytest.param(
            {"name": "kueue-isvc-raw-test", "add-kueue-label": True},
            RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
            {
                "name": "kueue-isvc-raw",
                "min-replicas": min_replicas,
                "max-replicas": max_replicas,
                "labels": {"kueue.x-k8s.io/queue-name": local_queue_name},
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "model-dir": "test-dir",
                "model-version": ModelVersion.OPSET13,
                "resources": isvc_resources
            },
            {
                "name": cluster_queue_name,
                "resource_flavor_name": resource_flavor_name,
                "cpu_quota": cpu_quota,
                "memory_quota": memory_quota,
                "namespace_selector": {}
            },
            {"name": resource_flavor_name},
            {"name": local_queue_name, "cluster_queue": cluster_queue_name},
        )
    ],
    indirect=True,
)
class TestKueueInferenceServiceRaw:
    """Test inference service with raw deployment"""
    def test_kueue_inference_service_raw(
        self,
        admin_client,
        kueue_cluster_queue_from_template,
        kueue_resource_flavor_from_template,
        kueue_local_queue_from_template,
        kueue_raw_inference_service,
        kueue_kserve_serving_runtime
    ):
        """Test inference service with raw deployment"""
        labels = [create_isvc_label_selector_str(
                isvc=kueue_raw_inference_service,
                resource_type="deployment",
                runtime_name=kueue_kserve_serving_runtime.name,
            )]
        deployments = list(Deployment.get(
            label_selector=",".join(labels),
            namespace=kueue_raw_inference_service.namespace,
            dyn_client=admin_client
        ))
        if len(deployments) != 1:
            raise DeploymentValidationError("Too many deployments found")

        deployment = deployments[0]
        deployment.wait_for_replicas(deployed=True)
        if deployment.instance.spec.replicas != 1:
            raise DeploymentValidationError("Deployment should have 1 replica")

        # Update inference service to request 2 replicas
        isvc_to_update = kueue_raw_inference_service.instance.to_dict()
        isvc_to_update["spec"]["predictor"]["minReplicas"] = 2
        kueue_raw_inference_service.update(isvc_to_update)

        # Give time for updated deployment
        time.sleep(seconds=10)

        # Verify deployment still has 1 pod due to Kueue admission control
        deployments = list(Deployment.get(
            label_selector=",".join(labels),
            namespace=kueue_raw_inference_service.namespace,
            dyn_client=admin_client
        ))
        if len(deployments) != 1:
            raise DeploymentValidationError("Too many deployments found")

        deployment = deployments[0]
        try:
            deployment.wait_for_replicas(deployed=True, timeout=Timeout.TIMEOUT_30SEC)
        except TimeoutExpiredError:
            if deployment.instance.status.availableReplicas != 1:
                raise DeploymentValidationError("Deployment should have 1 available replica")
