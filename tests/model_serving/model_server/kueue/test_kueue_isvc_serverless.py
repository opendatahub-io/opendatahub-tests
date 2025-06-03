"""
Integration test for Kueue and InferenceService admission control.
This test imports the reusable test logic from utilities.kueue_utils.
"""

import pytest
from ocp_resources.deployment import Deployment
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

local_queue_name = "local-queue-serverless"
cluster_queue_name = "cluster-queue-serverless"
resource_flavor_name = "default-flavor-serverless"
cpu_quota = 2
memory_quota = "10Gi"
isvc_resources = {"requests": {"cpu": "1", "memory": "8Gi"}, "limits": {"cpu": cpu_quota, "memory": memory_quota}}
min_replicas = (
    1  # min_replicas needs to be 1 or you need to change the test to check for the number of available replicas
)
max_replicas = 2


@pytest.mark.serverless
@pytest.mark.parametrize(
    "unprivileged_model_namespace, kueue_kserve_serving_runtime, kueue_kserve_inference_service, "
    "kueue_cluster_queue_from_template, kueue_resource_flavor_from_template, kueue_local_queue_from_template",
    [
        pytest.param(
            {"name": "kueue-isvc-serverless-test", "add-kueue-label": True},
            RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
            {
                **ONNX_SERVERLESS_INFERENCE_SERVICE_CONFIG,
                "name": "kueue",
                "min-replicas": min_replicas,
                "max-replicas": max_replicas,
                "labels": {"kueue.x-k8s.io/queue-name": local_queue_name},
                "deployment-mode": KServeDeploymentType.SERVERLESS,
                "resources": isvc_resources,
            },
            {
                "name": cluster_queue_name,
                "resource_flavor_name": resource_flavor_name,
                "cpu_quota": cpu_quota,
                "memory_quota": memory_quota,
                "namespace_selector": {},
            },
            {"name": resource_flavor_name},
            {"name": local_queue_name, "cluster_queue": cluster_queue_name},
        )
    ],
    indirect=True,
)
class TestKueueInferenceServiceServerless:
    """Test inference service with serverless deployment"""

    def test_kueue_inference_service_model_mesh(
        self,
        admin_client,
        kueue_cluster_queue_from_template,
        kueue_resource_flavor_from_template,
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
            raise DeploymentValidationError("Too many deployments found")

        deployment = deployments[0]
        deployment.wait_for_replicas(deployed=True)
        if deployment.instance.spec.replicas != 1:
            raise DeploymentValidationError("Deployment should have 1 replica")

        # Update inference service to request 2 replicas
        isvc_to_update = kueue_kserve_inference_service.instance.to_dict()
        isvc_to_update["spec"]["predictor"]["minReplicas"] = 2
        kueue_kserve_inference_service.update(isvc_to_update)

        # Give time for updated deployment
        time.sleep(10)
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
                total_available_replicas += deployment.instance.spec.replicas
        except TimeoutExpiredError:
            pass
        if total_available_replicas != 1:
            raise DeploymentValidationError("Total available replicas across all deployments should be 1")
