"""Tests for NeMo Guardrails configuration updates."""

import time

import pytest
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.nemo_guardrails import NemoGuardrails
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from timeout_sampler import TimeoutSampler

from tests.model_explainability.nemo_guardrails.constants import (
    CHAT_ENDPOINT,
    MODEL_NAME,
    SAFE_PROMPTS,
)
from tests.model_explainability.nemo_guardrails.utils import send_request


@pytest.mark.tier1
@pytest.mark.model_explainability
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-nemo-guardrails"})],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
class TestNemoGuardrailsConfigUpdate:
    """
    Tests for NeMo Guardrails configuration updates.

    This test class validates:
    1. ConfigMap updates trigger deployment rollout
    2. New configuration is mounted in pods
    3. Service continues to function with updated config
    """

    def test_nemo_config_update_triggers_rollout(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_guardrails_config_update: NemoGuardrails,
        nemo_config_update_configmap: ConfigMap,
        nemo_guardrails_config_update_route: Route,
        nemo_guardrails_config_update_healthcheck,
        openshift_ca_bundle_file: str,
    ):
        """
        Test that updating NemoGuardrails CR triggers deployment rollout.

        Given: NeMo Guardrails deployment with ConfigMap
        When: ConfigMap is updated and CR is annotated to trigger reconciliation
        Then: Deployment rolls out with new config
        """
        # Get initial deployment state
        deployment = Deployment(
            client=admin_client,
            name=nemo_guardrails_config_update.name,
            namespace=model_namespace.name,
        )

        initial_generation = deployment.instance.metadata.generation
        initial_pods = list(
            Pod.get(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector=f"app={nemo_guardrails_config_update.name}",
            )
        )
        initial_pod_names = {pod.name for pod in initial_pods}

        # Verify service is working with initial config
        url = f"https://{nemo_guardrails_config_update_route.host}{CHAT_ENDPOINT}"
        initial_response = send_request(
            url=url,
            token=None,
            ca_bundle_file=openshift_ca_bundle_file,
            message=SAFE_PROMPTS[0],
            model=MODEL_NAME,
            configuration=None,
        )
        assert initial_response.status_code == 200, f"Initial request failed: {initial_response.status_code}"

        # Update the ConfigMap with modified content
        # Read current config
        current_config = yaml.safe_load(nemo_config_update_configmap.instance.data["config.yaml"])

        # Add a comment to trigger update (this won't break functionality)
        current_config["_test_update_marker"] = f"updated-{int(time.time())}"

        # Update the ConfigMap
        nemo_config_update_configmap.update(
            resource_dict={
                "metadata": {
                    "name": nemo_config_update_configmap.name,
                },
                "data": {
                    "config.yaml": yaml.dump(current_config),
                    "prompts.yaml": nemo_config_update_configmap.instance.data["prompts.yaml"],
                    "rails.co": nemo_config_update_configmap.instance.data.get("rails.co", ""),
                },
            }
        )

        # Update the NemoGuardrails CR to trigger operator reconciliation
        # Add an annotation to force the operator to pick up the ConfigMap change
        restart_timestamp = f"{int(time.time())}"
        nemo_guardrails_config_update.update(
            resource_dict={
                "metadata": {
                    "name": nemo_guardrails_config_update.name,
                    "annotations": {
                        "test.opendatahub.io/config-updated": restart_timestamp,
                    },
                }
            }
        )

        # Wait for deployment generation to increase (indicates rollout started)
        def check_generation_increased():
            deployment_check = Deployment(
                client=admin_client,
                name=nemo_guardrails_config_update.name,
                namespace=model_namespace.name,
            )
            current_gen = deployment_check.instance.metadata.generation
            return current_gen > initial_generation

        samples = TimeoutSampler(
            wait_timeout=120,
            sleep=5,
            func=check_generation_increased,
        )
        for sample in samples:
            if sample:
                break

        # Get final generation for verification
        deployment = Deployment(
            client=admin_client,
            name=nemo_guardrails_config_update.name,
            namespace=model_namespace.name,
        )
        new_generation = deployment.instance.metadata.generation

        assert new_generation > initial_generation, (
            f"Deployment generation did not increase after CR update. "
            f"Initial: {initial_generation}, Current: {new_generation}"
        )

        # Wait for new pods to be ready
        deployment.wait_for_replicas()

        # Verify new pods are running
        new_pods = list(
            Pod.get(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector=f"app={nemo_guardrails_config_update.name}",
            )
        )
        new_pod_names = {pod.name for pod in new_pods}

        # At least one pod name should have changed
        assert new_pod_names != initial_pod_names, (
            f"Pod names did not change after rollout. Initial: {initial_pod_names}, New: {new_pod_names}"
        )

        # Identify which pods are actually new (not in initial set)
        newly_created_pod_names = new_pod_names - initial_pod_names
        assert len(newly_created_pod_names) > 0, "No new pods were created during rollout"

        # Wait specifically for the NEW pods to be Running and Ready (1/1)
        def check_new_pods_ready():
            current_pods = list(
                Pod.get(
                    client=admin_client,
                    namespace=model_namespace.name,
                    label_selector=f"app={nemo_guardrails_config_update.name}",
                )
            )
            for pod in current_pods:
                # Only check pods that are newly created
                if pod.name in newly_created_pod_names:
                    # Check if pod is Running
                    if pod.instance.status.phase != "Running":
                        return False
                    # Check if all containers are ready (1/1)
                    container_statuses = pod.instance.status.containerStatuses or []
                    for container_status in container_statuses:
                        if not container_status.ready:
                            return False
            return True

        samples = TimeoutSampler(
            wait_timeout=120,
            sleep=5,
            func=check_new_pods_ready,
        )
        for sample in samples:
            if sample:
                break

        # Verify service still works with updated config
        final_response = send_request(
            url=url,
            token=None,
            ca_bundle_file=openshift_ca_bundle_file,
            message=SAFE_PROMPTS[0],
            model=MODEL_NAME,
            configuration=None,
        )
        assert final_response.status_code == 200, f"Request failed after config update: {final_response.status_code}"
        response_json = final_response.json()
        assert "choices" in response_json, "Response should contain choices after config update"

        # Verify the updated config is mounted in the new pod
        # Get one of the NEWLY CREATED pods (not an old one)
        newly_created_pod = next(pod for pod in new_pods if pod.name in newly_created_pod_names)
        config_file_path = "/app/config/update-test/config.yaml"

        # Exec into the pod to read the mounted config file
        exec_command = ["cat", config_file_path]
        result = newly_created_pod.execute(command=exec_command, container="nemo-guardrails")
        mounted_config = yaml.safe_load(result)

        # Verify our marker is present in the mounted file
        assert "_test_update_marker" in mounted_config, (
            f"Updated config marker not found in mounted config file. Config: {mounted_config}"
        )
        assert mounted_config["_test_update_marker"] == current_config["_test_update_marker"], (
            "Config marker value mismatch in mounted file. "
            f"Expected: {current_config['_test_update_marker']}, "
            f"Got: {mounted_config.get('_test_update_marker')}"
        )
