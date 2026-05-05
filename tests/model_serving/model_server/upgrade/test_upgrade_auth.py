import pytest

from tests.model_serving.model_server.upgrade.utils import (
    get_isvc_baseline,
    verify_auth_enabled,
    verify_inference_generation,
    verify_isvc_pods_not_restarted_against_baseline,
    verify_serving_runtime_generation,
)
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import Protocols
from utilities.inference_utils import Inference, UserInference
from utilities.manifests.openvino import OPENVINO_KSERVE_INFERENCE_CONFIG

pytestmark = [pytest.mark.rawdeployment, pytest.mark.usefixtures("valid_aws_config")]


@pytest.mark.usefixtures("capture_auth_upgrade_baseline")
class TestPreUpgradeAuthModelServer:
    """Pre-upgrade tests for authentication-enabled model serving."""

    @pytest.mark.pre_upgrade
    def test_auth_raw_deployment_pre_upgrade_inference(
        self,
        auth_inference_service_fixture,
        auth_inference_token_fixture,
    ):
        """Verify authenticated model inference before upgrade"""
        verify_inference_response(
            inference_service=auth_inference_service_fixture,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
            token=auth_inference_token_fixture,
        )

    @pytest.mark.pre_upgrade
    def test_auth_enabled_pre_upgrade(self, auth_inference_service_fixture):
        """Verify authentication annotation is enabled on InferenceService before upgrade"""
        verify_auth_enabled(isvc=auth_inference_service_fixture)


class TestPostUpgradeAuthModelServer:
    """Post-upgrade tests for authentication-enabled model serving."""

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="auth_isvc_exists")
    def test_auth_raw_deployment_post_upgrade_exists(self, auth_inference_service_fixture):
        """Verify authenticated InferenceService exists after upgrade"""
        assert auth_inference_service_fixture.exists, (
            f"InferenceService {auth_inference_service_fixture.name} does not exist after upgrade"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["auth_isvc_exists"])
    def test_auth_enabled_post_upgrade(self, auth_inference_service_fixture):
        """Verify authentication annotation is preserved after upgrade"""
        verify_auth_enabled(isvc=auth_inference_service_fixture)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["auth_isvc_exists"])
    def test_auth_raw_deployment_post_upgrade_not_modified(
        self, auth_inference_service_fixture, upgrade_baseline_fixture
    ):
        """Verify authenticated InferenceService is not modified during upgrade"""
        baseline = get_isvc_baseline(
            baselines=upgrade_baseline_fixture,
            isvc_name=auth_inference_service_fixture.name,
        )
        verify_inference_generation(
            isvc=auth_inference_service_fixture,
            expected_generation=baseline["isvc_observed_generation"],
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["auth_isvc_exists"])
    def test_auth_raw_deployment_post_upgrade_runtime_not_modified(
        self, auth_inference_service_fixture, upgrade_baseline_fixture
    ):
        """Verify ServingRuntime is not modified during upgrade"""
        baseline = get_isvc_baseline(
            baselines=upgrade_baseline_fixture,
            isvc_name=auth_inference_service_fixture.name,
        )
        verify_serving_runtime_generation(
            isvc=auth_inference_service_fixture,
            expected_generation=baseline["runtime_generation"],
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["auth_isvc_exists"])
    def test_auth_raw_deployment_post_upgrade_inference(
        self,
        auth_inference_service_fixture,
        auth_inference_token_fixture,
    ):
        """Verify authenticated model inference after upgrade"""
        verify_inference_response(
            inference_service=auth_inference_service_fixture,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
            token=auth_inference_token_fixture,
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["auth_isvc_exists"])
    def test_auth_raw_deployment_post_upgrade_unauthorized_rejected(
        self,
        auth_inference_service_fixture,
    ):
        """Verify unauthorized requests are rejected after upgrade"""
        inference = UserInference(
            inference_service=auth_inference_service_fixture,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
        )

        res = inference.run_inference_flow(
            model_name=auth_inference_service_fixture.name,
            use_default_query=True,
            insecure=False,
        )

        output = res.get("output", res)
        if isinstance(output, dict):
            output = str(output)

        status_line = output.splitlines()[0] if output else ""
        assert "401" in status_line or "403" in status_line, (
            f"Expected 401 Unauthorized or 403 Forbidden, got: {status_line}"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["auth_isvc_exists"])
    def test_auth_raw_deployment_post_upgrade_fresh_token_inference(
        self,
        auth_inference_service_fixture,
        auth_fresh_token_fixture,
    ):
        """Verify a freshly created token also works for inference after upgrade"""
        verify_inference_response(
            inference_service=auth_inference_service_fixture,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
            token=auth_fresh_token_fixture,
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["auth_isvc_exists"])
    def test_auth_raw_deployment_post_upgrade_pods_not_restarted(
        self,
        admin_client,
        auth_inference_service_fixture,
        upgrade_baseline_fixture,
    ):
        """Verify InferenceService pods have not restarted beyond pre-upgrade baseline"""
        baseline = get_isvc_baseline(
            baselines=upgrade_baseline_fixture,
            isvc_name=auth_inference_service_fixture.name,
        )
        verify_isvc_pods_not_restarted_against_baseline(
            client=admin_client,
            isvc=auth_inference_service_fixture,
            baseline_restart_counts=baseline["pod_restart_counts"],
        )
