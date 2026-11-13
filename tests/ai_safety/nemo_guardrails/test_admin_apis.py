"""Tests for NeMo Guardrails TrustyAI admin discoverability APIs.

Covers the cluster-facing contracts from TrustyAI NeMo Guardrails PRs:
- #69 capability manifest: /admin/info, /admin/info/rails, /admin/info/actions
- #76/#77 config inspection: /admin/configs, /admin/config/{id}

Auth segregation of /admin/* vs /v1/* remains in test_admin_route.py; these tests
exercise the admin API payloads through the admin route.
"""

import http

import pytest
from ocp_resources.route import Route

from tests.ai_safety.nemo_guardrails.constants import (
    ADMIN_CONFIG_BY_ID_ENDPOINT,
    ADMIN_CONFIGS_ENDPOINT,
    ADMIN_INFO_ACTIONS_ENDPOINT,
    ADMIN_INFO_ENDPOINT,
    ADMIN_INFO_RAILS_ENDPOINT,
    EXPECTED_ADMIN_CATALOG_PATHS,
    LLM_JUDGE_INPUT_FLOW,
    LLM_JUDGE_OUTPUT_FLOW,
    MULTI_CONFIG_A_ID,
    MULTI_CONFIG_B_ID,
    PRESIDIO_CONFIG_ID,
    PRESIDIO_INPUT_FLOW,
    PRESIDIO_OUTPUT_FLOW,
)
from tests.ai_safety.nemo_guardrails.utils import (
    assert_admin_config_detail_shape,
    assert_admin_info_shape,
    assert_config_files_include_yaml,
    send_admin_get,
)


@pytest.mark.tier1
@pytest.mark.ai_safety
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-nemo-guardrails"})],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed", "nemo_guardrails_presidio_admin_route_healthcheck")
class TestNemoAdminCapabilityManifest:
    """Capability manifest endpoints from RHAI-517 (/admin/info*)."""

    def test_admin_info_authenticated_200(
        self,
        openshift_ca_bundle_file: str,
        current_client_token: str,
        nemo_guardrails_presidio_admin_route: Route,
    ) -> None:
        """
        Test that authenticated admins can read the fork capability manifest.

        Given: Auth-enabled NeMo Guardrails with a healthy admin route
        When: GET /admin/info with a valid token
        Then: Response is 200 with identity, endpoints, rails, and actions
        """
        response = send_admin_get(
            host=nemo_guardrails_presidio_admin_route.instance.spec.host,
            path=ADMIN_INFO_ENDPOINT,
            ca_bundle_file=openshift_ca_bundle_file,
            token=current_client_token,
        )
        assert response.status_code == http.HTTPStatus.OK, (
            f"Expected 200 from {ADMIN_INFO_ENDPOINT}, got {response.status_code}: {response.text[:300]}"
        )
        assert_admin_info_shape(body=response.json())

    def test_admin_info_lists_fork_endpoints(
        self,
        openshift_ca_bundle_file: str,
        current_client_token: str,
        nemo_guardrails_presidio_admin_route: Route,
    ) -> None:
        """
        Test that the endpoint catalog lists TrustyAI admin and checks paths.

        Given: Auth-enabled NeMo Guardrails with a healthy admin route
        When: GET /admin/info
        Then: endpoints includes /admin/info*, /admin/config*, and /v1/guardrail/checks
        """
        response = send_admin_get(
            host=nemo_guardrails_presidio_admin_route.instance.spec.host,
            path=ADMIN_INFO_ENDPOINT,
            ca_bundle_file=openshift_ca_bundle_file,
            token=current_client_token,
        )
        assert response.status_code == http.HTTPStatus.OK, response.text[:300]

        catalog_paths = {entry["path"] for entry in response.json()["endpoints"]}
        missing = [path for path in EXPECTED_ADMIN_CATALOG_PATHS if path not in catalog_paths]
        assert not missing, f"Missing catalog paths {missing}; got {sorted(catalog_paths)}"

        checks = next(entry for entry in response.json()["endpoints"] if entry["path"] == "/v1/guardrail/checks")
        assert checks["method"] == "POST"
        assert checks["description"]

    def test_admin_info_rails_and_actions_subpaths(
        self,
        openshift_ca_bundle_file: str,
        current_client_token: str,
        nemo_guardrails_presidio_admin_route: Route,
    ) -> None:
        """
        Test that rails and actions catalogs are available as subpaths.

        Given: Auth-enabled NeMo Guardrails with a healthy admin route
        When: GET /admin/info/rails and GET /admin/info/actions
        Then: Both return 200; rails matches the parent document; actions are named entries
        """
        host = nemo_guardrails_presidio_admin_route.instance.spec.host
        parent = send_admin_get(
            host=host,
            path=ADMIN_INFO_ENDPOINT,
            ca_bundle_file=openshift_ca_bundle_file,
            token=current_client_token,
        )
        rails = send_admin_get(
            host=host,
            path=ADMIN_INFO_RAILS_ENDPOINT,
            ca_bundle_file=openshift_ca_bundle_file,
            token=current_client_token,
        )
        actions = send_admin_get(
            host=host,
            path=ADMIN_INFO_ACTIONS_ENDPOINT,
            ca_bundle_file=openshift_ca_bundle_file,
            token=current_client_token,
        )

        assert parent.status_code == http.HTTPStatus.OK, parent.text[:300]
        assert rails.status_code == http.HTTPStatus.OK, rails.text[:300]
        assert actions.status_code == http.HTTPStatus.OK, actions.text[:300]

        assert rails.json() == parent.json()["rails"]
        assert isinstance(rails.json(), list)
        assert len(rails.json()) > 0, "Expected built-in rails modules in this image"

        action_list = actions.json()
        assert isinstance(action_list, list)
        assert all("name" in action for action in action_list), f"Unexpected actions payload: {action_list[:5]}"

    def test_admin_info_blocked_on_user_route(
        self,
        openshift_ca_bundle_file: str,
        current_client_token: str,
        nemo_guardrails_presidio_auth_route: Route,
        nemo_guardrails_presidio_auth_healthcheck: None,
    ) -> None:
        """
        Test that the user route rejects admin capability paths.

        Given: Auth-enabled NeMo Guardrails and a valid user token
        When: GET /admin/info via the user-facing route
        Then: kube-rbac-proxy path filter returns 404
        """
        response = send_admin_get(
            host=nemo_guardrails_presidio_auth_route.host,
            path=ADMIN_INFO_ENDPOINT,
            ca_bundle_file=openshift_ca_bundle_file,
            token=current_client_token,
        )
        assert response.status_code == http.HTTPStatus.NOT_FOUND, (
            f"Expected 404 from user route path filter, got {response.status_code}"
        )

    def test_admin_info_unauthenticated_401(
        self,
        openshift_ca_bundle_file: str,
        nemo_guardrails_presidio_admin_route: Route,
    ) -> None:
        """
        Test that unauthenticated requests cannot read /admin/info.

        Given: Auth-enabled NeMo Guardrails with a healthy admin route
        When: GET /admin/info without a token
        Then: Admin proxy returns 401 Unauthorized
        """
        response = send_admin_get(
            host=nemo_guardrails_presidio_admin_route.instance.spec.host,
            path=ADMIN_INFO_ENDPOINT,
            ca_bundle_file=openshift_ca_bundle_file,
            token=None,
        )
        assert response.status_code == http.HTTPStatus.UNAUTHORIZED, (
            f"Expected 401 from admin proxy, got {response.status_code}"
        )


@pytest.mark.tier1
@pytest.mark.ai_safety
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-nemo-guardrails"})],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed", "nemo_guardrails_presidio_admin_route_healthcheck")
class TestNemoAdminConfigs:
    """Per-config admin inspection endpoints from RHAI-520/521 (/admin/config*)."""

    def test_admin_configs_map_contains_mounted_config(
        self,
        openshift_ca_bundle_file: str,
        current_client_token: str,
        nemo_guardrails_presidio_admin_route: Route,
    ) -> None:
        """
        Test that /admin/configs lists the mounted Presidio config.

        Given: Auth-enabled Presidio NeMo Guardrails
        When: GET /admin/configs
        Then: Response is a map containing the 'presidio' config with AdminConfigDetail shape
        """
        response = send_admin_get(
            host=nemo_guardrails_presidio_admin_route.instance.spec.host,
            path=ADMIN_CONFIGS_ENDPOINT,
            ca_bundle_file=openshift_ca_bundle_file,
            token=current_client_token,
        )
        assert response.status_code == http.HTTPStatus.OK, (
            f"Expected 200 from {ADMIN_CONFIGS_ENDPOINT}, got {response.status_code}: {response.text[:300]}"
        )

        configs = response.json()
        assert isinstance(configs, dict), f"Expected config map, got {type(configs)}"
        assert PRESIDIO_CONFIG_ID in configs, f"Expected '{PRESIDIO_CONFIG_ID}' in {list(configs.keys())}"

        detail = configs[PRESIDIO_CONFIG_ID]
        assert_admin_config_detail_shape(detail=detail, config_id=PRESIDIO_CONFIG_ID)
        assert_config_files_include_yaml(files=detail["files"], config_id=PRESIDIO_CONFIG_ID)

    def test_admin_config_by_id_matches_configmap(
        self,
        openshift_ca_bundle_file: str,
        current_client_token: str,
        nemo_guardrails_presidio_admin_route: Route,
    ) -> None:
        """
        Test that /admin/config/{id} reflects the mounted Presidio rails and files.

        Given: Auth-enabled Presidio NeMo Guardrails
        When: GET /admin/config/presidio
        Then: Detail includes config.yaml and Presidio input/output flows
        """
        path = ADMIN_CONFIG_BY_ID_ENDPOINT.format(config_id=PRESIDIO_CONFIG_ID)
        response = send_admin_get(
            host=nemo_guardrails_presidio_admin_route.instance.spec.host,
            path=path,
            ca_bundle_file=openshift_ca_bundle_file,
            token=current_client_token,
        )
        assert response.status_code == http.HTTPStatus.OK, (
            f"Expected 200 from {path}, got {response.status_code}: {response.text[:300]}"
        )

        detail = response.json()
        assert_admin_config_detail_shape(detail=detail, config_id=PRESIDIO_CONFIG_ID)
        assert_config_files_include_yaml(files=detail["files"], config_id=PRESIDIO_CONFIG_ID)
        assert PRESIDIO_INPUT_FLOW in detail["rails"]["input"], detail["rails"]["input"]
        assert PRESIDIO_OUTPUT_FLOW in detail["rails"]["output"], detail["rails"]["output"]
        assert "models" in detail["rails_config"]

    def test_admin_config_unknown_id_404(
        self,
        openshift_ca_bundle_file: str,
        current_client_token: str,
        nemo_guardrails_presidio_admin_route: Route,
    ) -> None:
        """
        Test that unknown config ids return 404 from the admin config endpoint.

        Given: Auth-enabled NeMo Guardrails
        When: GET /admin/config/does-not-exist
        Then: Response is 404
        """
        path = ADMIN_CONFIG_BY_ID_ENDPOINT.format(config_id="does-not-exist")
        response = send_admin_get(
            host=nemo_guardrails_presidio_admin_route.instance.spec.host,
            path=path,
            ca_bundle_file=openshift_ca_bundle_file,
            token=current_client_token,
        )
        assert response.status_code == http.HTTPStatus.NOT_FOUND, (
            f"Expected 404 for unknown config id, got {response.status_code}: {response.text[:300]}"
        )

    def test_admin_configs_blocked_on_user_route(
        self,
        openshift_ca_bundle_file: str,
        current_client_token: str,
        nemo_guardrails_presidio_auth_route: Route,
        nemo_guardrails_presidio_auth_healthcheck: None,
    ) -> None:
        """
        Test that the user route rejects /admin/configs.

        Given: Auth-enabled NeMo Guardrails and a valid user token
        When: GET /admin/configs via the user-facing route
        Then: kube-rbac-proxy path filter returns 404
        """
        response = send_admin_get(
            host=nemo_guardrails_presidio_auth_route.host,
            path=ADMIN_CONFIGS_ENDPOINT,
            ca_bundle_file=openshift_ca_bundle_file,
            token=current_client_token,
        )
        assert response.status_code == http.HTTPStatus.NOT_FOUND, (
            f"Expected 404 from user route path filter, got {response.status_code}"
        )

    def test_admin_configs_unauthenticated_401(
        self,
        openshift_ca_bundle_file: str,
        nemo_guardrails_presidio_admin_route: Route,
    ) -> None:
        """
        Test that unauthenticated requests cannot read /admin/configs.

        Given: Auth-enabled NeMo Guardrails with a healthy admin route
        When: GET /admin/configs without a token
        Then: Admin proxy returns 401 Unauthorized
        """
        response = send_admin_get(
            host=nemo_guardrails_presidio_admin_route.instance.spec.host,
            path=ADMIN_CONFIGS_ENDPOINT,
            ca_bundle_file=openshift_ca_bundle_file,
            token=None,
        )
        assert response.status_code == http.HTTPStatus.UNAUTHORIZED, (
            f"Expected 401 from admin proxy, got {response.status_code}"
        )


@pytest.mark.tier2
@pytest.mark.ai_safety
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-nemo-guardrails"})],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed", "nemo_guardrails_multi_config_admin_route_healthcheck")
class TestNemoAdminMultiConfig:
    """Admin config inspection against a multi-config auth-enabled server."""

    def test_admin_configs_lists_all_mounted_ids(
        self,
        openshift_ca_bundle_file: str,
        current_client_token: str,
        nemo_guardrails_multi_config_admin_route: Route,
    ) -> None:
        """
        Test that /admin/configs lists every mounted multi-config id.

        Given: Auth-enabled multi-config NeMo Guardrails (config-a, config-b)
        When: GET /admin/configs
        Then: Both config ids are present with AdminConfigDetail shape
        """
        response = send_admin_get(
            host=nemo_guardrails_multi_config_admin_route.instance.spec.host,
            path=ADMIN_CONFIGS_ENDPOINT,
            ca_bundle_file=openshift_ca_bundle_file,
            token=current_client_token,
        )
        assert response.status_code == http.HTTPStatus.OK, (
            f"Expected 200 from {ADMIN_CONFIGS_ENDPOINT}, got {response.status_code}: {response.text[:300]}"
        )

        configs = response.json()
        for config_id in (MULTI_CONFIG_A_ID, MULTI_CONFIG_B_ID):
            assert config_id in configs, f"Expected '{config_id}' in {list(configs.keys())}"
            assert_admin_config_detail_shape(detail=configs[config_id], config_id=config_id)
            assert_config_files_include_yaml(files=configs[config_id]["files"], config_id=config_id)

    def test_admin_config_by_id_distinguishes_configs(
        self,
        openshift_ca_bundle_file: str,
        current_client_token: str,
        nemo_guardrails_multi_config_admin_route: Route,
    ) -> None:
        """
        Test that per-id admin details reflect distinct mounted configs.

        Given: Auth-enabled multi-config NeMo Guardrails
        When: GET /admin/config/config-a and GET /admin/config/config-b
        Then: config-a exposes LLM-as-judge flows; config-b exposes Presidio flows
        """
        host = nemo_guardrails_multi_config_admin_route.instance.spec.host

        config_a = send_admin_get(
            host=host,
            path=ADMIN_CONFIG_BY_ID_ENDPOINT.format(config_id=MULTI_CONFIG_A_ID),
            ca_bundle_file=openshift_ca_bundle_file,
            token=current_client_token,
        )
        config_b = send_admin_get(
            host=host,
            path=ADMIN_CONFIG_BY_ID_ENDPOINT.format(config_id=MULTI_CONFIG_B_ID),
            ca_bundle_file=openshift_ca_bundle_file,
            token=current_client_token,
        )

        assert config_a.status_code == http.HTTPStatus.OK, config_a.text[:300]
        assert config_b.status_code == http.HTTPStatus.OK, config_b.text[:300]

        detail_a = config_a.json()
        detail_b = config_b.json()
        assert_admin_config_detail_shape(detail=detail_a, config_id=MULTI_CONFIG_A_ID)
        assert_admin_config_detail_shape(detail=detail_b, config_id=MULTI_CONFIG_B_ID)

        assert LLM_JUDGE_INPUT_FLOW in detail_a["rails"]["input"], detail_a["rails"]["input"]
        assert LLM_JUDGE_OUTPUT_FLOW in detail_a["rails"]["output"], detail_a["rails"]["output"]
        assert PRESIDIO_INPUT_FLOW in detail_b["rails"]["input"], detail_b["rails"]["input"]
        assert PRESIDIO_OUTPUT_FLOW in detail_b["rails"]["output"], detail_b["rails"]["output"]
        assert detail_a["rails"] != detail_b["rails"]
