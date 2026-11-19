"""Tests for NeMo Guardrails TrustyAI admin discoverability APIs.

Covers the cluster-facing contracts from TrustyAI NeMo Guardrails PR #69:
- capability manifest: /admin/info, /admin/info/rails, /admin/info/actions

Admin payloads are exercised through the in-cluster admin Service (port 444),
matching test_admin_route.py NetworkPolicy access patterns. User-route deny
checks still go through the OpenShift Route + kube-rbac-proxy path filter.

Config inspection for RHAI-520/521 is out of scope for this suite.
"""

from __future__ import annotations

import http
import json
from collections.abc import Generator
from contextlib import ExitStack
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.route import Route

from tests.ai_safety.nemo_guardrails.constants import (
    ADMIN_INFO_ACTIONS_ENDPOINT,
    ADMIN_INFO_ENDPOINT,
    ADMIN_INFO_RAILS_ENDPOINT,
    ADMIN_INFO_UNKNOWN_SUBPATH,
    EXPECTED_ADMIN_CATALOG_PATHS,
    INVALID_BEARER_TOKEN,
    NEMO_API_TOKEN_SECRET_VALUE,
)
from tests.ai_safety.nemo_guardrails.utils import (
    assert_admin_info_shape,
    build_nemo_admin_service_url,
    execute_admin_service_get,
    send_admin_get,
)
from utilities.general import generate_random_name
from utilities.resources.nemo_guardrails import NemoGuardrails


@pytest.fixture
def nemo_admin_api_caller_pod(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[Pod, Any, Any]:
    """Same-namespace curl caller pod for admin-service API requests."""
    namespace_name = model_namespace.name
    assert namespace_name is not None, "Caller namespace must have a name"
    with ExitStack() as stack:
        pod_name = generate_random_name(prefix="nemo-admin-api-caller")
        pod = stack.enter_context(
            cm=Pod(
                client=admin_client,
                name=pod_name,
                namespace=namespace_name,
                label={"app": "nemo-admin-api-caller-pod"},
                containers=[
                    {
                        "name": pod_name,
                        "image": "image-registry.openshift-image-registry.svc:5000/openshift/tools:latest",
                        "imagePullPolicy": "Always",
                        "args": ["sleep", "infinity"],
                        "securityContext": {
                            "allowPrivilegeEscalation": False,
                            "seccompProfile": {"type": "RuntimeDefault"},
                            "capabilities": {"drop": ["ALL"]},
                        },
                    }
                ],
            )
        )
        pod.wait_for_condition(condition="Ready", status="True")
        yield pod


@pytest.mark.tier1
@pytest.mark.ai_safety
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-nemo-guardrails"})],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed", "nemo_guardrails_presidio_auth_healthcheck")
class TestNemoAdminCapabilityManifest:
    """Capability manifest endpoints from RHAI-517 (/admin/info*)."""

    def test_admin_info_authenticated_200(
        self,
        current_client_token: str,
        nemo_guardrails_presidio_auth: NemoGuardrails,
        nemo_admin_api_caller_pod: Pod,
    ) -> None:
        """
        Test that authenticated callers can read the fork capability manifest.

        Given: Auth-enabled NeMo Guardrails and an in-cluster caller pod
        When: GET /admin/info via the admin Service with a valid token
        Then: Response is 200 with identity, endpoints, rails, and actions
        """
        status, body_text = execute_admin_service_get(
            pod=nemo_admin_api_caller_pod,
            service_url=build_nemo_admin_service_url(
                nemo_cr=nemo_guardrails_presidio_auth,
                path=ADMIN_INFO_ENDPOINT,
            ),
            token=current_client_token,
        )
        assert status == http.HTTPStatus.OK, f"Expected 200 from {ADMIN_INFO_ENDPOINT}, got {status}: {body_text[:300]}"
        assert_admin_info_shape(body=json.loads(body_text))

    def test_admin_info_lists_fork_endpoints(
        self,
        current_client_token: str,
        nemo_guardrails_presidio_auth: NemoGuardrails,
        nemo_admin_api_caller_pod: Pod,
    ) -> None:
        """
        Test that the endpoint catalog lists TrustyAI-tagged fork paths.

        Given: Auth-enabled NeMo Guardrails and an in-cluster caller pod
        When: GET /admin/info
        Then: endpoints includes /admin/info*, /admin/info/rails, /admin/info/actions,
              and /v1/guardrail/checks
        """
        status, body_text = execute_admin_service_get(
            pod=nemo_admin_api_caller_pod,
            service_url=build_nemo_admin_service_url(
                nemo_cr=nemo_guardrails_presidio_auth,
                path=ADMIN_INFO_ENDPOINT,
            ),
            token=current_client_token,
        )
        assert status == http.HTTPStatus.OK, body_text[:300]

        body = json.loads(body_text)
        catalog_paths = {entry["path"] for entry in body["endpoints"]}
        missing = [path for path in EXPECTED_ADMIN_CATALOG_PATHS if path not in catalog_paths]
        assert not missing, f"Missing catalog paths {missing}; got {sorted(catalog_paths)}"

        checks = next(entry for entry in body["endpoints"] if entry["path"] == "/v1/guardrail/checks")
        assert checks["method"] == "POST"
        assert checks["description"]

    def test_admin_info_rails_and_actions_subpaths(
        self,
        current_client_token: str,
        nemo_guardrails_presidio_auth: NemoGuardrails,
        nemo_admin_api_caller_pod: Pod,
    ) -> None:
        """
        Test that rails and actions catalogs are available as subpaths.

        Given: Auth-enabled NeMo Guardrails and an in-cluster caller pod
        When: GET /admin/info/rails and GET /admin/info/actions
        Then: Both return 200; rails matches the parent document; actions are named entries
        """
        parent_status, parent_body = execute_admin_service_get(
            pod=nemo_admin_api_caller_pod,
            service_url=build_nemo_admin_service_url(
                nemo_cr=nemo_guardrails_presidio_auth,
                path=ADMIN_INFO_ENDPOINT,
            ),
            token=current_client_token,
        )
        rails_status, rails_body = execute_admin_service_get(
            pod=nemo_admin_api_caller_pod,
            service_url=build_nemo_admin_service_url(
                nemo_cr=nemo_guardrails_presidio_auth,
                path=ADMIN_INFO_RAILS_ENDPOINT,
            ),
            token=current_client_token,
        )
        actions_status, actions_body = execute_admin_service_get(
            pod=nemo_admin_api_caller_pod,
            service_url=build_nemo_admin_service_url(
                nemo_cr=nemo_guardrails_presidio_auth,
                path=ADMIN_INFO_ACTIONS_ENDPOINT,
            ),
            token=current_client_token,
        )

        assert parent_status == http.HTTPStatus.OK, parent_body[:300]
        assert rails_status == http.HTTPStatus.OK, rails_body[:300]
        assert actions_status == http.HTTPStatus.OK, actions_body[:300]

        parent = json.loads(parent_body)
        rails = json.loads(rails_body)
        actions = json.loads(actions_body)

        assert rails == parent["rails"]
        assert isinstance(rails, list)
        assert len(rails) > 0, "Expected built-in rails modules in this image"
        assert isinstance(actions, list)
        assert all("name" in action for action in actions), f"Unexpected actions payload: {actions[:5]}"

    def test_admin_info_blocked_on_user_route(
        self,
        openshift_ca_bundle_file: str,
        current_client_token: str,
        nemo_guardrails_presidio_auth_route: Route,
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
        nemo_guardrails_presidio_auth: NemoGuardrails,
        nemo_admin_api_caller_pod: Pod,
    ) -> None:
        """
        Test that unauthenticated requests cannot read /admin/info.

        Given: Auth-enabled NeMo Guardrails and an in-cluster caller pod
        When: GET /admin/info via the admin Service without a token
        Then: Admin proxy returns 401 Unauthorized
        """
        status, _body = execute_admin_service_get(
            pod=nemo_admin_api_caller_pod,
            service_url=build_nemo_admin_service_url(
                nemo_cr=nemo_guardrails_presidio_auth,
                path=ADMIN_INFO_ENDPOINT,
            ),
            token=None,
        )
        assert status == http.HTTPStatus.UNAUTHORIZED, f"Expected 401 from admin service, got {status}"


@pytest.mark.tier2
@pytest.mark.ai_safety
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-nemo-guardrails"})],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed", "nemo_guardrails_presidio_auth_healthcheck")
class TestNemoAdminCapabilityManifestEdgeCases:
    """Minimal cluster edge pack for /admin/info*.

    Keep: secret non-leak, bad token, unknown /admin/info/... 404, subpath auth/deny.
    Parent no-token 401, user-route deny, and catalog paths live in
    TestNemoAdminCapabilityManifest (tier1).
    """

    def test_admin_info_does_not_leak_mounted_secret_value(
        self,
        current_client_token: str,
        nemo_guardrails_presidio_auth: NemoGuardrails,
        nemo_admin_api_caller_pod: Pod,
    ) -> None:
        """
        Test that /admin/info does not echo mounted model API token values.

        Given: Auth-enabled NeMo Guardrails with OPENAI_API_KEY mounted from a Secret
        When: GET /admin/info via the admin Service
        Then: The Secret token value is not present in the response body
        """
        status, body_text = execute_admin_service_get(
            pod=nemo_admin_api_caller_pod,
            service_url=build_nemo_admin_service_url(
                nemo_cr=nemo_guardrails_presidio_auth,
                path=ADMIN_INFO_ENDPOINT,
            ),
            token=current_client_token,
        )
        assert status == http.HTTPStatus.OK, body_text[:300]
        assert NEMO_API_TOKEN_SECRET_VALUE not in body_text, (
            "Mounted Secret token value must not appear in /admin/info response"
        )

    def test_admin_info_invalid_token_401(
        self,
        nemo_guardrails_presidio_auth: NemoGuardrails,
        nemo_admin_api_caller_pod: Pod,
    ) -> None:
        """
        Test that an invalid bearer token cannot read /admin/info.

        Given: Auth-enabled NeMo Guardrails and an in-cluster caller pod
        When: GET /admin/info with Authorization: Bearer totally-invalid
        Then: Admin proxy returns 401/403
        """
        status, _body = execute_admin_service_get(
            pod=nemo_admin_api_caller_pod,
            service_url=build_nemo_admin_service_url(
                nemo_cr=nemo_guardrails_presidio_auth,
                path=ADMIN_INFO_ENDPOINT,
            ),
            token=INVALID_BEARER_TOKEN,
        )
        assert status in {
            http.HTTPStatus.UNAUTHORIZED,
            http.HTTPStatus.FORBIDDEN,
        }, f"Expected 401/403 from admin service for invalid token, got {status}"

    def test_admin_info_unknown_subpath_404(
        self,
        current_client_token: str,
        nemo_guardrails_presidio_auth: NemoGuardrails,
        nemo_admin_api_caller_pod: Pod,
    ) -> None:
        """
        Test that unknown /admin/info/... subpaths return 404.

        Given: Auth-enabled NeMo Guardrails and an in-cluster caller pod
        When: GET /admin/info/does-not-exist with a valid token
        Then: Response is 404
        """
        status, body_text = execute_admin_service_get(
            pod=nemo_admin_api_caller_pod,
            service_url=build_nemo_admin_service_url(
                nemo_cr=nemo_guardrails_presidio_auth,
                path=ADMIN_INFO_UNKNOWN_SUBPATH,
            ),
            token=current_client_token,
        )
        assert status == http.HTTPStatus.NOT_FOUND, (
            f"Expected 404 for {ADMIN_INFO_UNKNOWN_SUBPATH}, got {status}: {body_text[:300]}"
        )

    @pytest.mark.parametrize(
        "admin_path",
        [
            pytest.param(ADMIN_INFO_RAILS_ENDPOINT, id="test_rails_subpath"),
            pytest.param(ADMIN_INFO_ACTIONS_ENDPOINT, id="test_actions_subpath"),
        ],
    )
    def test_admin_info_subpaths_blocked_on_user_route(
        self,
        openshift_ca_bundle_file: str,
        current_client_token: str,
        nemo_guardrails_presidio_auth_route: Route,
        admin_path: str,
    ) -> None:
        """
        Test that admin info subpaths are blocked on the user route.

        Given: Auth-enabled NeMo Guardrails and a valid user token
        When: GET /admin/info/rails or /admin/info/actions via the user route
        Then: kube-rbac-proxy path filter returns 404
        """
        response = send_admin_get(
            host=nemo_guardrails_presidio_auth_route.host,
            path=admin_path,
            ca_bundle_file=openshift_ca_bundle_file,
            token=current_client_token,
        )
        assert response.status_code == http.HTTPStatus.NOT_FOUND, (
            f"Expected 404 from user route for {admin_path}, got {response.status_code}"
        )

    @pytest.mark.parametrize(
        "admin_path",
        [
            pytest.param(ADMIN_INFO_RAILS_ENDPOINT, id="test_rails_subpath"),
            pytest.param(ADMIN_INFO_ACTIONS_ENDPOINT, id="test_actions_subpath"),
        ],
    )
    def test_admin_info_subpaths_unauthenticated_401(
        self,
        nemo_guardrails_presidio_auth: NemoGuardrails,
        nemo_admin_api_caller_pod: Pod,
        admin_path: str,
    ) -> None:
        """
        Test that admin info subpaths require authentication.

        Given: Auth-enabled NeMo Guardrails and an in-cluster caller pod
        When: GET /admin/info/rails or /admin/info/actions without a token
        Then: Admin proxy returns 401 Unauthorized
        """
        status, _body = execute_admin_service_get(
            pod=nemo_admin_api_caller_pod,
            service_url=build_nemo_admin_service_url(
                nemo_cr=nemo_guardrails_presidio_auth,
                path=admin_path,
            ),
            token=None,
        )
        assert status == http.HTTPStatus.UNAUTHORIZED, f"Expected 401 from admin service for {admin_path}, got {status}"
