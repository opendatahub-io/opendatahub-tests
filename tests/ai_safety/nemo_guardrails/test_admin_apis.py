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
from ocp_resources.secret import Secret

from tests.ai_safety.nemo_guardrails.constants import (
    ADMIN_INFO_ACTIONS_ENDPOINT,
    ADMIN_INFO_ENDPOINT,
    ADMIN_INFO_RAILS_ENDPOINT,
    ADMIN_INFO_UNKNOWN_SUBPATH,
    ADMIN_INVALID_TOKEN_FILE,
    ADMIN_VALID_TOKEN_FILE,
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
from utilities.image_constants import SharedImages
from utilities.resources.nemo_guardrails import NemoGuardrails


@pytest.fixture
def nemo_admin_api_caller_pod(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    current_client_token: str,
) -> Generator[Pod, Any, Any]:
    """Same-namespace curl caller with mounted valid/invalid admin token files."""
    namespace_name = model_namespace.name
    assert namespace_name is not None, "Caller namespace must have a name"

    with ExitStack() as stack:
        valid_token_secret = stack.enter_context(
            cm=Secret(
                client=admin_client,
                name=generate_random_name(prefix="nemo-admin-valid-token"),
                namespace=namespace_name,
                # str() unwraps RedactedString so Secret deepcopy can serialize it
                string_data={"token": str(current_client_token)},
                type="Opaque",
            )
        )
        invalid_token_secret = stack.enter_context(
            cm=Secret(
                client=admin_client,
                name=generate_random_name(prefix="nemo-admin-invalid-token"),
                namespace=namespace_name,
                string_data={"token": INVALID_BEARER_TOKEN},
                type="Opaque",
            )
        )
        pod_name = generate_random_name(prefix="nemo-admin-api-caller")
        pod = stack.enter_context(
            cm=Pod(
                client=admin_client,
                name=pod_name,
                namespace=namespace_name,
                label={"app": "nemo-admin-api-caller-pod"},
                volumes=[
                    {"name": "valid-token", "secret": {"secretName": valid_token_secret.name}},
                    {"name": "invalid-token", "secret": {"secretName": invalid_token_secret.name}},
                ],
                containers=[
                    {
                        "name": pod_name,
                        "image": SharedImages.OPENSHIFT_TOOLS,
                        "imagePullPolicy": "IfNotPresent",
                        "args": ["sleep", "infinity"],
                        "volumeMounts": [
                            {
                                "name": "valid-token",
                                "mountPath": "/var/run/secrets/nemo-admin-tokens/valid",
                                "readOnly": True,
                            },
                            {
                                "name": "invalid-token",
                                "mountPath": "/var/run/secrets/nemo-admin-tokens/invalid",
                                "readOnly": True,
                            },
                        ],
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


@pytest.mark.ai_safety
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-nemo-guardrails"})],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed", "nemo_guardrails_presidio_auth_healthcheck")
class TestNemoAdminCapabilityManifest:
    """Compact /admin/info* coverage for RHAI-517."""

    @pytest.mark.tier1
    def test_admin_info_authenticated_contract(
        self,
        nemo_guardrails_presidio_auth: NemoGuardrails,
        nemo_admin_api_caller_pod: Pod,
    ) -> None:
        """
        Test authenticated /admin/info response shape and fork endpoint catalog.

        Given: Auth-enabled NeMo Guardrails and an in-cluster caller pod
        When: GET /admin/info via the admin Service with a valid token file
        Then: Response is 200 with capability shape and expected catalog paths
        """
        status, body_text = execute_admin_service_get(
            pod=nemo_admin_api_caller_pod,
            service_url=build_nemo_admin_service_url(
                nemo_cr=nemo_guardrails_presidio_auth,
                path=ADMIN_INFO_ENDPOINT,
            ),
            token_file=ADMIN_VALID_TOKEN_FILE,
        )
        assert status == http.HTTPStatus.OK, f"Expected 200 from {ADMIN_INFO_ENDPOINT}, got {status}: {body_text[:300]}"

        body = json.loads(body_text)
        assert_admin_info_shape(body=body)

        catalog_paths = {entry["path"] for entry in body["endpoints"]}
        missing = [path for path in EXPECTED_ADMIN_CATALOG_PATHS if path not in catalog_paths]
        assert not missing, f"Missing catalog paths {missing}; got {sorted(catalog_paths)}"

        checks = next(entry for entry in body["endpoints"] if entry["path"] == "/v1/guardrail/checks")
        assert checks["method"] == "POST"
        assert checks["description"]

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        "admin_path",
        [
            pytest.param(ADMIN_INFO_RAILS_ENDPOINT, id="test_rails"),
            pytest.param(ADMIN_INFO_ACTIONS_ENDPOINT, id="test_actions"),
        ],
    )
    def test_admin_info_subpath_authenticated(
        self,
        nemo_guardrails_presidio_auth: NemoGuardrails,
        nemo_admin_api_caller_pod: Pod,
        admin_path: str,
    ) -> None:
        """
        Test authenticated /admin/info rails and actions subpaths.

        Given: Auth-enabled NeMo Guardrails and an in-cluster caller pod
        When: GET /admin/info/rails or /admin/info/actions with a valid token file
        Then: Response is 200; rails matches parent; actions are named entries
        """
        parent_status, parent_body = execute_admin_service_get(
            pod=nemo_admin_api_caller_pod,
            service_url=build_nemo_admin_service_url(
                nemo_cr=nemo_guardrails_presidio_auth,
                path=ADMIN_INFO_ENDPOINT,
            ),
            token_file=ADMIN_VALID_TOKEN_FILE,
        )
        status, body_text = execute_admin_service_get(
            pod=nemo_admin_api_caller_pod,
            service_url=build_nemo_admin_service_url(
                nemo_cr=nemo_guardrails_presidio_auth,
                path=admin_path,
            ),
            token_file=ADMIN_VALID_TOKEN_FILE,
        )

        assert parent_status == http.HTTPStatus.OK, parent_body[:300]
        assert status == http.HTTPStatus.OK, f"Expected 200 from {admin_path}, got {status}: {body_text[:300]}"

        payload = json.loads(body_text)
        assert isinstance(payload, list)
        if admin_path == ADMIN_INFO_RAILS_ENDPOINT:
            assert payload == json.loads(parent_body)["rails"]
            assert len(payload) > 0, "Expected built-in rails modules in this image"
        else:
            assert all("name" in action for action in payload), f"Unexpected actions payload: {payload[:5]}"

    @pytest.mark.tier2
    def test_admin_info_secret_not_leaked(
        self,
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
            token_file=ADMIN_VALID_TOKEN_FILE,
        )
        assert status == http.HTTPStatus.OK, body_text[:300]
        assert NEMO_API_TOKEN_SECRET_VALUE not in body_text, (
            "Mounted Secret token value must not appear in /admin/info response"
        )

    @pytest.mark.tier2
    @pytest.mark.parametrize(
        ("token_file", "expected_statuses"),
        [
            pytest.param(None, {http.HTTPStatus.UNAUTHORIZED}, id="test_missing_token"),
            pytest.param(
                ADMIN_INVALID_TOKEN_FILE,
                {http.HTTPStatus.UNAUTHORIZED, http.HTTPStatus.FORBIDDEN},
                id="test_invalid_token",
            ),
        ],
    )
    def test_admin_service_rejects_missing_or_invalid_token(
        self,
        nemo_guardrails_presidio_auth: NemoGuardrails,
        nemo_admin_api_caller_pod: Pod,
        token_file: str | None,
        expected_statuses: set[http.HTTPStatus],
    ) -> None:
        """
        Test that missing or invalid tokens cannot read /admin/info.

        Given: Auth-enabled NeMo Guardrails and an in-cluster caller pod
        When: GET /admin/info with no token file or an invalid token file
        Then: Admin proxy returns 401/403
        """
        status, _body = execute_admin_service_get(
            pod=nemo_admin_api_caller_pod,
            service_url=build_nemo_admin_service_url(
                nemo_cr=nemo_guardrails_presidio_auth,
                path=ADMIN_INFO_ENDPOINT,
            ),
            token_file=token_file,
        )
        assert status in {code.value for code in expected_statuses}, (
            f"Expected {[code.value for code in expected_statuses]} from admin service, got {status}"
        )

    @pytest.mark.tier2
    def test_admin_info_unknown_subpath_404(
        self,
        nemo_guardrails_presidio_auth: NemoGuardrails,
        nemo_admin_api_caller_pod: Pod,
    ) -> None:
        """
        Test that unknown /admin/info/... subpaths return 404.

        Given: Auth-enabled NeMo Guardrails and an in-cluster caller pod
        When: GET /admin/info/does-not-exist with a valid token file
        Then: Response is 404
        """
        status, body_text = execute_admin_service_get(
            pod=nemo_admin_api_caller_pod,
            service_url=build_nemo_admin_service_url(
                nemo_cr=nemo_guardrails_presidio_auth,
                path=ADMIN_INFO_UNKNOWN_SUBPATH,
            ),
            token_file=ADMIN_VALID_TOKEN_FILE,
        )
        assert status == http.HTTPStatus.NOT_FOUND, (
            f"Expected 404 for {ADMIN_INFO_UNKNOWN_SUBPATH}, got {status}: {body_text[:300]}"
        )

    @pytest.mark.tier2
    @pytest.mark.parametrize(
        "admin_path",
        [
            pytest.param(ADMIN_INFO_ENDPOINT, id="test_info"),
            pytest.param(ADMIN_INFO_RAILS_ENDPOINT, id="test_rails"),
            pytest.param(ADMIN_INFO_ACTIONS_ENDPOINT, id="test_actions"),
        ],
    )
    def test_user_route_blocks_admin_info_paths(
        self,
        openshift_ca_bundle_file: str,
        current_client_token: str,
        nemo_guardrails_presidio_auth_route: Route,
        admin_path: str,
    ) -> None:
        """
        Test that the user route rejects /admin/info* paths.

        Given: Auth-enabled NeMo Guardrails and a valid user token
        When: GET /admin/info, /admin/info/rails, or /admin/info/actions via the user route
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
