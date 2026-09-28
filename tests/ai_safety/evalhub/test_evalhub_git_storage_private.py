"""Private repository test cases for EvalHub Git Storage Source (RHAISTRAT-2058)."""

import re
from collections.abc import Callable

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.secret import Secret

from tests.ai_safety.evalhub.constants import (
    GIT_CLONE_INIT_CONTAINER_NAME,
    GIT_TOKENIZER_PATH,
)
from tests.ai_safety.evalhub.utils import (
    get_evalhub_job_http,
    wait_for_evalhub_job,
    wait_for_evalhub_runtime_job_count,
)

GIT_MODEL_NAMESPACE = pytest.param({"name": "test-evalhub-git-private-storage"})


@pytest.mark.parametrize("model_namespace", [GIT_MODEL_NAMESPACE], indirect=True)
@pytest.mark.tier2
@pytest.mark.ai_safety
@pytest.mark.skip_on_disconnected
class TestEvalHubGitStoragePrivate:
    """Private git repository test cases for EvalHub Git Storage Source.

    Covers: TC-API-005, TC-GIT-004, TC-SEC-001 through TC-SEC-004,
    TC-NEG-003, TC-NEG-005, TC-E2E-001.
    """

    def test_private_repo_job_e2e(
        self,
        admin_client: DynamicClient,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        submit_git_job: Callable[..., str],
        git_private_repo_config: dict[str, str],
        git_test_creds_secret: Secret,
    ) -> None:
        """Given a private git repo with valid credentials,
        when an evaluation job is submitted with test_data_ref.git and secret_ref,
        then the job completes and git_commit_sha metadata is recorded."""
        job_id = submit_git_job(
            url=git_private_repo_config["url"],
            ref=git_private_repo_config["ref"],
            sub_path=git_private_repo_config["sub_path"],
            secret_ref=git_test_creds_secret.name,
            tokenizer_path=GIT_TOKENIZER_PATH,
            job_name="git-private-e2e",
        )

        job_data = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
        )
        assert job_data.get("status", {}).get("state") == "completed", (
            f"Private repo job should complete, got: {job_data.get('status')}"
        )

        # Extract resolved_sha from test_data_ref in the job spec benchmarks
        benchmarks = job_data.get("benchmarks", [])
        assert benchmarks, "Expected benchmarks in job spec"

        test_data_ref = benchmarks[0].get("test_data_ref", {})
        resolved_sha = test_data_ref.get("resolved_sha")

        assert resolved_sha, f"Expected resolved_sha in benchmarks[0].test_data_ref, got test_data_ref: {test_data_ref}"
        assert re.fullmatch(r"[0-9a-f]{40}", resolved_sha), f"Expected 40-char hex SHA, got: {resolved_sha}"

    def test_api_accepts_secret_ref(
        self,
        admin_client: DynamicClient,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        submit_git_job: Callable[..., str],
        git_private_repo_config: dict[str, str],
        git_test_creds_secret: Secret,
    ) -> None:
        """Given a valid git credential Secret,
        when a job is submitted with test_data_ref.git including secret_ref,
        then the API accepts the job and preserves secret_ref in the job detail response.

        Note: submit_git_job internally validates 202 Accepted via submit_evalhub_job."""
        job_id = submit_git_job(
            url=git_private_repo_config["url"],
            ref=git_private_repo_config["ref"],
            sub_path=git_private_repo_config["sub_path"],
            secret_ref=git_test_creds_secret.name,
            tokenizer_path=GIT_TOKENIZER_PATH,
            job_name="git-api-secret-ref",
        )

        response = get_evalhub_job_http(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
        )
        response.raise_for_status()
        job_detail = response.json()

        benchmarks = job_detail.get("benchmarks", [])
        assert benchmarks, "Job should have benchmarks in response"
        git_config = benchmarks[0].get("test_data_ref", {}).get("git", {})
        assert git_config.get("secret_ref") == git_test_creds_secret.name, (
            f"Expected secret_ref '{git_test_creds_secret.name}' preserved in response, got: {git_config}"
        )

    def test_init_container_clone_and_security(
        self,
        admin_client: DynamicClient,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        submit_git_job: Callable[..., str],
        git_private_repo_config: dict[str, str],
        git_test_creds_secret: Secret,
    ) -> None:
        """Given a private repo job,
        when the pod spec is inspected,
        then the git-clone init container has correct security posture
        and the credential Secret is mounted only in the init container."""
        job_id = submit_git_job(
            url=git_private_repo_config["url"],
            ref=git_private_repo_config["ref"],
            sub_path=git_private_repo_config["sub_path"],
            secret_ref=git_test_creds_secret.name,
            tokenizer_path=GIT_TOKENIZER_PATH,
            job_name="git-init-security",
        )

        batch_jobs = wait_for_evalhub_runtime_job_count(
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_job_id=job_id,
            minimum=1,
        )
        spec = batch_jobs[0].instance.spec.template.spec

        git_init = None
        for container in spec.initContainers or []:
            if container.name == GIT_CLONE_INIT_CONTAINER_NAME:
                git_init = container
                break
        assert git_init is not None, (
            f"Expected '{GIT_CLONE_INIT_CONTAINER_NAME}' init container, "
            f"got: {[c.name for c in (spec.initContainers or [])]}"
        )

        # TC-SEC-001: Secret mounted only in init container, not in main containers
        init_volume_names = {vm.name for vm in (git_init.volumeMounts or [])}
        secret_volumes = {
            vol.name
            for vol in (spec.volumes or [])
            if getattr(vol, "secret", None) is not None
            and getattr(vol.secret, "secretName", None) == git_test_creds_secret.name
        }
        assert secret_volumes, f"Expected Secret volume for '{git_test_creds_secret.name}' in pod spec"
        assert secret_volumes & init_volume_names, (
            "Credential Secret volume must be mounted in the git-clone init container"
        )

        for container in spec.containers or []:
            container_volume_names = {vm.name for vm in (container.volumeMounts or [])}
            leaked = secret_volumes & container_volume_names
            assert not leaked, (
                f"Credential Secret must NOT be mounted in container '{container.name}', but found volume(s): {leaked}"
            )

        # TC-SEC-002: Non-root
        sec_ctx = git_init.securityContext
        assert sec_ctx is not None, "Init container must have a securityContext"
        assert getattr(sec_ctx, "runAsNonRoot", None) is True, "Init container must have runAsNonRoot: true"

        # TC-SEC-003: SeccompProfile RuntimeDefault
        seccomp = getattr(sec_ctx, "seccompProfile", None)
        assert seccomp is not None, "Init container must have a seccompProfile"
        assert getattr(seccomp, "type", None) == "RuntimeDefault", (
            f"Expected SeccompProfile type 'RuntimeDefault', got: {getattr(seccomp, 'type', None)}"
        )

        # TC-SEC-004: Drop ALL capabilities
        caps = getattr(sec_ctx, "capabilities", None)
        assert caps is not None, "Init container must have capabilities configured"
        drop_list = [str(d) for d in (getattr(caps, "drop", None) or [])]
        assert "ALL" in drop_list, f"Init container must drop ALL capabilities, got: {drop_list}"
        add_list = list(getattr(caps, "add", None) or [])
        assert not add_list, f"Init container must not add capabilities, got: {add_list}"

        # Wait for job to finish to confirm clone succeeded (TC-GIT-004 exit code 0)
        job_data = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
        )
        assert job_data.get("status", {}).get("state") == "completed", (
            f"Private repo job should complete (init container exit 0), got: {job_data.get('status')}"
        )

    def test_invalid_credentials_fails(
        self,
        admin_client: DynamicClient,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        submit_git_job: Callable[..., str],
        git_private_repo_config: dict[str, str],
        git_bad_creds_secret: Secret,
    ) -> None:
        """Given a Secret with invalid credentials,
        when a job is submitted for a private repo,
        then the job fails due to authentication error."""
        job_id = submit_git_job(
            url=git_private_repo_config["url"],
            ref=git_private_repo_config["ref"],
            sub_path=git_private_repo_config["sub_path"],
            secret_ref=git_bad_creds_secret.name,
            tokenizer_path=GIT_TOKENIZER_PATH,
            job_name="git-bad-creds",
        )

        job_data = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
            timeout=300,
        )
        status = job_data.get("status", {})
        assert status.get("state") == "failed", f"Job with invalid credentials should fail, got: {status}"

        # Verify the failure was due to git-clone init container termination (authentication failure)
        error_message = status.get("message", {}).get("message", "")
        benchmarks = job_data.get("benchmarks") or status.get("benchmarks", [])
        if benchmarks:
            benchmark_error = benchmarks[0].get("error_message", {}).get("message", "")
            error_message = error_message or benchmark_error

        assert "init: terminated" in error_message.lower(), (
            f"Expected git-clone init container termination due to auth failure, got: {error_message}"
        )
        assert "exitcode=1" in error_message.lower() or "reason=error" in error_message.lower(), (
            f"Expected init container to exit with error code, got: {error_message}"
        )
