"""Git repository as a storage source for evaluation provider test data (RHAISTRAT-2058).

Defaults to eval-hub's own vendored offline lm-eval cache at tests/git-testdata
(https://github.com/eval-hub/eval-hub/tree/main/tests/git-testdata), which exists
specifically for FVT of git clone/checkout without live Hugging Face downloads.
GIT_TEST_PUBLIC_REPO_URL / _REF / _SUB_PATH env vars override these defaults.
Private-repo coverage is deferred to a follow-up PR.
"""

from collections.abc import Callable

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.ai_safety.evalhub.constants import (
    GIT_COMMIT_SHA_PATTERN,
    GIT_FULL_REPO_TOKENIZER_PATH,
    GIT_NONEXISTENT_REPO_URL,
    GIT_TOKENIZER_PATH,
)
from tests.ai_safety.evalhub.utils import (
    build_evalhub_job_payload,
    post_evalhub_job_raw,
    validate_evalhub_job_completed,
    wait_for_evalhub_job,
    wait_for_evalhub_runtime_job_count,
)

GIT_MODEL_NAMESPACE = pytest.param({"name": "test-evalhub-git-storage"})


@pytest.mark.parametrize("model_namespace", [GIT_MODEL_NAMESPACE], indirect=True)
@pytest.mark.tier1
@pytest.mark.ai_safety
@pytest.mark.skip_on_disconnected
class TestEvalHubGitStorage:
    """Git-backed test data source for evaluation jobs (test_data_ref.git)."""

    def test_public_repo_clone_job_completes(
        self,
        admin_client: DynamicClient,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        submit_git_job: Callable[..., str],
        git_public_repo_config: dict[str, str],
    ) -> None:
        """Given a public git repository URL and a branch ref, with no sub_path narrowing,
        when an evaluation job is submitted with test_data_ref.git,
        then the job completes successfully with the full repository cloned by the
        git-clone init container, and no S3 credentials leak into the adapter."""
        job_id = submit_git_job(
            url=git_public_repo_config["url"],
            ref=git_public_repo_config["ref"],
            tokenizer_path=GIT_FULL_REPO_TOKENIZER_PATH,
            job_name="git-public-clone-test",
        )
        job_data = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
        )
        validate_evalhub_job_completed(job_data=job_data)

        batch_jobs = wait_for_evalhub_runtime_job_count(
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_job_id=job_id,
            minimum=1,
        )
        spec = batch_jobs[0].instance.spec.template.spec

        init_containers = spec.initContainers or []
        assert init_containers, "Expected a git-clone init container to populate /test_data/"

        adapter_container = next((c for c in spec.containers if c.name == "adapter"), None)
        assert adapter_container is not None, "Expected adapter container in pod spec"
        s3_env_names = {env_var.name for env_var in (adapter_container.env or []) if "AWS" in env_var.name.upper()}
        assert not s3_env_names, f"Git jobs should not have S3 credential env vars, found: {s3_env_names}"

    def test_public_repo_sub_path_loading(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        submit_git_job: Callable[..., str],
        git_public_repo_config: dict[str, str],
    ) -> None:
        """Given a public repository with a sub-directory of evaluation data,
        when a job specifies test_data_ref.git with sub_path,
        then the job completes successfully using data cloned from that sub-path,
        narrowed so the sub-path's contents appear at the /test_data/ mount root."""
        job_id = submit_git_job(
            url=git_public_repo_config["url"],
            ref=git_public_repo_config["ref"],
            sub_path=git_public_repo_config["sub_path"],
            tokenizer_path=GIT_TOKENIZER_PATH,
            job_name="git-sub-path-test",
        )
        job_data = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
        )
        validate_evalhub_job_completed(job_data=job_data)

    def test_git_commit_sha_recorded_in_job_metadata(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        submit_git_job: Callable[..., str],
        git_public_repo_config: dict[str, str],
    ) -> None:
        """Given a completed git-clone evaluation job,
        when GET /api/v1/evaluations/jobs/{id} is called,
        then the response includes the resolved commit SHA of the cloned ref."""
        job_id = submit_git_job(
            url=git_public_repo_config["url"],
            ref=git_public_repo_config["ref"],
            sub_path=git_public_repo_config["sub_path"],
            tokenizer_path=GIT_TOKENIZER_PATH,
            job_name="git-commit-sha-test",
        )
        job_data = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
        )
        validate_evalhub_job_completed(job_data=job_data)

        benchmarks = job_data.get("results", {}).get("benchmarks", [])
        arc_easy_bench = next((b for b in benchmarks if b.get("id") == "arc_easy"), {})
        commit_sha = arc_easy_bench.get("test_data_ref", {}).get("resolved_sha")
        assert commit_sha, f"Expected 'resolved_sha' in benchmark's test_data_ref, got: {arc_easy_bench}"
        assert GIT_COMMIT_SHA_PATTERN.match(commit_sha), (
            f"'resolved_sha' value '{commit_sha}' is not a valid commit hash"
        )

    def test_missing_repository_job_fails(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        submit_git_job: Callable[..., str],
    ) -> None:
        """Given a job referencing a nonexistent git repository,
        when the job is submitted,
        then the git-clone init container fails and the job reaches a failed state."""
        job_id = submit_git_job(
            url=GIT_NONEXISTENT_REPO_URL,
            ref="main",
            job_name="git-missing-repo-test",
        )
        job_data = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
            timeout=600,
        )
        assert job_data.get("status", {}).get("state") == "failed", (
            "Job referencing a nonexistent git repository should fail"
        )

    def test_mutual_exclusion_s3_and_git_rejected(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """Given a job payload specifying both test_data_ref.s3 and test_data_ref.git,
        when the job is submitted,
        then the API rejects it with HTTP 400 rather than constructing a job."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="git-s3-mutual-exclusion-test",
        )
        for benchmark in payload["benchmarks"]:
            benchmark["test_data_ref"] = {
                "s3": {
                    "bucket": "some-bucket",
                    "key": "some-key",
                    "secret_ref": "some-secret",  # pragma: allowlist secret
                },
                "git": {"url": GIT_NONEXISTENT_REPO_URL, "ref": "main"},
            }

        response = post_evalhub_job_raw(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        assert response.status_code == 400, (
            f"Expected 400 for mutually exclusive s3+git test_data_ref, got {response.status_code}: {response.text}"
        )
