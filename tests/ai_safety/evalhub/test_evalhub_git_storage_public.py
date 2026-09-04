"""Public git repository as a storage source for EvalHub evaluation test data (RHAISTRAT-2058)."""

from collections.abc import Callable

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.ai_safety.evalhub.constants import (
    ENV_GIT_REF,
    ENV_GIT_URL,
    EVALHUB_LOG_ADAPTER_CONTAINER,
    GIT_DEFAULT_REF,
    GIT_INVALID_REF,
    GIT_PUBLIC_REPO_TAG,
    GIT_PUBLIC_REPO_TAG_COMMIT,
    GIT_PUBLIC_REPO_URL,
    GIT_TOKENIZER_PATH,
    TEST_DATA_MOUNT_PATH,
)
from tests.ai_safety.evalhub.utils import (
    build_git_test_data_ref,
    build_pvc_test_data_ref,
    effective_security_context_field,
    find_resolved_sha,
    get_git_job_spec_and_init_container,
    post_evalhub_job_with_test_data_ref,
    validate_evalhub_job_completed,
    wait_for_evalhub_job,
)

GIT_MODEL_NAMESPACE = pytest.param({"name": "test-evalhub-git-storage-public"})


# Table-driven rejections for test_data_ref.git. ``exact_400`` distinguishes conflicts / read-only
# violations (a specific 400) from schema-shape errors that may surface as any 4xx.
GIT_REJECTION_CASES = [
    pytest.param({"git": {"ref": GIT_DEFAULT_REF}}, False, id="missing-repository-url"),
    pytest.param({"git": {"url": GIT_PUBLIC_REPO_URL}}, False, id="missing-ref"),
    pytest.param(
        build_git_test_data_ref(url="not a valid url", ref=GIT_DEFAULT_REF), False, id="invalid-repository-url"
    ),
    pytest.param(
        {
            **build_git_test_data_ref(url=GIT_PUBLIC_REPO_URL, ref=GIT_DEFAULT_REF),
            **build_pvc_test_data_ref(claim_name="some-pvc"),
        },
        True,
        id="git-and-pvc-conflict",
    ),
    pytest.param(
        {
            **build_git_test_data_ref(url=GIT_PUBLIC_REPO_URL, ref=GIT_DEFAULT_REF),
            "s3": {"bucket": "some-bucket", "key": "some-key"},
            **build_pvc_test_data_ref(claim_name="some-pvc"),
        },
        True,
        id="git-s3-and-pvc-conflict",
    ),
    pytest.param(
        {
            **build_git_test_data_ref(url=GIT_PUBLIC_REPO_URL, ref=GIT_DEFAULT_REF),
            "resolved_sha": "0" * 40,
        },
        True,
        id="client-supplied-resolved-sha",
    ),
]


@pytest.mark.parametrize("model_namespace", [GIT_MODEL_NAMESPACE], indirect=True)
@pytest.mark.tier2
@pytest.mark.ai_safety
@pytest.mark.skip_on_disconnected
class TestEvalHubGitStoragePublic:
    """Public git-backed test data for EvalHub: ref variety, clone wiring, security context, clean
    failure, and API-schema validation of test_data_ref.git.

    A single class (rather than one per concern) keeps the derived tenant namespace name within the
    63-character Kubernetes limit and reconciles the evalhub-mt deployment once for the whole file.
    """

    @pytest.mark.parametrize(
        "git_ref, ref_slug",
        [
            pytest.param(GIT_PUBLIC_REPO_TAG, "tag", id="tag"),
            pytest.param(GIT_PUBLIC_REPO_TAG_COMMIT, "commit", id="commit-sha"),
        ],
    )
    def test_public_repo_clone_ref_variety(
        self,
        admin_client: DynamicClient,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        submit_git_job: Callable[..., str],
        git_public_repo_config: dict[str, str],
        git_ref: str,
        ref_slug: str,
    ) -> None:
        """Given a public repository pinned to an immutable ref (a tag or an explicit commit SHA),
        when an evaluation job is submitted with test_data_ref.git,
        then the init container checks out that exact ref, the job completes, and it records the
        exact resolved commit SHA the ref points at.

        The main git-storage suite covers only a branch ref with a pattern-level SHA check. The refs
        here are pinned to the public fixture repo's immutable tag and its commit, so the resolved
        commit SHA is deterministic (a moving branch cannot anchor an exact commit)."""
        job_id = submit_git_job(
            url=git_public_repo_config["url"],
            ref=git_ref,
            sub_path=git_public_repo_config["sub_path"],
            tokenizer_path=GIT_TOKENIZER_PATH,
            job_name=f"git-ref-{ref_slug}",
        )
        _, init_container = get_git_job_spec_and_init_container(
            admin_client=admin_client, namespace=tenant_a_namespace.name, evalhub_job_id=job_id
        )
        init_env = {env.name: env.value for env in (init_container.env or [])}
        assert init_env.get(ENV_GIT_REF) == git_ref, (
            f"Init container {ENV_GIT_REF} mismatch: {init_env.get(ENV_GIT_REF)!r}"
        )
        assert init_env.get(ENV_GIT_URL) == git_public_repo_config["url"], (
            f"Init container {ENV_GIT_URL} mismatch: {init_env.get(ENV_GIT_URL)!r}"
        )

        job_data = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
            timeout=600,
        )
        validate_evalhub_job_completed(job_data=job_data)

        # Schema hedge: the resolved commit may surface as a nested resolved_sha or a top-level
        # git_commit_sha depending on the merged API contract. Both the tag and the commit ref
        # resolve to the same pinned commit.
        resolved_sha = find_resolved_sha(obj=job_data) or job_data.get("git_commit_sha")
        assert resolved_sha == GIT_PUBLIC_REPO_TAG_COMMIT, (
            f"Expected resolved commit {GIT_PUBLIC_REPO_TAG_COMMIT!r} for ref {git_ref!r}, got {resolved_sha!r}"
        )

    def test_git_init_container_clones_into_shared_volume(
        self,
        admin_client: DynamicClient,
        tenant_a_namespace: Namespace,
        submit_git_job: Callable[..., str],
        git_public_repo_config: dict[str, str],
    ) -> None:
        """Given a public repository at a branch ref,
        when an evaluation job is submitted,
        then a git clone init container populates the shared /test_data volume that the adapter
        container then consumes."""
        job_id = submit_git_job(
            url=git_public_repo_config["url"],
            ref=git_public_repo_config["ref"],
            job_name="git-shared-volume",
        )
        spec, init_container = get_git_job_spec_and_init_container(
            admin_client=admin_client, namespace=tenant_a_namespace.name, evalhub_job_id=job_id
        )
        init_env = {env.name: env.value for env in (init_container.env or [])}
        assert init_env.get(ENV_GIT_URL) == git_public_repo_config["url"], (
            f"Init container {ENV_GIT_URL} mismatch: {init_env.get(ENV_GIT_URL)!r}"
        )

        init_mounts = {mount.mountPath: mount.name for mount in (init_container.volumeMounts or [])}
        assert TEST_DATA_MOUNT_PATH in init_mounts, (
            f"init container must mount the shared test-data volume at {TEST_DATA_MOUNT_PATH}, got {init_mounts}"
        )
        shared_volume = init_mounts[TEST_DATA_MOUNT_PATH]

        adapter_container = next(
            (container for container in spec.containers if container.name == EVALHUB_LOG_ADAPTER_CONTAINER), None
        )
        assert adapter_container is not None, (
            f"Expected an {EVALHUB_LOG_ADAPTER_CONTAINER!r} container, "
            f"got: {[container.name for container in (spec.containers or [])]}"
        )
        adapter_mounts = {mount.name for mount in (adapter_container.volumeMounts or [])}
        assert shared_volume in adapter_mounts, (
            f"adapter must consume the same {shared_volume!r} volume the init container populated"
        )

    def test_git_init_container_security_context(
        self,
        admin_client: DynamicClient,
        tenant_a_namespace: Namespace,
        submit_git_job: Callable[..., str],
        git_public_repo_config: dict[str, str],
    ) -> None:
        """Given the git-clone init container,
        when its pod spec is inspected,
        then it runs with a hardened security context: non-root, RuntimeDefault seccomp, and all
        Linux capabilities dropped."""
        job_id = submit_git_job(
            url=git_public_repo_config["url"],
            ref=git_public_repo_config["ref"],
            job_name="git-init-security",
        )
        spec, init_container = get_git_job_spec_and_init_container(
            admin_client=admin_client, namespace=tenant_a_namespace.name, evalhub_job_id=job_id
        )
        pod_sc = spec.securityContext
        container_sc = init_container.securityContext
        assert container_sc is not None, "git init container must define a securityContext"

        # TC-SEC-002: init container runs as non-root.
        run_as_non_root = effective_security_context_field(
            container_sc=container_sc, pod_sc=pod_sc, field="runAsNonRoot"
        )
        run_as_user = effective_security_context_field(container_sc=container_sc, pod_sc=pod_sc, field="runAsUser")
        assert run_as_non_root is True or (run_as_user not in (None, 0)), (
            f"init container must run as non-root (runAsNonRoot={run_as_non_root}, runAsUser={run_as_user})"
        )

        # TC-SEC-003: RuntimeDefault seccomp profile.
        seccomp_profile = effective_security_context_field(
            container_sc=container_sc, pod_sc=pod_sc, field="seccompProfile"
        )
        seccomp_type = getattr(seccomp_profile, "type", None)
        assert seccomp_type == "RuntimeDefault", (
            f"init container seccompProfile.type must be RuntimeDefault, got {seccomp_type!r}"
        )

        # TC-SEC-004: all Linux capabilities dropped.
        dropped = getattr(getattr(container_sc, "capabilities", None), "drop", None) or []
        assert "ALL" in dropped, f"init container must drop ALL capabilities, got drop={list(dropped)}"

    def test_nonexistent_ref_fails_cleanly(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        submit_git_job: Callable[..., str],
        git_public_repo_config: dict[str, str],
    ) -> None:
        """Given a reachable public repository but a nonexistent ref,
        when the job is submitted,
        then it fails cleanly with no evaluation metrics and no recorded resolved commit SHA.

        The main git-storage suite covers only a nonexistent repository."""
        job_id = submit_git_job(
            url=git_public_repo_config["url"],
            ref=GIT_INVALID_REF,
            job_name="git-invalid-ref",
        )
        job_data = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
            timeout=600,
        )
        state = job_data.get("status", {}).get("state")
        assert state == "failed", f"Job with a nonexistent git ref should fail, got state {state!r}"

        benchmarks_with_metrics = [
            benchmark
            for benchmark in ((job_data.get("results", {}) or {}).get("benchmarks") or [])
            if benchmark.get("metrics")
        ]
        assert not benchmarks_with_metrics, (
            f"Evaluation must not produce metrics when the git ref is invalid, got: {benchmarks_with_metrics}"
        )
        assert find_resolved_sha(obj=job_data) is None, "resolved_sha must not be recorded for a failed checkout"

    @pytest.mark.parametrize("test_data_ref, exact_400", GIT_REJECTION_CASES)
    def test_reject_invalid_git_test_data_ref(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
        test_data_ref: dict,
        exact_400: bool,
    ) -> None:
        """Given a job whose test_data_ref.git is invalid (missing a required field, a malformed URL,
        a storage-source conflict, or the read-only resolved_sha),
        when the job is submitted,
        then the API rejects it (a specific 400 for conflicts / read-only violations, any 4xx for
        schema-shape errors) and no job is created.

        The main git-storage suite covers only the plain s3+git conflict."""
        response = post_evalhub_job_with_test_data_ref(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            namespace=tenant_a_namespace.name,
            service_name=evalhub_vllm_emulator_service.name,
            job_name="git-reject",
            test_data_ref=test_data_ref,
        )
        if exact_400:
            assert response.status_code == 400, (
                f"Invalid git test_data_ref must be rejected with 400, got {response.status_code}: {response.text}"
            )
        else:
            assert 400 <= response.status_code < 500, (
                f"Invalid git test_data_ref must be rejected with a 4xx, got {response.status_code}: {response.text}"
            )
