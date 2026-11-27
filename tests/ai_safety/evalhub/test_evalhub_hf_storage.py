"""HuggingFace Hub as a storage source for evaluation provider test data (RHAISTRAT-2059).

Covers eval-hub PR #996 / FVT @hf scenarios: public dataset download via test_data_ref.hf,
resolved_sha population, init-container wiring, API validation, and runtime init failures.

Defaults to eval-hub-test/evalhub-offline-testdata @ main (public FVT mirror of tests/git-testdata).
Requires cluster egress to huggingface.co (or HF_ENDPOINT mirror on the init image).
"""

from collections.abc import Callable

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.ai_safety.evalhub.constants import (
    ENV_HF_REPO_ID,
    ENV_HF_REVISION,
    GIT_CLONE_INIT_CONTAINER_NAME,
    GIT_COMMIT_SHA_PATTERN,
    GIT_TEST_DATA_MOUNT_PATH,
    HF_BAD_REPO_ID,
    HF_BAD_REVISION,
    HF_BAD_SUB_PATH,
    HF_DEFAULT_REVISION,
    HF_MUTUAL_EXCLUSION_MESSAGE,
    HF_PUBLIC_REPO_ID,
    HF_RESOLVED_SHA_READONLY_CODE,
    HF_RESOLVED_SHA_READONLY_MESSAGE,
    HF_TOKENIZER_PATH,
)
from tests.ai_safety.evalhub.utils import (
    build_evalhub_job_payload,
    build_git_test_data_ref,
    build_hf_arc_easy_benchmark,
    build_hf_test_data_ref,
    build_hf_truthfulqa_mc1_benchmark,
    build_pvc_test_data_ref,
    delete_evalhub_job,
    post_evalhub_job_raw,
    submit_evalhub_job,
    validate_evalhub_job_completed,
    wait_for_evalhub_job,
    wait_for_evalhub_runtime_job_count,
)

HF_MODEL_NAMESPACE = pytest.param({"name": "test-evalhub-hf-storage"})

HF_REJECTION_CASES = [
    pytest.param(
        build_hf_test_data_ref(repo_id=HF_PUBLIC_REPO_ID, revision=HF_DEFAULT_REVISION)
        | build_pvc_test_data_ref(claim_name="some-pvc"),
        "request_validation_failed",
        HF_MUTUAL_EXCLUSION_MESSAGE,
        id="test_hf_and_pvc_conflict",
    ),
    pytest.param(
        build_hf_test_data_ref(repo_id=HF_PUBLIC_REPO_ID, revision=HF_DEFAULT_REVISION)
        | {
            "s3": {
                "bucket": "some-bucket",
                "key": "some-key",
                "secret_ref": "some-secret",  # pragma: allowlist secret
            }
        },
        "request_validation_failed",
        HF_MUTUAL_EXCLUSION_MESSAGE,
        id="test_hf_and_s3_conflict",
    ),
    pytest.param(
        build_hf_test_data_ref(repo_id=HF_PUBLIC_REPO_ID, revision=HF_DEFAULT_REVISION)
        | build_git_test_data_ref(url="https://github.com/eval-hub/eval-hub", ref="main"),
        "request_validation_failed",
        HF_MUTUAL_EXCLUSION_MESSAGE,
        id="test_hf_and_git_conflict",
    ),
    pytest.param(
        {
            **build_hf_test_data_ref(repo_id=HF_PUBLIC_REPO_ID, revision=HF_DEFAULT_REVISION),
            "resolved_sha": "deadbeefdeadbeefdeadbeefdeadbeef00000000",
        },
        HF_RESOLVED_SHA_READONLY_CODE,
        HF_RESOLVED_SHA_READONLY_MESSAGE,
        id="test_client_supplied_resolved_sha",
    ),
    pytest.param(
        {"hf": {"revision": HF_DEFAULT_REVISION}},
        "request_validation_failed",
        None,
        id="test_missing_repo_id",
    ),
    pytest.param(
        build_hf_test_data_ref(repo_id="   ", revision=HF_DEFAULT_REVISION),
        "request_validation_failed",
        None,
        id="test_whitespace_repo_id",
    ),
]


@pytest.mark.parametrize("model_namespace", [HF_MODEL_NAMESPACE], indirect=True)
@pytest.mark.tier1
@pytest.mark.ai_safety
@pytest.mark.skip_on_disconnected
class TestEvalHubHFStorage:
    """HuggingFace-backed test data source for evaluation jobs (test_data_ref.hf)."""

    def test_hf_multi_benchmark_job_completes(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        submit_hf_job: Callable[..., str],
        hf_public_repo_config: dict[str, str],
    ) -> None:
        """Given a public HuggingFace dataset repo with arc_easy and nested sub_path data,
        when an evaluation job is submitted with test_data_ref.hf on two benchmarks,
        then both benchmarks complete and each records a resolved commit SHA."""
        job_id = submit_hf_job(
            repo_id=hf_public_repo_config["repo_id"],
            revision=hf_public_repo_config["revision"],
            nested_sub_path=hf_public_repo_config["nested_sub_path"],
            sha_revision=hf_public_repo_config["sha_revision"],
            job_name="hf-multi-benchmark",
            multi_benchmark=True,
        )
        job_data = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
            timeout=900,
        )
        validate_evalhub_job_completed(job_data=job_data)

        benchmarks = job_data.get("benchmarks", [])
        assert len(benchmarks) == 2, f"Expected 2 benchmarks in job spec, got: {benchmarks}"

        for benchmark in benchmarks:
            resolved_sha = benchmark.get("test_data_ref", {}).get("resolved_sha")
            assert resolved_sha, (
                f"Expected resolved_sha for benchmark '{benchmark.get('id')}', "
                f"got test_data_ref: {benchmark.get('test_data_ref')}"
            )
            assert GIT_COMMIT_SHA_PATTERN.match(resolved_sha), (
                f"'resolved_sha' value '{resolved_sha}' is not a valid commit hash"
            )

        result_benchmarks = job_data.get("results", {}).get("benchmarks", [])
        result_ids = {benchmark.get("id") for benchmark in result_benchmarks}
        assert result_ids == {"arc_easy", "truthfulqa_mc1"}, f"Unexpected benchmark results: {result_ids}"

    def test_hf_init_container_downloads_into_shared_volume(
        self,
        admin_client: DynamicClient,
        tenant_a_namespace: Namespace,
        submit_hf_job: Callable[..., str],
        hf_public_repo_config: dict[str, str],
    ) -> None:
        """Given an HF-backed evaluation job,
        when the pod spec is inspected,
        then the init container receives HF env vars and populates the shared /test_data volume."""
        job_id = submit_hf_job(
            repo_id=hf_public_repo_config["repo_id"],
            revision=hf_public_repo_config["revision"],
            job_name="hf-init-volume",
        )
        batch_jobs = wait_for_evalhub_runtime_job_count(
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_job_id=job_id,
            minimum=1,
        )
        spec = batch_jobs[0].instance.spec.template.spec

        hf_init = next(
            (container for container in (spec.initContainers or []) if container.name == GIT_CLONE_INIT_CONTAINER_NAME),
            None,
        )
        assert hf_init is not None, (
            f"Expected '{GIT_CLONE_INIT_CONTAINER_NAME}' init container, "
            f"got: {[container.name for container in (spec.initContainers or [])]}"
        )

        init_env = {env_var.name: env_var.value for env_var in (hf_init.env or [])}
        assert init_env.get(ENV_HF_REPO_ID) == hf_public_repo_config["repo_id"], (
            f"Init container {ENV_HF_REPO_ID} mismatch: {init_env.get(ENV_HF_REPO_ID)!r}"
        )
        assert init_env.get(ENV_HF_REVISION) == hf_public_repo_config["revision"], (
            f"Init container {ENV_HF_REVISION} mismatch: {init_env.get(ENV_HF_REVISION)!r}"
        )

        init_mounts = {mount.mountPath: mount.name for mount in (hf_init.volumeMounts or [])}
        assert GIT_TEST_DATA_MOUNT_PATH in init_mounts, (
            f"init container must mount test data at {GIT_TEST_DATA_MOUNT_PATH}, got {init_mounts}"
        )

        adapter_container = next((container for container in spec.containers if container.name == "adapter"), None)
        assert adapter_container is not None, "Expected adapter container in pod spec"
        adapter_mounts = {mount.mountPath for mount in (adapter_container.volumeMounts or [])}
        assert GIT_TEST_DATA_MOUNT_PATH in adapter_mounts, (
            f"adapter must mount {GIT_TEST_DATA_MOUNT_PATH}, got {adapter_mounts}"
        )

    def test_hf_init_container_security_context(
        self,
        admin_client: DynamicClient,
        tenant_a_namespace: Namespace,
        submit_hf_job: Callable[..., str],
        hf_public_repo_config: dict[str, str],
    ) -> None:
        """Given the HF download init container,
        when its pod spec is inspected,
        then it runs non-root with RuntimeDefault seccomp and all capabilities dropped."""
        job_id = submit_hf_job(
            repo_id=hf_public_repo_config["repo_id"],
            revision=hf_public_repo_config["revision"],
            job_name="hf-init-security",
        )
        batch_jobs = wait_for_evalhub_runtime_job_count(
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_job_id=job_id,
            minimum=1,
        )
        spec = batch_jobs[0].instance.spec.template.spec
        hf_init = next(
            (container for container in (spec.initContainers or []) if container.name == GIT_CLONE_INIT_CONTAINER_NAME),
            None,
        )
        assert hf_init is not None, f"Expected '{GIT_CLONE_INIT_CONTAINER_NAME}' init container"

        sec_ctx = hf_init.securityContext
        assert sec_ctx is not None, "Init container must have a securityContext"
        assert getattr(sec_ctx, "runAsNonRoot", None) is True, "Init container must have runAsNonRoot: true"

        seccomp = getattr(sec_ctx, "seccompProfile", None)
        assert seccomp is not None, "Init container must have a seccompProfile"
        assert getattr(seccomp, "type", None) == "RuntimeDefault", (
            f"Expected SeccompProfile type 'RuntimeDefault', got: {getattr(seccomp, 'type', None)}"
        )

        caps = getattr(sec_ctx, "capabilities", None)
        assert caps is not None, "Init container must have capabilities configured"
        drop_list = [str(capability) for capability in (getattr(caps, "drop", None) or [])]
        assert "ALL" in drop_list, f"Init container must drop ALL capabilities, got: {drop_list}"


@pytest.mark.parametrize("model_namespace", [HF_MODEL_NAMESPACE], indirect=True)
@pytest.mark.tier2
@pytest.mark.ai_safety
@pytest.mark.skip_on_disconnected
class TestEvalHubHFStorageNegative:
    """API validation and runtime failure scenarios for test_data_ref.hf."""

    @pytest.mark.parametrize(
        ("test_data_ref", "expected_message_code", "expected_message"),
        HF_REJECTION_CASES,
    )
    def test_reject_invalid_hf_test_data_ref(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
        test_data_ref: dict,
        expected_message_code: str,
        expected_message: str | None,
    ) -> None:
        """Given an invalid test_data_ref.hf configuration,
        when the job is submitted,
        then the API rejects it with the expected validation error."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="hf-validation-test",
        )
        for benchmark in payload["benchmarks"]:
            benchmark["test_data_ref"] = test_data_ref
            benchmark["parameters"]["tokenizer"] = HF_TOKENIZER_PATH
            benchmark.pop("hardware_config", None)

        response = post_evalhub_job_raw(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        assert response.status_code == 400, (
            f"Expected 400 for invalid HF test_data_ref, got {response.status_code}: {response.text}"
        )
        body = response.json()
        assert body.get("message_code") == expected_message_code, (
            f"Expected message_code '{expected_message_code}', got: {body}"
        )
        if expected_message is not None:
            assert expected_message in body.get("message", ""), (
                f"Expected message to contain '{expected_message}', got: {body.get('message')}"
            )

    def test_invalid_repo_and_revision_fail(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
        hf_public_repo_config: dict[str, str],
    ) -> None:
        """Given invalid HuggingFace repo_id and revision values,
        when a multi-benchmark job is submitted,
        then the job fails without producing evaluation metrics."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="hf-runtime-failures",
        )
        payload["benchmarks"] = [
            build_hf_arc_easy_benchmark(repo_id=HF_BAD_REPO_ID, revision=HF_DEFAULT_REVISION),
            build_hf_truthfulqa_mc1_benchmark(
                repo_id=hf_public_repo_config["repo_id"],
                revision=HF_BAD_REVISION,
                sub_path=hf_public_repo_config["nested_sub_path"],
            ),
        ]

        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        job_id = data["resource"]["id"]

        try:
            job_data = wait_for_evalhub_job(
                host=evalhub_mt_route.host,
                token=tenant_a_token,
                ca_bundle_file=evalhub_mt_ca_bundle_file,
                tenant=tenant_a_namespace.name,
                job_id=job_id,
                timeout=600,
            )
            assert job_data.get("status", {}).get("state") == "failed", (
                f"Job with invalid HF repo/revision should fail, got: {job_data.get('status')}"
            )

            benchmarks_with_metrics = [
                benchmark
                for benchmark in ((job_data.get("results", {}) or {}).get("benchmarks") or [])
                if benchmark.get("metrics")
            ]
            assert not benchmarks_with_metrics, (
                f"Evaluation must not produce metrics when HF init fails, got: {benchmarks_with_metrics}"
            )
        finally:
            delete_evalhub_job(
                host=evalhub_mt_route.host,
                token=tenant_a_token,
                ca_bundle_file=evalhub_mt_ca_bundle_file,
                tenant=tenant_a_namespace.name,
                job_id=job_id,
                hard_delete=True,
            )

    def test_missing_sub_path_fails(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        submit_hf_job: Callable[..., str],
        hf_public_repo_config: dict[str, str],
    ) -> None:
        """Given a HuggingFace repo with a nonexistent sub_path,
        when an evaluation job is submitted,
        then the init container fails and the job reaches a failed state."""
        job_id = submit_hf_job(
            repo_id=hf_public_repo_config["repo_id"],
            revision=hf_public_repo_config["revision"],
            sub_path=HF_BAD_SUB_PATH,
            job_name="hf-bad-subpath",
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
            f"Job with invalid HF sub_path should fail, got: {job_data.get('status')}"
        )
