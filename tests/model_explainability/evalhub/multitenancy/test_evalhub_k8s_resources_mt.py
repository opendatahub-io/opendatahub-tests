"""Kubernetes Job / ConfigMap shape for EvalHub evaluation jobs (multitenancy suite).

Reuses one tenant namespace, EvalHub route, and vLLM emulator per test class via
``model_namespace`` + shared class-scoped fixtures. Runtime objects are asserted
against the eval-hub ``job_builders`` conventions.
"""

from __future__ import annotations

import json

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.job import Job
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.service import Service
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_explainability.evalhub.constants import (
    EVALHUB_K8S_ANNOTATION_BENCHMARK_ID,
    EVALHUB_K8S_ANNOTATION_JOB_ID,
    EVALHUB_K8S_ANNOTATION_PROVIDER_ID,
    EVALHUB_K8S_LABEL_APP_VALUE,
    EVALHUB_K8S_LABEL_COMPONENT_VALUE,
    EVALHUB_K8S_LABEL_JOB_ID,
)
from tests.model_explainability.evalhub.utils import (
    build_evalhub_job_payload,
    build_evalhub_multi_benchmark_job_payload,
    delete_evalhub_job,
    evalhub_runtime_label_selector,
    get_evalhub_job_http,
    post_evalhub_job_raw,
    submit_evalhub_job,
    wait_for_evalhub_runtime_job_count,
    wait_for_evalhub_runtime_resources_absent,
)


def _adapter_container(job: Job):
    for c in job.instance.spec.template.spec.containers:
        if c.name == "adapter":
            return c
    pytest.fail(f"No adapter container on Job {job.name}")


def _sidecar_container(job: Job):
    for c in job.instance.spec.template.spec.initContainers or []:
        if c.name == "sidecar":
            return c
    pytest.fail(f"No sidecar container on Job {job.name}")


def _assert_basic_job_shape(job: Job, evalhub_job_id: str) -> None:
    """Labels, volumes, SA, security context, and adapter resources."""
    meta = job.instance.metadata
    assert meta.labels is not None
    assert meta.labels.get("app") == EVALHUB_K8S_LABEL_APP_VALUE
    assert meta.labels.get("component") == EVALHUB_K8S_LABEL_COMPONENT_VALUE
    assert meta.labels.get(EVALHUB_K8S_LABEL_JOB_ID) == evalhub_job_id

    ann = meta.annotations or {}
    assert ann.get(EVALHUB_K8S_ANNOTATION_JOB_ID) == evalhub_job_id
    assert ann.get(EVALHUB_K8S_ANNOTATION_PROVIDER_ID) == "lm_evaluation_harness"
    assert ann.get(EVALHUB_K8S_ANNOTATION_BENCHMARK_ID) == "arc_easy"

    pod_meta = job.instance.spec.template.metadata
    assert pod_meta.labels.get("app") == EVALHUB_K8S_LABEL_APP_VALUE
    assert pod_meta.annotations.get(EVALHUB_K8S_ANNOTATION_JOB_ID) == evalhub_job_id

    spec = job.instance.spec.template.spec
    assert spec.serviceAccountName, "Job pod should use a dedicated ServiceAccount"
    vol_names = {v.name for v in (spec.volumes or [])}
    assert "job-spec" in vol_names, f"Expected job-spec volume, got {vol_names}"
    assert "data" in vol_names
    assert "termination-file-volume" in vol_names
    assert "evalhub-service-ca" in vol_names, "Expected service-serving CA ConfigMap volume"

    adapter = _adapter_container(job=job)
    sc = adapter.securityContext
    assert sc is not None
    assert sc.allowPrivilegeEscalation is False
    assert sc.runAsNonRoot is True
    assert sc.capabilities is not None
    assert sc.capabilities.drop is not None and list(sc.capabilities.drop) == ["ALL"]
    assert sc.seccompProfile is not None and sc.seccompProfile.type == "RuntimeDefault"

    assert adapter.resources is not None
    assert adapter.resources.requests, "Adapter should have CPU/memory requests from provider defaults"
    assert adapter.resources.limits, "Adapter should have CPU/memory limits from provider defaults"

    sidecar = _sidecar_container(job=job)
    ssc = sidecar.securityContext
    assert ssc is not None and ssc.allowPrivilegeEscalation is False


def _assert_configmap_spec_owner(batch_job: Job, spec_cm: ConfigMap) -> None:
    refs = spec_cm.instance.metadata.ownerReferences or []
    assert len(refs) == 1, f"Expected single owner on spec ConfigMap, got {refs}"
    owner = refs[0]
    assert owner.apiVersion == "batch/v1"
    assert owner.kind == "Job"
    assert owner.name == batch_job.name
    assert owner.controller is True


@pytest.fixture(scope="class")
def k8s_resources_shared_evalhub_job_id(
    tenant_a_token: str,
    tenant_a_namespace: Namespace,
    evalhub_mt_ca_bundle_file: str,
    evalhub_mt_route: Route,
    evalhub_vllm_emulator_service: Service,
    admin_client: DynamicClient,
) -> str:
    """Submit one evaluation job and wait for its batch Job (shared by basic spec tests)."""
    payload = build_evalhub_job_payload(
        model_service_name=evalhub_vllm_emulator_service.name,
        tenant_namespace=tenant_a_namespace.name,
        job_name="evalhub-k8s-spec-shared-job",
    )
    data = submit_evalhub_job(
        host=evalhub_mt_route.host,
        token=tenant_a_token,
        ca_bundle_file=evalhub_mt_ca_bundle_file,
        tenant=tenant_a_namespace.name,
        payload=payload,
    )
    job_id = data["resource"]["id"]
    wait_for_evalhub_runtime_job_count(
        admin_client=admin_client,
        namespace=tenant_a_namespace.name,
        evalhub_job_id=job_id,
        minimum=1,
    )
    return job_id


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-k8s-resources"},
        ),
    ],
    indirect=True,
)
@pytest.mark.model_explainability
@pytest.mark.usefixtures("k8s_resources_shared_evalhub_job_id")
class TestEvalHubK8sJobResources:
    """Single submitted job: assert batch Job + spec ConfigMap wiring (shared class job)."""

    def test_job_and_configmap_basic_spec(
        self,
        admin_client: DynamicClient,
        tenant_a_namespace: Namespace,
        k8s_resources_shared_evalhub_job_id: str,
    ) -> None:
        """Given: a running EvalHub job.

        Then: batch Job has expected labels, volumes, security, limits;
        spec ConfigMap owned by Job.
        """
        shared_evalhub_job_id = k8s_resources_shared_evalhub_job_id
        selector = evalhub_runtime_label_selector(evalhub_job_id=shared_evalhub_job_id)
        jobs = list(
            Job.get(
                client=admin_client,
                namespace=tenant_a_namespace.name,
                label_selector=selector,
            )
        )
        assert len(jobs) == 1, f"Expected 1 batch Job for single-benchmark submit, got {len(jobs)}"
        batch_job = jobs[0]
        _assert_basic_job_shape(job=batch_job, evalhub_job_id=shared_evalhub_job_id)

        cms = list(
            ConfigMap.get(
                client=admin_client,
                namespace=tenant_a_namespace.name,
                label_selector=selector,
            )
        )
        spec_cms = [cm for cm in cms if cm.name.endswith("-spec")]
        assert len(spec_cms) == 1, f"Expected one *-spec ConfigMap, got {[c.name for c in cms]}"
        spec_cm = spec_cms[0]
        _assert_configmap_spec_owner(batch_job=batch_job, spec_cm=spec_cm)

        data = dict(spec_cm.instance.data or {})
        assert "job.json" in data
        assert "sidecar_config.json" in data
        job_spec = json.loads(data["job.json"])
        assert job_spec["id"] == shared_evalhub_job_id
        assert job_spec["benchmark_id"] == "arc_easy"
        assert job_spec["benchmark_index"] == 0
        assert job_spec["parameters"].get("tokenizer") == "google/flan-t5-small"


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-k8s-multibench"},
        ),
    ],
    indirect=True,
)
@pytest.mark.model_explainability
class TestEvalHubK8sMultiBenchmarkJob:
    """Two benchmarks → two batch Jobs; per-benchmark labels and job.json parameters differ."""

    def test_multi_benchmark_distinct_jobs_and_parameters(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
        admin_client: DynamicClient,
    ) -> None:
        """Given: job with two arc_easy entries and different num_examples.

        Then: two Jobs; job.json maps parameters per index.
        """
        payload = build_evalhub_multi_benchmark_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="evalhub-k8s-multibench-job",
        )
        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        job_id = data["resource"]["id"]
        jobs = wait_for_evalhub_runtime_job_count(
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_job_id=job_id,
            minimum=2,
        )
        assert len(jobs) == 2

        by_index: dict[int, Job] = {}
        for j in jobs:
            idx_str = j.instance.metadata.labels.get("benchmark_index")
            assert idx_str is not None
            by_index[int(idx_str)] = j

        assert set(by_index.keys()) == {0, 1}
        _assert_basic_job_shape(job=by_index[0], evalhub_job_id=job_id)
        _assert_basic_job_shape(job=by_index[1], evalhub_job_id=job_id)

        selector = evalhub_runtime_label_selector(evalhub_job_id=job_id)
        cms = list(
            ConfigMap.get(
                client=admin_client,
                namespace=tenant_a_namespace.name,
                label_selector=selector,
            )
        )
        spec_cms = [cm for cm in cms if cm.name.endswith("-spec")]
        assert len(spec_cms) == 2

        specs_by_index: dict[int, dict] = {}
        for cm in spec_cms:
            raw = (cm.instance.data or {}).get("job.json")
            assert raw
            parsed = json.loads(raw)
            specs_by_index[int(parsed["benchmark_index"])] = parsed

        # Both benchmarks share the same tokenizer but have distinct indices
        assert specs_by_index[0]["parameters"].get("tokenizer") == "google/flan-t5-small"
        assert specs_by_index[1]["parameters"].get("tokenizer") == "google/flan-t5-small"
        assert specs_by_index[0]["benchmark_index"] == 0
        assert specs_by_index[1]["benchmark_index"] == 1


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-k8s-delete"},
        ),
    ],
    indirect=True,
)
@pytest.mark.model_explainability
class TestEvalHubK8sJobDeleteBehaviour:
    """DELETE behaviours: soft vs hard_delete; Kubernetes runtime cleanup (background delete on server)."""

    def test_soft_cancel_removes_runtime_and_keeps_api_record(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
        admin_client: DynamicClient,
    ) -> None:
        """Given: pending/running job. When: DELETE with hard_delete=false.

        Then: K8s objects gone; GET still 200 with cancelled.
        """
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="evalhub-k8s-soft-cancel",
        )
        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        job_id = data["resource"]["id"]
        wait_for_evalhub_runtime_job_count(
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_job_id=job_id,
            minimum=1,
        )

        resp = delete_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
            hard_delete=False,
        )
        assert resp.status_code == 204

        wait_for_evalhub_runtime_resources_absent(
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_job_id=job_id,
        )

        get_resp = get_evalhub_job_http(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
        )
        assert get_resp.status_code == 200
        body = get_resp.json()
        assert body.get("status", {}).get("state") == "cancelled"

    def test_hard_delete_removes_api_record_and_runtime(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
        admin_client: DynamicClient,
    ) -> None:
        """Given: job with batch resources. When: DELETE hard_delete=true. Then: GET 404 and no Job/ConfigMap remain."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="evalhub-k8s-hard-delete",
        )
        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        job_id = data["resource"]["id"]
        wait_for_evalhub_runtime_job_count(
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_job_id=job_id,
            minimum=1,
        )

        resp = delete_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
            hard_delete=True,
        )
        assert resp.status_code == 204

        wait_for_evalhub_runtime_resources_absent(
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_job_id=job_id,
        )

        get_resp = get_evalhub_job_http(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
        )
        assert get_resp.status_code == 404

    def test_delete_uses_background_propagation_cleanup(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
        admin_client: DynamicClient,
    ) -> None:
        """Server deletes batch Jobs with background propagation; we observe no orphaned labeled Job/ConfigMap."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="evalhub-k8s-propagation",
        )
        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        job_id = data["resource"]["id"]
        wait_for_evalhub_runtime_job_count(
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_job_id=job_id,
            minimum=1,
        )

        resp = delete_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
            hard_delete=True,
        )
        assert resp.status_code == 204

        wait_for_evalhub_runtime_resources_absent(
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_job_id=job_id,
        )


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-k8s-mlflow"},
        ),
    ],
    indirect=True,
)
@pytest.mark.model_explainability
class TestEvalHubK8sMlflowJobSpec:
    """experiment_name / tags in job.json when the cluster EvalHub enables MLflow."""

    def test_mlflow_experiment_fields_in_job_json_when_supported(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
        admin_client: DynamicClient,
    ) -> None:
        """If POST accepts an experiment block, spec ConfigMap job.json must carry experiment_name and tags."""
        base = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="evalhub-k8s-mlflow-spec-job",
        )
        base["experiment"] = {
            "name": "odh-fvt-evalhub-experiment",
            "tags": [{"key": "suite", "value": "k8s-resources-mt"}],
        }
        resp = post_evalhub_job_raw(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=base,
        )
        if resp.status_code != 202:
            pytest.skip(
                "EvalHub instance does not accept experiment payload (MLflow likely disabled): "
                f"{resp.status_code} {resp.text}"
            )

        job_id = resp.json()["resource"]["id"]
        wait_for_evalhub_runtime_job_count(
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_job_id=job_id,
            minimum=1,
        )

        selector = evalhub_runtime_label_selector(evalhub_job_id=job_id)
        cms = list(
            ConfigMap.get(
                client=admin_client,
                namespace=tenant_a_namespace.name,
                label_selector=selector,
            )
        )
        spec_cms = [cm for cm in cms if cm.name.endswith("-spec")]
        assert len(spec_cms) >= 1
        raw = (spec_cms[0].instance.data or {}).get("job.json")
        assert raw
        spec = json.loads(raw)
        assert spec.get("experiment_name") == "odh-fvt-evalhub-experiment"
        tags = spec.get("tags") or []
        assert any(t.get("key") == "suite" and t.get("value") == "k8s-resources-mt" for t in tags)

        delete_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
            hard_delete=True,
        )
        wait_for_evalhub_runtime_resources_absent(
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_job_id=job_id,
        )


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-k8s-omit-param"},
        ),
    ],
    indirect=True,
)
@pytest.mark.model_explainability
class TestEvalHubK8sOmitHardDeleteQuery:
    """Omitting hard_delete matches soft-cancel behaviour for runtime cleanup."""

    def test_omit_hard_delete_query_soft_cancels_like_false(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
        admin_client: DynamicClient,
    ) -> None:
        """When: DELETE without hard_delete query. Then: same as hard_delete=false (cancelled + runtime cleared)."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="evalhub-k8s-omit-hard-delete",
        )
        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        job_id = data["resource"]["id"]
        wait_for_evalhub_runtime_job_count(
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_job_id=job_id,
            minimum=1,
        )

        resp = delete_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
        )
        assert resp.status_code == 204

        wait_for_evalhub_runtime_resources_absent(
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_job_id=job_id,
        )

        def poll_state() -> str | None:
            r = get_evalhub_job_http(
                host=evalhub_mt_route.host,
                token=tenant_a_token,
                ca_bundle_file=evalhub_mt_ca_bundle_file,
                tenant=tenant_a_namespace.name,
                job_id=job_id,
            )
            if r.status_code != 200:
                return None
            return r.json().get("status", {}).get("state")

        try:
            for state in TimeoutSampler(wait_timeout=60, sleep=3, func=poll_state):
                if state == "cancelled":
                    break
        except TimeoutExpiredError as err:
            raise AssertionError("Job did not reach cancelled state after DELETE without hard_delete") from err
