import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.event import Event
from ocp_resources.job import Job
from ocp_resources.role_binding import RoleBinding
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.evalhub.constants import EVALHUB_EVENTS_CLUSTERROLE, EVALHUB_VLLM_EMULATOR_PORT
from tests.ai_safety.evalhub.k8s_lifecycle_signals.constants import (
    LIFECYCLE_EVENT_EMISSION_TIMEOUT,
    LIFECYCLE_JOB_LABEL_TIMEOUT,
    LIFECYCLE_JOB_SUBMIT_TIMEOUT,
    LIFECYCLE_OOM_MEMORY_LIMIT,
    LIFECYCLE_PHASE_FAILED,
    LIFECYCLE_PHASE_LABEL,
    LIFECYCLE_PHASE_SUCCEEDED,
    LIFECYCLE_REASON_COMPLETED,
    LIFECYCLE_REASON_FAILED,
    LIFECYCLE_SOURCE_OPERATOR,
    LIFECYCLE_STATUS_ANNOTATION,
    LIFECYCLE_THRESHOLD_ACCURACY_HIGH,
    LIFECYCLE_THRESHOLD_ACCURACY_PASS,
)
from tests.ai_safety.evalhub.utils import (
    build_evalhub_job_payload,
    build_vllm_arc_easy_benchmark,
    submit_evalhub_job,
)

LOGGER = structlog.get_logger(name=__name__)

KUEUE_LOCAL_QUEUE_NAME = "evalhub-local-queue"


def read_job_label(
    admin_client: DynamicClient,
    job_name: str,
    namespace: str,
    key: str,
) -> str | None:
    """Return a label value from a batch Job using a fresh API read."""
    job = Job(client=admin_client, name=job_name, namespace=namespace)
    if not job.exists:
        return None
    labels = job.instance.metadata.labels or {}
    return labels.get(key)


def _read_job_annotation(
    admin_client: DynamicClient,
    job_name: str,
    namespace: str,
    key: str,
) -> str | None:
    """Return an annotation value from a batch Job using a fresh API read."""
    job = Job(client=admin_client, name=job_name, namespace=namespace)
    if not job.exists:
        return None
    annotations = job.instance.metadata.annotations or {}
    return annotations.get(key)


def get_batch_job_uid(
    admin_client: DynamicClient,
    job_name: str,
    namespace: str,
) -> str | None:
    """Return the UID of a batch Job, or None when the Job does not exist."""
    job = Job(client=admin_client, name=job_name, namespace=namespace)
    if not job.exists:
        return None
    return job.instance.metadata.uid


def is_kueue_installed(admin_client: DynamicClient) -> bool:
    """Return True when the Kueue Workload CRD is available on the cluster."""
    try:
        crd = CustomResourceDefinition(client=admin_client, name="workloads.kueue.x-k8s.io")
        return crd.exists
    except AttributeError, KeyError:
        return False


def kueue_local_queue_exists(admin_client: DynamicClient, namespace: str, queue_name: str) -> bool:
    """Return True when a Kueue LocalQueue exists in the given namespace."""
    from utilities.kueue_utils import LocalQueue

    local_queue = LocalQueue(client=admin_client, name=queue_name, namespace=namespace)
    return local_queue.exists


def wait_for_job_label(
    admin_client: DynamicClient,
    job_name: str,
    namespace: str,
    key: str,
    expected_value: str,
    timeout: int = LIFECYCLE_JOB_LABEL_TIMEOUT,
    sleep: int = 5,
) -> str:
    """Poll until a batch Job has the expected label value; return it.

    Raises TimeoutExpiredError if not set within timeout seconds.
    """
    last_value: str | None = None

    def _get_label() -> str | None:
        nonlocal last_value
        last_value = read_job_label(
            admin_client=admin_client,
            job_name=job_name,
            namespace=namespace,
            key=key,
        )
        return last_value

    for value in TimeoutSampler(wait_timeout=timeout, sleep=sleep, func=_get_label):
        if value == expected_value:
            return value
    raise TimeoutExpiredError(
        f"Job {job_name}/{namespace} label {key} never reached {expected_value!r}; last observed value={last_value!r}"
    )


def wait_for_success_phase_signals(
    admin_client: DynamicClient,
    job_name: str,
    namespace: str,
    *,
    label_timeout: int = LIFECYCLE_JOB_LABEL_TIMEOUT,
    event_timeout: int = LIFECYCLE_EVENT_EMISSION_TIMEOUT,
) -> None:
    """Wait for success-path lifecycle signals after the EvalHub API reports completion.

    Waits for EvaluationCompleted then the Succeeded phase label (operator may lag API state).
    """
    wait_for_event(
        admin_client=admin_client,
        job_name=job_name,
        namespace=namespace,
        reason=LIFECYCLE_REASON_COMPLETED,
        timeout=event_timeout,
    )
    try:
        wait_for_job_label(
            admin_client=admin_client,
            job_name=job_name,
            namespace=namespace,
            key=LIFECYCLE_PHASE_LABEL,
            expected_value=LIFECYCLE_PHASE_SUCCEEDED,
            timeout=label_timeout,
            sleep=2,
        )
    except TimeoutExpiredError:
        raw = _read_job_annotation(
            admin_client=admin_client,
            job_name=job_name,
            namespace=namespace,
            key=LIFECYCLE_STATUS_ANNOTATION,
        )
        if raw:
            phase = parse_status_annotation(raw).get("phase", "")
            if phase in ("Completed", "Succeeded"):
                LOGGER.info(
                    f"Succeeded phase label not observed on {job_name}; "
                    f"evaluation-status annotation phase={phase!r} confirms completion"
                )
                return
        raise


def wait_for_failure_phase_signals(
    admin_client: DynamicClient,
    job_name: str,
    namespace: str,
    *,
    label_timeout: int = LIFECYCLE_JOB_LABEL_TIMEOUT,
    event_timeout: int = LIFECYCLE_EVENT_EMISSION_TIMEOUT,
    source_component: str | None = None,
) -> None:
    """Wait for failure-path lifecycle signals after the EvalHub API reports failure.

    Waits for EvaluationFailed then the Failed phase label (operator may lag API state).
    When source_component is set, waits for that component's EvaluationFailed event only.
    """
    if source_component is not None:
        wait_for_event_from_component(
            admin_client=admin_client,
            job_name=job_name,
            namespace=namespace,
            reason=LIFECYCLE_REASON_FAILED,
            source_component=source_component,
            timeout=event_timeout,
        )
    else:
        wait_for_event(
            admin_client=admin_client,
            job_name=job_name,
            namespace=namespace,
            reason=LIFECYCLE_REASON_FAILED,
            timeout=event_timeout,
        )

    try:
        wait_for_job_label(
            admin_client=admin_client,
            job_name=job_name,
            namespace=namespace,
            key=LIFECYCLE_PHASE_LABEL,
            expected_value=LIFECYCLE_PHASE_FAILED,
            timeout=label_timeout,
            sleep=2,
        )
    except TimeoutExpiredError:
        job = Job(client=admin_client, name=job_name, namespace=namespace)
        if not job.exists:
            LOGGER.info(
                f"Failed phase label not observed on {job_name}; Job was removed after failure event (operator cleanup)"
            )
            return
        raw = _read_job_annotation(
            admin_client=admin_client,
            job_name=job_name,
            namespace=namespace,
            key=LIFECYCLE_STATUS_ANNOTATION,
        )
        if raw:
            phase = parse_status_annotation(raw).get("phase", "")
            if phase == LIFECYCLE_PHASE_FAILED:
                LOGGER.info(
                    f"Failed phase label not observed on {job_name}; "
                    f"evaluation-status annotation phase={phase!r} confirms failure"
                )
                return
        raise


def assert_failure_phase_label(
    admin_client: DynamicClient,
    job_name: str,
    namespace: str,
) -> None:
    """Assert the batch Job carries evaluation-phase=Failed, tolerating operator cleanup.

    When the operator deletes the Job after patching the Failed label, the label is no
    longer readable but failure-path signals were already observed.
    """
    label_value = read_job_label(
        admin_client=admin_client,
        job_name=job_name,
        namespace=namespace,
        key=LIFECYCLE_PHASE_LABEL,
    )
    if label_value == LIFECYCLE_PHASE_FAILED:
        return
    job = Job(client=admin_client, name=job_name, namespace=namespace)
    if label_value is None and not job.exists:
        LOGGER.info(
            f"Failure phase label not read on {job_name}; "
            "operator removed Job after failure sync (label patched before delete)"
        )
        return
    raise AssertionError(f"Expected {LIFECYCLE_PHASE_LABEL}={LIFECYCLE_PHASE_FAILED!r}, got {label_value!r}")


def wait_for_failed_label_near_operator_event(
    admin_client: DynamicClient,
    job_name: str,
    namespace: str,
    *,
    timeout: int = LIFECYCLE_JOB_LABEL_TIMEOUT,
    sleep: int = 1,
) -> str:
    """Poll aggressively for Failed label while the operator failure path runs.

    The operator patches evaluation-phase=Failed before emitting EvaluationFailed
    and deleting the Job, so label checks must run in the same polling window.
    """
    last_value: str | None = None

    def _poll() -> str | None:
        nonlocal last_value
        last_value = read_job_label(
            admin_client=admin_client,
            job_name=job_name,
            namespace=namespace,
            key=LIFECYCLE_PHASE_LABEL,
        )
        if last_value == LIFECYCLE_PHASE_FAILED:
            return last_value

        operator_events = list_events_for_job(
            admin_client=admin_client,
            job_name=job_name,
            namespace=namespace,
            reason=LIFECYCLE_REASON_FAILED,
            source_component=LIFECYCLE_SOURCE_OPERATOR,
        )
        if operator_events:
            last_value = read_job_label(
                admin_client=admin_client,
                job_name=job_name,
                namespace=namespace,
                key=LIFECYCLE_PHASE_LABEL,
            )
            if last_value == LIFECYCLE_PHASE_FAILED:
                return last_value
            job = Job(client=admin_client, name=job_name, namespace=namespace)
            if not job.exists:
                LOGGER.info(
                    f"Failed phase label not read on {job_name}; "
                    "operator removed Job after failure sync (label patched before delete)"
                )
                return LIFECYCLE_PHASE_FAILED

        return None

    for value in TimeoutSampler(wait_timeout=timeout, sleep=sleep, func=_poll):
        if value is not None:
            return value
    raise TimeoutExpiredError(
        f"Job {job_name}/{namespace} label {LIFECYCLE_PHASE_LABEL} never reached "
        f"{LIFECYCLE_PHASE_FAILED!r} near operator failure event; last observed value={last_value!r}"
    )


def get_job_annotation(
    admin_client: DynamicClient,
    job_name: str,
    namespace: str,
    key: str,
) -> str | None:
    """Return the value of an annotation on a batch Job, or None if absent."""
    return _read_job_annotation(
        admin_client=admin_client,
        job_name=job_name,
        namespace=namespace,
        key=key,
    )


def list_events_for_job(
    admin_client: DynamicClient,
    job_name: str,
    namespace: str,
    reason: str | None = None,
    source_component: str | None = None,
) -> list[dict[str, Any]]:
    """Return Kubernetes Events for a specific batch Job.

    Filters by involvedObject.name and optionally by reason or source.component.
    """
    field_selector = f"involvedObject.name={job_name},involvedObject.kind=Job"
    if reason:
        field_selector += f",reason={reason}"

    events = [
        event.to_dict()
        for event in Event.list(
            client=admin_client,
            namespace=namespace,
            field_selector=field_selector,
        )
    ]

    if source_component:
        events = filter_events_by_source(events=events, source_component=source_component)

    return events


def filter_events_by_source(
    events: list[dict[str, Any]],
    source_component: str,
) -> list[dict[str, Any]]:
    """Return events whose source.component matches source_component."""
    return [event for event in events if (event.get("source") or {}).get("component") == source_component]


def wait_for_event(
    admin_client: DynamicClient,
    job_name: str,
    namespace: str,
    reason: str,
    timeout: int = LIFECYCLE_EVENT_EMISSION_TIMEOUT,
) -> dict[str, Any]:
    """Wait until at least one Kubernetes Event with the given reason exists for the Job.

    Returns the first matching Event dict. Raises TimeoutExpiredError on timeout.
    """

    def _find_event() -> dict[str, Any] | None:
        events = list_events_for_job(
            admin_client=admin_client,
            job_name=job_name,
            namespace=namespace,
            reason=reason,
        )
        return events[0] if events else None

    for event in TimeoutSampler(wait_timeout=timeout, sleep=2, func=_find_event):
        if event is not None:
            LOGGER.info(f"Event {reason} emitted for job {job_name}")
            return event
    raise TimeoutExpiredError(f"Event {reason!r} for job {job_name} not emitted within {timeout}s")


def wait_for_event_from_component(
    admin_client: DynamicClient,
    job_name: str,
    namespace: str,
    reason: str,
    source_component: str,
    timeout: int = LIFECYCLE_EVENT_EMISSION_TIMEOUT,
) -> dict[str, Any]:
    """Wait until a Kubernetes Event with the given reason and source.component exists.

    Returns the first matching Event dict. Raises TimeoutExpiredError on timeout.
    """

    def _find_event() -> dict[str, Any] | None:
        events = list_events_for_job(
            admin_client=admin_client,
            job_name=job_name,
            namespace=namespace,
            reason=reason,
            source_component=source_component,
        )
        return events[0] if events else None

    for event in TimeoutSampler(wait_timeout=timeout, sleep=2, func=_find_event):
        if event is not None:
            LOGGER.info(f"Event {reason} from {source_component} emitted for job {job_name}")
            return event
    raise TimeoutExpiredError(
        f"Event {reason!r} from {source_component!r} for job {job_name} not emitted within {timeout}s"
    )


def get_evaluation_job_name(
    admin_client: DynamicClient,
    namespace: str,
    evalhub_job_id: str,
) -> str | None:
    """Find the batch Job name for a given EvalHub logical job ID via label selector."""
    from tests.ai_safety.evalhub.constants import (
        EVALHUB_K8S_LABEL_APP,
        EVALHUB_K8S_LABEL_APP_VALUE,
        EVALHUB_K8S_LABEL_COMPONENT,
        EVALHUB_K8S_LABEL_COMPONENT_VALUE,
        EVALHUB_K8S_LABEL_JOB_ID,
    )

    selector = (
        f"{EVALHUB_K8S_LABEL_APP}={EVALHUB_K8S_LABEL_APP_VALUE},"
        f"{EVALHUB_K8S_LABEL_COMPONENT}={EVALHUB_K8S_LABEL_COMPONENT_VALUE},"
        f"{EVALHUB_K8S_LABEL_JOB_ID}={evalhub_job_id}"
    )
    jobs = list(Job.get(client=admin_client, namespace=namespace, label_selector=selector))
    return jobs[0].name if jobs else None


def wait_for_evaluation_job_name(
    admin_client: DynamicClient,
    namespace: str,
    evalhub_job_id: str,
    timeout: int = LIFECYCLE_JOB_SUBMIT_TIMEOUT,
) -> str:
    """Wait until the batch Job for an EvalHub job exists and return its name."""

    def _find() -> str | None:
        return get_evaluation_job_name(
            admin_client=admin_client,
            namespace=namespace,
            evalhub_job_id=evalhub_job_id,
        )

    for name in TimeoutSampler(wait_timeout=timeout, sleep=3, func=_find):
        if name is not None:
            return name
    raise TimeoutExpiredError(
        f"Batch Job for evalhub job_id={evalhub_job_id} not found in {namespace} within {timeout}s"
    )


def submit_evalhub_job_and_capture_runtime_job(
    admin_client: DynamicClient,
    *,
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant: str,
    payload: dict,
    runtime_job_timeout: int = LIFECYCLE_JOB_SUBMIT_TIMEOUT,
) -> tuple[str, str]:
    """Submit an EvalHub job and capture the backing batch Job name before API completion.

    Returns:
        Tuple of (evalhub_job_id, batch_job_name).
    """
    job_id = submit_evalhub_job(
        host=host,
        token=token,
        ca_bundle_file=ca_bundle_file,
        tenant=tenant,
        payload=payload,
    )["resource"]["id"]
    job_name = wait_for_evaluation_job_name(
        admin_client=admin_client,
        namespace=tenant,
        evalhub_job_id=job_id,
        timeout=runtime_job_timeout,
    )
    return job_id, job_name


def build_nonexistent_adapter_payload(
    vllm_service_name: str,
    namespace: str,
    job_name: str,
) -> dict[str, Any]:
    """Build a job payload with an unreachable model URL.

    The adapter cannot reach the model endpoint, so EvalHub reports failure and emits
    EvaluationFailed (same pattern as eval-hub FVT evalcard_invalid_model.jsonnet).
    """
    model_url = f"http://nonexistent-model.{namespace}.svc.cluster.local:{EVALHUB_VLLM_EMULATOR_PORT}/v1"
    return {
        "name": job_name,
        "model": {"url": model_url, "name": "invalid-model"},
        "benchmarks": [build_vllm_arc_easy_benchmark(num_examples=3)],
    }


def build_lifecycle_success_payload(
    vllm_service_name: str,
    namespace: str,
    job_name: str,
    threshold_accuracy: float = LIFECYCLE_THRESHOLD_ACCURACY_PASS,
) -> dict[str, Any]:
    """Build a job payload that completes without threshold violation on the vLLM emulator.

    Overrides provider default pass_criteria (acc_norm >= 0.25) so lifecycle success-path
    tests observe evaluation-phase=Succeeded and no EvaluationThresholdViolated events.
    """
    payload = build_evalhub_job_payload(
        model_service_name=vllm_service_name,
        tenant_namespace=namespace,
        job_name=job_name,
    )
    benchmark = payload["benchmarks"][0]
    benchmark["primary_score"] = {"metric": "acc_norm", "lower_is_better": False}
    benchmark["pass_criteria"] = {"threshold": threshold_accuracy}
    return payload


def build_threshold_violation_payload(
    vllm_service_name: str,
    namespace: str,
    job_name: str,
    threshold_accuracy: float = LIFECYCLE_THRESHOLD_ACCURACY_HIGH,
) -> dict[str, Any]:
    """Build a job payload with a threshold that the emulator will always fail.

    Uses threshold_accuracy >= 1.01 so any evaluation output triggers ThresholdViolated.
    """
    model_url = f"http://{vllm_service_name}.{namespace}.svc.cluster.local:{EVALHUB_VLLM_EMULATOR_PORT}/v1"
    benchmark = build_vllm_arc_easy_benchmark(num_examples=3)
    benchmark["primary_score"] = {"metric": "acc_norm", "lower_is_better": False}
    benchmark["pass_criteria"] = {"threshold": threshold_accuracy}
    return {
        "name": job_name,
        "model": {"url": model_url, "name": "emulatedModel"},
        "benchmarks": [benchmark],
    }


def build_oom_job_payload(
    vllm_service_name: str,
    namespace: str,
    job_name: str,
    memory_limit: str = LIFECYCLE_OOM_MEMORY_LIMIT,
) -> dict[str, Any]:
    """Build a job payload with an artificially low memory limit.

    The adapter container will be OOMKilled, triggering an operator EvaluationFailed Event.
    """
    model_url = f"http://{vllm_service_name}.{namespace}.svc.cluster.local:{EVALHUB_VLLM_EMULATOR_PORT}/v1"
    benchmark = {
        "id": "arc_easy",
        "provider_id": "lm_evaluation_harness",
        "parameters": {
            "num_examples": 3,
            "tokenizer": "google/flan-t5-small",
        },
        "hardware_config": {
            "memory": {"request": memory_limit, "limit": memory_limit},
        },
    }
    return {
        "name": job_name,
        "model": {"url": model_url, "name": "emulatedModel"},
        "benchmarks": [benchmark],
    }


def build_bad_image_payload(
    vllm_service_name: str,
    namespace: str,
    job_name: str,
    bad_image_provider_id: str,
) -> dict[str, Any]:
    """Build a job payload using a tenant provider whose k8s runtime image does not exist.

    The pod will enter ImagePullBackOff, triggering an operator EvaluationFailed Event.
    """
    model_url = f"http://{vllm_service_name}.{namespace}.svc.cluster.local:{EVALHUB_VLLM_EMULATOR_PORT}/v1"
    benchmark = build_vllm_arc_easy_benchmark(num_examples=3)
    benchmark["provider_id"] = bad_image_provider_id
    return {
        "name": job_name,
        "model": {"url": model_url, "name": "emulatedModel"},
        "benchmarks": [benchmark],
    }


def build_oom_kueue_job_payload(
    vllm_service_name: str,
    namespace: str,
    job_name: str,
    queue_name: str,
    memory_limit: str = LIFECYCLE_OOM_MEMORY_LIMIT,
) -> dict[str, Any]:
    """OOM job payload with Kueue queue on benchmark hardware_config (direct mode)."""
    payload = build_oom_job_payload(
        vllm_service_name=vllm_service_name,
        namespace=namespace,
        job_name=job_name,
        memory_limit=memory_limit,
    )
    payload["benchmarks"][0]["hardware_config"]["queue"] = {
        "kind": "kueue",
        "name": queue_name,
    }
    return payload


def parse_status_annotation(annotation_value: str) -> dict[str, Any]:
    """Parse the evaluation-status annotation JSON. Raises ValueError on invalid JSON."""
    try:
        return json.loads(annotation_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"evaluation-status annotation is not valid JSON: {annotation_value!r}") from exc


_RESOURCE_API_GROUPS: dict[str, str] = {
    "events": "",
    "jobs": "batch",
}


def check_rbac_can_i(
    admin_client: DynamicClient,
    verb: str,
    resource: str,
    sa_namespace: str,
    sa_name: str,
    *,
    target_namespace: str | None = None,
) -> bool:
    """Check if a ServiceAccount has a permission via SubjectAccessReview.

    Args:
        admin_client: Cluster client with permission to create SubjectAccessReviews.
        verb: RBAC verb (e.g. create, patch).
        resource: Resource name (e.g. events, jobs).
        sa_namespace: Namespace where the ServiceAccount lives.
        sa_name: ServiceAccount name.
        target_namespace: Namespace to check permission in. Defaults to sa_namespace.

    Returns:
        True if allowed, False if denied.

    Raises:
        Exception: Kubernetes API errors from the SubjectAccessReview request.
    """
    if resource not in _RESOURCE_API_GROUPS:
        raise ValueError(f"Unsupported resource for RBAC check: {resource!r}")

    permission_ns = target_namespace if target_namespace is not None else sa_namespace
    as_user = f"system:serviceaccount:{sa_namespace}:{sa_name}"
    sar_api = admin_client.resources.get(
        api_version="authorization.k8s.io/v1",
        kind="SubjectAccessReview",
    )
    review = sar_api.create(
        body={
            "apiVersion": "authorization.k8s.io/v1",
            "kind": "SubjectAccessReview",
            "spec": {
                "user": as_user,
                "resourceAttributes": {
                    "namespace": permission_ns,
                    "verb": verb,
                    "group": _RESOURCE_API_GROUPS[resource],
                    "resource": resource,
                    "version": "v1",
                },
            },
        }
    )
    return bool(review.status.allowed)


def find_evalhub_events_role_binding(
    admin_client: DynamicClient,
    *,
    evalhub_cr_name: str,
    evalhub_sa_namespace: str,
    tenant_namespace: str,
) -> RoleBinding:
    """Return the operator-provisioned events RoleBinding for the EvalHub API ServiceAccount."""
    evalhub_sa_name = f"{evalhub_cr_name}-service"
    bindings = list(RoleBinding.get(client=admin_client, namespace=tenant_namespace))
    events_bindings = [
        binding
        for binding in bindings
        if binding.name.startswith(evalhub_cr_name)
        and binding.instance.roleRef.name == EVALHUB_EVENTS_CLUSTERROLE
        and any(
            subject.kind == "ServiceAccount"
            and subject.name == evalhub_sa_name
            and subject.namespace == evalhub_sa_namespace
            for subject in (binding.instance.subjects or [])
        )
    ]
    if len(events_bindings) != 1:
        binding_names = [binding.name for binding in bindings if binding.name.startswith(evalhub_cr_name)]
        raise AssertionError(
            f"Expected exactly one events RoleBinding for SA {evalhub_sa_name!r} in {tenant_namespace!r}, "
            f"found {len(events_bindings)} (candidate bindings: {binding_names})"
        )
    return events_bindings[0]


def _role_binding_restore_body(binding: RoleBinding, tenant_namespace: str) -> dict[str, Any]:
    """Build a create-ready RoleBinding manifest from an existing binding."""
    subjects: list[dict[str, str]] = []
    for subject in binding.instance.subjects or []:
        entry: dict[str, str] = {"kind": subject.kind, "name": subject.name}
        if subject.namespace:
            entry["namespace"] = subject.namespace
        if subject.apiGroup:
            entry["apiGroup"] = subject.apiGroup
        subjects.append(entry)
    return {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "RoleBinding",
        "metadata": {
            "name": binding.name,
            "namespace": tenant_namespace,
            "labels": dict(binding.instance.metadata.labels or {}),
        },
        "roleRef": {
            "apiGroup": binding.instance.roleRef.apiGroup,
            "kind": binding.instance.roleRef.kind,
            "name": binding.instance.roleRef.name,
        },
        "subjects": subjects,
    }


def _restore_role_binding(admin_client: DynamicClient, body: dict[str, Any]) -> None:
    """Recreate a RoleBinding when the operator has not already restored it."""
    name = body["metadata"]["name"]
    namespace = body["metadata"]["namespace"]
    existing = RoleBinding(client=admin_client, name=name, namespace=namespace)
    if existing.exists:
        LOGGER.info("Events RoleBinding already present; skipping restore", name=name, namespace=namespace)
        return
    RoleBinding(client=admin_client, kind_dict=body, teardown=False).create(wait=True)
    LOGGER.info("Restored events RoleBinding", name=name, namespace=namespace)


def wait_until_events_create_denied(
    admin_client: DynamicClient,
    *,
    evalhub_sa_namespace: str,
    evalhub_sa_name: str,
    tenant_namespace: str,
    timeout: int = 30,
) -> None:
    """Wait until SubjectAccessReview reports events create is denied for the EvalHub SA."""
    try:
        for denied in TimeoutSampler(
            wait_timeout=timeout,
            sleep=2,
            func=lambda: (
                not check_rbac_can_i(
                    admin_client=admin_client,
                    verb="create",
                    resource="events",
                    sa_namespace=evalhub_sa_namespace,
                    sa_name=evalhub_sa_name,
                    target_namespace=tenant_namespace,
                )
            ),
        ):
            if denied:
                return
    except TimeoutExpiredError as exc:
        raise AssertionError(
            f"EvalHub SA {evalhub_sa_name!r} still has events create permission in {tenant_namespace!r} "
            f"after RoleBinding deletion"
        ) from exc
    raise AssertionError(
        f"EvalHub SA {evalhub_sa_name!r} still has events create permission in {tenant_namespace!r} "
        f"after RoleBinding deletion"
    )


@contextmanager
def revoked_evalhub_events_create_permission(
    admin_client: DynamicClient,
    *,
    evalhub_cr_name: str,
    evalhub_sa_namespace: str,
    tenant_namespace: str,
) -> Iterator[None]:
    """Temporarily revoke EvalHub events create permission by deleting its events RoleBinding."""
    evalhub_sa_name = f"{evalhub_cr_name}-service"
    binding = find_evalhub_events_role_binding(
        admin_client=admin_client,
        evalhub_cr_name=evalhub_cr_name,
        evalhub_sa_namespace=evalhub_sa_namespace,
        tenant_namespace=tenant_namespace,
    )
    restore_body = _role_binding_restore_body(binding=binding, tenant_namespace=tenant_namespace)
    binding.delete(wait=True)
    try:
        wait_until_events_create_denied(
            admin_client=admin_client,
            evalhub_sa_namespace=evalhub_sa_namespace,
            evalhub_sa_name=evalhub_sa_name,
            tenant_namespace=tenant_namespace,
        )
        yield
    finally:
        try:
            _restore_role_binding(admin_client=admin_client, body=restore_body)
        except Exception:
            LOGGER.exception(
                "Failed to restore events RoleBinding after test",
                name=restore_body["metadata"]["name"],
                namespace=tenant_namespace,
            )
            raise


def is_valid_camel_case(s: str) -> bool:
    """Return True if string matches CamelCase (starts with uppercase, letters only)."""
    return bool(re.match(r"^[A-Z][a-zA-Z]+$", s))
