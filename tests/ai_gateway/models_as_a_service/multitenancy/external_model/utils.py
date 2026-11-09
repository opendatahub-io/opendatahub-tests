"""Shared helpers for multi-tenant external model tests.

Ported from models-as-a-service ``test/e2e/tests/tenant_external_model_helpers.py``
and ``external_model_helpers.py`` (RHAI-12994 P0 scenarios), adapted to the
openshift-python-wrapper / DynamicClient style of this repository.

Covers the AGC/Praxis multi-tenant external model contract:

- ExternalModels in user namespaces explicitly attached to tenant gateways
  via ``spec.gatewayRefs``, with per-gateway ``status.gateways`` results
  when the AGC publishes them (see the status fallback notes below).
- Controller-owned credential copies delivered into the tenant namespace.
- Real inference through a specific tenant gateway against a header-echo
  external endpoint, proving which provider served the request and which
  credential was injected.
- ReferenceGrant-gated cross-namespace providers (P1).

Contracts that live in the ai-gateway-controller repo and are probed
defensively here (assert on observable behavior rather than assumed
resource names):

- Attachment readiness is read from the per-gateway ``status.gateways[]``
  entry when the AGC populates it; AGC builds that report aggregate status
  only fall back to the model's top-level conditions, whose reason/message
  carry the first failing (or the aggregate ready) attachment result —
  gateway reasons such as ``NotAllowedByListeners`` and
  ``ModelNameConflict`` flow through verbatim. Condition types are matched
  across the plausible spellings ("Ready"/"Accepted"/"Attached").
- Provider auth must be ``apikey`` with an ``api-key`` Secret key (the only
  strategy the AGC resolver accepts). Providers in user namespaces stay
  ``Pending``/``TenantNotReady`` by design: the attachment path validates
  them directly and does not gate on their own status.
- Provider DestinationRules are found by content, not hard-coded names: the
  AGC publishes them into the Gateway namespace under hashed names labeled
  with the provider, gateway, and tenant (they carry no ownerReferences).
  Credential copies delivered into the tenant are found by matching
  ``api-key`` Secret data.
- The inference URL is ``{gateway}/{model-namespace}/{modelName}`` plus the
  provider's own path: the AGC's attachment route matches that path prefix
  and rewrites the matched prefix to ``/``, so whatever the client appends
  after the model name reaches the provider verbatim (OpenAI-style clients
  append ``/v1/chat/completions``; the echo endpoint serves
  ``/anything/<tag>``). Models whose client-facing modelName differs from
  their CRD name cannot be called on this stack while
  ``status.resolvedModelAlias`` carries the CRD name instead of
  ``spec.modelName`` (RHOAIENG-101609); the conflicts module gates its
  same-named serving proof on that gap.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import time
import uuid
from collections.abc import Callable, Generator, Iterable
from contextlib import ExitStack, contextmanager
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import NotFoundError
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.deployment import Deployment
from ocp_resources.maas_auth_policy import MaaSAuthPolicy
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.maas_subscription import MaaSSubscription
from ocp_resources.namespace import Namespace
from ocp_resources.project_project_openshift_io import Project
from ocp_resources.resource import ResourceEditor
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_gateway.models_as_a_service.maas_subscription.utils import (
    MAAS_SUBSCRIPTION_NAMESPACE,
)
from tests.ai_gateway.models_as_a_service.multitenancy.aitenant.utils import (
    AITenantTestContext,
    expected_tenant_namespace_name,
)
from tests.ai_gateway.models_as_a_service.multitenancy.utils import (
    gateway_access_label_key,
    gateway_access_namespace_labels,
)
from tests.ai_gateway.models_as_a_service.utils import (
    AITENANT_INFRA_NAMESPACE,
    aitenant_from_spec,
    build_aitenant_spec,
    create_api_key,
    revoke_api_key,
)
from utilities.constants import MAAS_GATEWAY_NAMESPACE
from utilities.general import generate_random_name
from utilities.image_constants import SharedImages
from utilities.infra import create_ns
from utilities.resources.aitenant import AITenant
from utilities.resources.destination_rule import DestinationRule
from utilities.resources.external_model import ExternalModel
from utilities.resources.external_provider import ExternalProvider
from utilities.resources.http_route import HTTPRoute
from utilities.resources.maastenantconfig import MaasTenantConfig

LOGGER = structlog.get_logger(name=__name__)


TENANT_EXTERNAL_MODEL_READY_TIMEOUT = int(os.environ.get("E2E_TENANT_EXTERNAL_MODEL_READY_TIMEOUT", "300"))
TENANT_EXTERNAL_MODEL_POLL_INTERVAL = 5

EXTERNAL_ENDPOINT = os.environ.get("E2E_EXTERNAL_ENDPOINT", "httpbingo.org")

CREDENTIAL_DATA_KEY = "api-key"

EXTERNAL_TARGET_MODEL = "gpt-3.5-turbo"
EXTERNAL_API_FORMAT = "openai-chat"

DEFAULT_TENANT_NAMESPACE = MAAS_SUBSCRIPTION_NAMESPACE
TENANT_CONFIG_NAME = "default-tenant"
ANNOTATION_PAYLOAD_PROCESSING_TYPE = "maas.opendatahub.io/payload-processing-type"

DEFAULT_BACKEND_SWITCH_TIMEOUT = int(os.environ.get("E2E_TENANT_BACKEND_SWITCH_TIMEOUT", "420"))
READY_TIMEOUT = int(os.environ.get("E2E_TENANT_EXTERNAL_MODEL_READY_TIMEOUT", "300"))
INFERENCE_RETRIES = 6
INFERENCE_RETRY_DELAY = 5
INFERENCE_REQUEST_TIMEOUT = 60

SA_SERVICE_CA = "/var/run/secrets/kubernetes.io/serviceaccount/service-ca.crt"
SERVING_CERT_ANNOTATION = "service.beta.openshift.io/serving-cert-secret-name"
LLM_D_INFERENCE_SIM_IMAGE: str = SharedImages.LLMD_INFERENCE_SIM_UPSTREAM

AITENANT_DELETE_TIMEOUT = 600


@contextmanager
def deploy_aitenant_best_effort(
    admin_client: DynamicClient,
    aitenant_name: str,
    cr_namespace: str,
    aitenant_spec: dict[str, Any],
    teardown: bool = True,
) -> Generator[AITenant, Any, Any]:
    """Deploy an AITenant and clean it up with source-suite best-effort semantics.

    Mirrors the source suite's ``cleanup_discovery_case``: AITenant deletion is
    bounded and never fails the test. When the controller's cleanup exceeds
    ``AITENANT_DELETE_TIMEOUT``, the ``aitenant-cleanup`` finalizer is cleared
    so the tenant object goes away instead of wedging forever once its Gateway
    is removed; the tenant namespace is deleted afterwards either way.
    """
    with aitenant_from_spec(
        admin_client=admin_client,
        aitenant_name=aitenant_name,
        cr_namespace=cr_namespace,
        aitenant_spec=aitenant_spec,
        teardown=False,
    ) as aitenant:
        try:
            yield aitenant
        finally:
            if teardown:
                try:
                    delete_timed_out = not aitenant.delete(wait=True, timeout=AITENANT_DELETE_TIMEOUT)
                except Exception as error:  # noqa: BLE001 - teardown stays best-effort
                    LOGGER.warning(f"failed to delete AITenant {cr_namespace}/{aitenant_name}: {error}")
                    delete_timed_out = False
                if delete_timed_out:
                    LOGGER.warning(
                        f"AITenant {cr_namespace}/{aitenant_name} was not deleted within "
                        f"{AITENANT_DELETE_TIMEOUT}s; clearing the aitenant-cleanup finalizer "
                        "(best-effort cleanup, see the source suite's cleanup_discovery_case)"
                    )
                    try:
                        ResourceEditor(
                            patches={aitenant: {"metadata": {"finalizers": []}}},
                        ).update()
                        aitenant.wait_deleted(timeout=60)
                    except Exception as error:  # noqa: BLE001 - teardown stays best-effort
                        LOGGER.warning(
                            f"failed to clear the finalizer/delete AITenant {cr_namespace}/{aitenant_name}: {error}"
                        )
                try:
                    tenant_namespace = Namespace(
                        client=admin_client,
                        name=expected_tenant_namespace_name(aitenant_name=aitenant_name),
                    )
                    if tenant_namespace.exists:
                        tenant_namespace.delete(wait=False)
                        LOGGER.info(f"Deleted tenant namespace '{tenant_namespace.name}'")
                except Exception as error:  # noqa: BLE001 - teardown stays best-effort
                    LOGGER.warning(
                        f"failed to delete tenant namespace of AITenant {cr_namespace}/{aitenant_name}: {error}"
                    )


AGC_DEPLOYMENT_NAME = os.environ.get("AGC_DEPLOYMENT_NAME", "ai-gateway-controller")
AGC_DEPLOYMENT_NAMESPACE = os.environ.get("AGC_DEPLOYMENT_NAMESPACE", "")
AGC_DEPLOYMENT_NAMESPACE_CANDIDATES = tuple(
    dict.fromkeys([
        AGC_DEPLOYMENT_NAMESPACE,
        py_config.get("applications_namespace", ""),
        "redhat-ods-applications",
        "odh-applications",
        "opendatahub",
    ])
)

EXTERNAL_MODEL_CRD_NAME = "externalmodels.inference.opendatahub.io"
REFERENCE_GRANT_CRD_NAME = "referencegrants.gateway.networking.k8s.io"

READY_CONDITION_TYPES = ("Ready", "Accepted", "Attached")

AGC_LONGEST_RESOURCE_BASE = "payload-processing-external-model-filters-"
MAX_AITENANT_NAME_LENGTH = 63 - len(AGC_LONGEST_RESOURCE_BASE)

ATTACHMENT_ROUTE_HASH_RE = re.compile(r"[0-9a-f]{12}")

PER_TENANT_PROCESSING_DEPLOYMENT_PREFIX = "payload-processing-"
PRAXIS_IMAGE_MARKER = "praxis"
TENANT_EXTERNAL_MODEL_NAME_PREFIX = "e2e-extmod"
# Attempts to generate an unused AITenant name before giving up.
AITENANT_NAME_MAX_ATTEMPTS = 5


_JSON_WRAPPERS: dict[str, type] = {
    "secret": Secret,
    "namespace": Namespace,
    "deployment": Deployment,
    "httproute": HTTPRoute,
    "destinationrule": DestinationRule,
    "externalmodel": ExternalModel,
    "externalprovider": ExternalProvider,
    "maasmodelref": MaaSModelRef,
    "maasauthpolicy": MaaSAuthPolicy,
    "maassubscription": MaaSSubscription,
    "maastenantconfig": MaasTenantConfig,
    "crd": CustomResourceDefinition,
    "service": Service,
}


def _instance_to_dict(instance: Any) -> dict[str, Any]:
    """Normalize a wrapper ``instance`` (ResourceInstance or dict) to a dict."""
    if isinstance(instance, dict):
        return instance
    return instance.to_dict()


def get_json_or_none(
    client: DynamicClient,
    kind: str,
    name: str,
    namespace: str | None = None,
) -> dict[str, Any] | None:
    """Return one resource's raw JSON dict, or None when it does not exist."""
    wrapper = _JSON_WRAPPERS[kind]
    try:
        instances = list(wrapper.get(client=client, name=name, namespace=namespace))
        return _instance_to_dict(instances[0].instance) if instances else None
    except NotFoundError, TimeoutExpiredError:
        return None


def list_json(
    client: DynamicClient,
    kind: str,
    namespace: str | None = None,
) -> list[dict[str, Any]]:
    """Return raw JSON dicts for every resource of *kind* in *namespace*.

    A resource deleted between the list and its (lazy, per-name) instance
    fetch raises NotFoundError after the wrapper's retry budget: when
    polling for resources to disappear that means "not there", so such
    items are skipped instead of failing the poll.
    """
    result: list[dict[str, Any]] = []
    for instance in _JSON_WRAPPERS[kind].get(client=client, namespace=namespace):
        try:
            result.append(_instance_to_dict(instance.instance))
        except NotFoundError, TimeoutExpiredError:
            continue
    return result


def wait_until(
    check: Callable[[], Any],
    timeout: int,
    what: str,
    interval: int = TENANT_EXTERNAL_MODEL_POLL_INTERVAL,
) -> Any:
    """Poll *check* until it stops raising, within a bounded time budget."""
    try:
        for result in TimeoutSampler(
            wait_timeout=timeout,
            sleep=interval,
            func=check,
            exceptions_dict={AssertionError: []},
        ):
            return result
    except TimeoutExpiredError as exc:
        raise TimeoutError(f"timed out after {timeout}s waiting for: {what}") from exc


def wait_for_not_found(
    client: DynamicClient,
    kind: str,
    name: str,
    namespace: str | None = None,
    timeout: int = TENANT_EXTERNAL_MODEL_READY_TIMEOUT,
) -> None:
    """Wait until a resource is actually gone (a failed delete cannot pass vacuously)."""

    def _check() -> bool:
        assert get_json_or_none(client, kind, name, namespace) is None, f"{kind}/{namespace}/{name} still exists"
        return True

    wait_until(
        check=_check,
        timeout=timeout,
        what=f"{kind}/{namespace}/{name} deletion",
    )


def assert_during_settle_window(
    check: Callable[[], None],
    timeout: int,
    what: str,
    interval: int = TENANT_EXTERNAL_MODEL_POLL_INTERVAL,
) -> None:
    """Assert a check keeps passing for a whole settle window.

    The inverse of wait_until: exhausting the window is the success path —
    the check proves a state never changes during the window (nothing
    appears, nothing is recreated) rather than waiting for a target state.
    """
    try:
        for _ in TimeoutSampler(
            wait_timeout=timeout,
            sleep=interval,
            func=check,
            exceptions_dict={},
        ):
            pass
    except TimeoutExpiredError as exc:
        if exc.last_exp is not None:
            raise TimeoutError(f"{what} did not stay absent during the {timeout}s settle window") from exc.last_exp
        return


def wait_for_model_attached_to_gateways(
    admin_client: DynamicClient,
    model: dict[str, Any],
    cases: Iterable[dict[str, Any]],
) -> None:
    """Wait until a model is fully attached to every given tenant gateway.

    Fully attached means: the gateway attachment is Ready, the attachment's
    HTTPRoute exists in the model namespace, the tenant's credential copy
    exists, and the provider DestinationRule for that gateway references
    the model.
    """
    for case in cases:
        wait_for_gateway_attachment_ready(
            admin_client=admin_client,
            model_name=model["name"],
            model_namespace=model["namespace"],
            gateway_name=case["gateway_name"],
        )
        wait_for_attachment_httproute(
            admin_client=admin_client,
            route_namespace=model["namespace"],
            gateway_name=case["gateway_name"],
            model_name=model["name"],
        )
        wait_for_credential_copy(
            admin_client=admin_client,
            tenant_namespace=case["tenant_ns"],
            source_secret_name=model["secret_name"],
            source_namespace=model["secret_namespace"],
        )
        assert destination_rules_referencing(  # noqa: FCN001 - *namespaces must follow positionally
            admin_client, model["name"], MAAS_GATEWAY_NAMESPACE, also_matching=(case["gateway_name"],)
        ), f"provider DestinationRule for {model['name']} on {case['gateway_name']} must exist while attached"


def delete_external_model_and_ref(
    admin_client: DynamicClient,
    name: str,
    namespace: str,
    timeout: int = TENANT_EXTERNAL_MODEL_READY_TIMEOUT,
) -> None:
    """Delete a model's MaaSModelRef and ExternalModel, waiting for each to be gone."""
    MaaSModelRef(client=admin_client, name=name, namespace=namespace, ensure_exists=True).delete()
    wait_for_not_found(client=admin_client, kind="maasmodelref", name=name, namespace=namespace, timeout=timeout)
    ExternalModel(client=admin_client, name=name, namespace=namespace, ensure_exists=True).delete()
    wait_for_not_found(client=admin_client, kind="externalmodel", name=name, namespace=namespace, timeout=timeout)


def redact_sensitive(value: Any, *, max_length: int = 300) -> str:
    """Redact secret-looking fields and truncate for log/assert messages."""
    text = repr(value)
    for needle in ("sk-", "api-key", "password", "token"):
        text = re.sub(rf"{needle}[^'\"\s,}}]*", f"{needle}***", text, flags=re.IGNORECASE)
    return text if len(text) <= max_length else f"{text[:max_length]}..."


def _crd_schema_properties(crd: dict[str, Any], section: str | None) -> dict[str, Any]:
    """Merge the schema properties of *section* ("spec"/"status") across served versions.

    Unserved versions are skipped: Kubernetes only accepts objects through a
    served version, so fields that exist solely in an unserved one cannot be
    used and must not enable suite capabilities.
    """
    merged: dict[str, Any] = {}
    for version in (crd.get("spec") or {}).get("versions") or []:
        if not version.get("served"):
            continue
        schema = ((version.get("schema") or {}).get("openAPIV3Schema") or {}).get("properties") or {}
        if section is None:
            merged.update(schema)
        else:
            merged.update(schema.get(section, {}).get("properties") or {})
    return merged


def installed_crd_capabilities(admin_client: DynamicClient, crd: dict[str, Any] | None) -> dict[str, bool]:
    """Report which additive fields the installed ExternalModel CRD carries."""
    crd = crd if crd is not None else get_json_or_none(admin_client, "crd", EXTERNAL_MODEL_CRD_NAME)
    if not crd:
        return {
            "gatewayRefs": False,
            "statusGateways": False,
            "providerNamespaceRef": False,
            "perRefAuthOverride": False,
        }
    spec_props = _crd_schema_properties(crd=crd, section="spec")
    status_props = _crd_schema_properties(crd=crd, section="status")
    provider_ref_items = ((spec_props.get("externalProviderRefs") or {}).get("items") or {}).get("properties") or {}
    ref_props = (provider_ref_items.get("ref") or {}).get("properties") or {}
    return {
        "gatewayRefs": "gatewayRefs" in spec_props,
        "statusGateways": "gateways" in status_props,
        "providerNamespaceRef": "namespace" in ref_props,
        "perRefAuthOverride": "auth" in provider_ref_items,
    }


def find_agc_deployment(admin_client: DynamicClient) -> dict[str, Any] | None:
    """Locate the ai-gateway-controller Deployment, if installed.

    Searches the well-known deployment namespaces (explicit env override
    first, then the operator applications namespace and the companion-deploy
    defaults); falls back to a name match in case the companion deploy
    renames it.
    """
    for namespace in AGC_DEPLOYMENT_NAMESPACE_CANDIDATES:
        if not namespace:
            continue
        deployment = get_json_or_none(
            client=admin_client, kind="deployment", name=AGC_DEPLOYMENT_NAME, namespace=namespace
        )
        if deployment:
            return deployment
        for candidate in list_json(admin_client, "deployment", namespace):
            name = (candidate.get("metadata") or {}).get("name") or ""
            if "ai-gateway-controller" in name or name.startswith("aigc"):
                return candidate
    return None


def deployment_image(deployment: dict[str, Any] | None) -> str:
    """First container image of a Deployment JSON dict."""
    if not deployment:
        return ""
    containers = ((deployment.get("spec") or {}).get("template") or {}).get("spec", {}).get("containers") or []
    return (containers[0].get("image") if containers else "") or ""


def per_tenant_processing_uses_praxis(
    admin_client: DynamicClient,
    tenant_name: str,
    *,
    timeout: int = 120,
) -> bool:
    """Whether the tenant's payload-processing Deployment runs praxis-extproc.

    Polls for the Deployment to exist and settle before deciding: a single
    read can race the AGC's per-tenant reconcile and would skip the module on
    a cluster that actually supports the contract. Returns False only after
    the bounded wait confirms the image is not praxis (or never appears).
    """
    deployment_name = f"{PER_TENANT_PROCESSING_DEPLOYMENT_PREFIX}{tenant_name}"

    def _check() -> None:
        deployment = get_json_or_none(
            client=admin_client,
            kind="deployment",
            name=deployment_name,
            namespace=MAAS_GATEWAY_NAMESPACE,
        )
        assert deployment is not None, f"Deployment {MAAS_GATEWAY_NAMESPACE}/{deployment_name} not found"
        assert PRAXIS_IMAGE_MARKER in deployment_image(deployment), (
            f"Deployment {MAAS_GATEWAY_NAMESPACE}/{deployment_name} image "
            f"{deployment_image(deployment)!r} is not praxis-extproc"
        )

    try:
        wait_until(
            check=_check,
            timeout=timeout,
            what=f"praxis image on {MAAS_GATEWAY_NAMESPACE}/{deployment_name}",
        )
    except TimeoutError:
        return False
    return True


def extmod_tenant_test_params(
    tenant_label: str,
    admin_client: DynamicClient,
    cr_namespace: str = AITENANT_INFRA_NAMESPACE,
) -> dict[str, Any]:
    """AITenant name and spec for one external model suite tenant.

    The random suffix is regenerated while the candidate name is already taken
    by a retained AITenant from an earlier run, so provisioning never collides
    with stale tenants.
    """
    for _ in range(AITENANT_NAME_MAX_ATTEMPTS):
        aitenant_name = f"{TENANT_EXTERNAL_MODEL_NAME_PREFIX}-{tenant_label}-{generate_random_name(length=4)}"
        assert len(aitenant_name) <= MAX_AITENANT_NAME_LENGTH, (
            f"tenant name {aitenant_name!r} is {len(aitenant_name)} characters; AGC per-tenant "
            f"praxis resource naming needs it to stay at or under {MAX_AITENANT_NAME_LENGTH} "
            "(63-char limit minus the longest resource base name)"
        )
        if not AITenant(client=admin_client, name=aitenant_name, namespace=cr_namespace).exists:
            break
        LOGGER.info(f"extmod_tenant_test_params: AITenant {aitenant_name} already exists, regenerating suffix")
    else:
        raise AssertionError(
            f"could not generate an unused AITenant name for label {tenant_label!r} "
            f"after {AITENANT_NAME_MAX_ATTEMPTS} attempts"
        )
    return {
        "aitenant_name": aitenant_name,
        "aitenant_spec": build_aitenant_spec(aitenant_name=aitenant_name),
    }


def tenant_case(
    context: AITenantTestContext,
    gateway_name: str,
    maas_host: str,
) -> dict[str, str]:
    """Enrich an AITenant context with the external model suite's case keys."""
    gateway_host = f"{gateway_name}.{maas_host.removeprefix('maas.')}"
    return {
        "tenant_label_name": context["aitenant_name"],
        "tenant_ns": context["tenant_namespace_name"],
        "gateway_name": gateway_name,
        "gateway_host": gateway_host,
        "gateway_url": f"https://{gateway_host}",
        "maas_api_url": f"https://{gateway_host}/maas-api",
    }


def require_tenant_external_model_support(admin_client: DynamicClient) -> dict[str, bool]:
    """Fail unless the installed cluster can serve multi-tenant external models.

    Requires the additive schema (gatewayRefs + status.gateways) in the
    *installed* ExternalModel CRD and a deployed ai-gateway-controller;
    missing prerequisites are a setup failure, not a skip.
    """
    crd = get_json_or_none(client=admin_client, kind="crd", name=EXTERNAL_MODEL_CRD_NAME)
    if crd is None:
        pytest.fail(f"ExternalModel CRD {EXTERNAL_MODEL_CRD_NAME} is not installed")
    capabilities = installed_crd_capabilities(admin_client=admin_client, crd=crd)
    if not (capabilities["gatewayRefs"] and capabilities["statusGateways"]):
        pytest.fail(
            "installed ExternalModel CRD lacks the gatewayRefs additive schema "
            f"(capabilities: {capabilities}); AGC gatewayRefs support not distributed"
        )
    if find_agc_deployment(admin_client) is None:
        pytest.fail(
            f"ai-gateway-controller deployment not found "
            f"({AGC_DEPLOYMENT_NAME} in {AGC_DEPLOYMENT_NAMESPACE_CANDIDATES}); AGC companion not installed"
        )
    return capabilities


def env_bool(name: str, default: bool = False) -> bool:
    """Environment flag as bool (the MaaS-suite gate convention)."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


_WITHDRAWAL_ISSUE_FLAGS = {
    "#106": "TENANT_EXTERNAL_MODEL_LAST_PROVIDER_WITHDRAWAL_E2E",
    "#108": "TENANT_EXTERNAL_MODEL_PARTIAL_WITHDRAWAL_E2E",
}


def withdrawal_gate_enabled(issue: str) -> bool:
    """Effective per-issue withdrawal gate state (per-issue flag OR umbrella)."""
    return env_bool(_WITHDRAWAL_ISSUE_FLAGS[issue]) or env_bool("TENANT_EXTERNAL_MODEL_WITHDRAWAL_E2E")


def record_tenant_external_model_evidence(admin_client: DynamicClient) -> dict[str, Any]:
    """Write installed-version/configuration evidence for this suite (best effort).

    The acceptance criteria require recording installed AGC/Praxis versions
    and configuration as completion evidence. Neither the cluster lookups
    nor the artifact write ever fail tests.
    """
    from tests.ai_gateway.models_as_a_service.observability.utils import (
        MAAS_CONTROLLER_DEPLOYMENT_NAME,
    )

    def _safe(func: Callable[[], Any], default: Any) -> Any:
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - evidence is best effort
            LOGGER.warning(f"evidence lookup failed: {exc}")
            return default

    evidence = {
        "recordedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "agc": {
            "deploymentName": AGC_DEPLOYMENT_NAME,
            "deploymentNamespace": AGC_DEPLOYMENT_NAMESPACE or "auto-discovered",
            "deploymentNamespaceCandidates": list(AGC_DEPLOYMENT_NAMESPACE_CANDIDATES),
            "image": _safe(
                lambda: deployment_image(find_agc_deployment(admin_client)),
                "",
            ),
        },
        "praxis": {
            "defaultPayloadProcessingImage": _safe(
                lambda: deployment_image(
                    get_json_or_none(admin_client, "deployment", "payload-processing", MAAS_GATEWAY_NAMESPACE)
                ),
                "",
            ),
        },
        "maasController": {
            "deploymentName": MAAS_CONTROLLER_DEPLOYMENT_NAME,
        },
        "externalModelCrd": _safe(lambda: installed_crd_capabilities(admin_client, None), {}),
        "referenceGrantCrdInstalled": _safe(
            lambda: get_json_or_none(admin_client, "crd", REFERENCE_GRANT_CRD_NAME) is not None,
            False,
        ),
        "gates": {
            "TENANT_EXTERNAL_MODEL_LEGACY_IPP_E2E": env_bool("TENANT_EXTERNAL_MODEL_LEGACY_IPP_E2E"),
            "TENANT_EXTERNAL_MODEL_LAST_PROVIDER_WITHDRAWAL_E2E": withdrawal_gate_enabled("#106"),
            "TENANT_EXTERNAL_MODEL_PARTIAL_WITHDRAWAL_E2E": withdrawal_gate_enabled("#108"),
            "TENANT_EXTERNAL_MODEL_WITHDRAWAL_E2E": env_bool("TENANT_EXTERNAL_MODEL_WITHDRAWAL_E2E"),
        },
        "externalEndpoint": EXTERNAL_ENDPOINT,
    }
    try:
        artifacts_dir = os.environ.get("ARTIFACTS_DIR") or "reports"
        os.makedirs(artifacts_dir, exist_ok=True)
        worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
        path = os.path.join(artifacts_dir, f"tenant-external-model-evidence-{worker}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(evidence, handle, indent=2, sort_keys=True)
        LOGGER.info(f"tenant external model evidence written to {path}")
    except OSError as exc:
        LOGGER.warning(f"failed to write tenant external model evidence artifact: {exc}")
    return evidence


def maas_model_ref_alias(admin_client: DynamicClient, namespace: str, name: str) -> str:
    """The MaaSModelRef's ``status.resolvedModelAlias`` (model identity for gateway auth)."""
    ref = get_json_or_none(admin_client, "maasmodelref", name, namespace) or {}
    return ((ref.get("status") or {}).get("resolvedModelAlias")) or ""


def echo_provider_path(path_tag: str) -> str:
    """Provider path of an echo binding: the echo endpoint serves ``/anything/<tag>``."""
    return f"/anything/{path_tag}"


def external_model_binding(
    provider_name: str,
    *,
    provider_namespace: str | None = None,
    path_tag: str | None = None,
    provider_path: str | None = None,
    credential_secret: str | None = None,
    target_model: str = EXTERNAL_TARGET_MODEL,
) -> dict[str, Any]:
    """One ``spec.externalProviderRefs[]`` entry (see ``external_model_stack``)."""
    ref: dict[str, str] = {"name": provider_name}
    if provider_namespace:
        ref["namespace"] = provider_namespace
    binding: dict[str, Any] = {
        "ref": ref,
        "targetModel": target_model,
        "apiFormat": EXTERNAL_API_FORMAT,
        "path": provider_path or echo_provider_path(path_tag or uuid.uuid4().hex[:10]),
    }
    if credential_secret:
        binding["auth"] = {
            "type": "apikey",
            "secretRef": {"name": credential_secret},
        }
    return binding


def gateway_ref(gateway_name: str, gateway_namespace: str = MAAS_GATEWAY_NAMESPACE) -> dict[str, str]:
    """One ``spec.gatewayRefs[]`` entry."""
    return {"name": gateway_name, "namespace": gateway_namespace}


def external_model_route_name(model_name: str) -> str:
    """AGC tenant-local route name for an ExternalModel (``external-model-<name>``)."""
    return f"external-model-{model_name}"


class TenantPinnedMaaSModelRef(MaaSModelRef):
    """MaaSModelRef with ``spec.tenantRef`` (per-tenant refs for shared models).

    The generated ocp_resources wrapper does not carry the tenantRef field;
    models attached to several gateways require one modelRef per tenant with
    ``spec.tenantRef`` pinned (maas-controller's selectGatewayRoute never
    falls back to another gateway).
    """

    def __init__(self, tenant_ref: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tenant_ref = tenant_ref

    def to_dict(self) -> None:
        super().to_dict()
        if self.tenant_ref is not None and not self.kind_dict and not self.yaml_file:
            self.res.setdefault("spec", {})
            self.res["spec"]["tenantRef"] = self.tenant_ref


@contextmanager
def external_model_stack(
    admin_client: DynamicClient,
    *,
    name: str,
    namespace: str,
    gateway_refs: list[dict[str, str]] | None = None,
    model_name: str | None = None,
    tenant_refs: list[str] | None = None,
    provider_name: str | None = None,
    provider_namespace: str | None = None,
    credential: str | None = None,
    credential_secret_name: str | None = None,
    override_credential: str | None = None,
    override_secret_name: str | None = None,
    endpoint: str = EXTERNAL_ENDPOINT,
    provider_path: str | None = None,
    provider_config: dict[str, str] | None = None,
    teardown: bool = True,
) -> Generator[dict[str, Any], Any, Any]:
    """Create the full serving stack for one external model and clean it up.

    Credential Secret (in the provider namespace) + ExternalProvider +
    ExternalModel + MaaSModelRef, all under one context manager.``override_credential``
    additionally creates a model-namespace Secret wired as the per-model auth
    override. ``endpoint``/``provider_path``/``credential`` re-point the
    provider away from the header-echo default. ``tenant_refs`` serves a
    model shared across tenant gateways: instead of one tenantRef-less
    MaaSModelRef named after the model, it creates one modelRef per tenant
    (``<name>-<tenant>``, ``spec.tenantRef`` pinned).

    Yields identifiers used by the assertions: metadata/client names, the
    provider binding's echo ``path_tag`` (identifies which provider served a
    request) and the credential markers (identify which Secret was injected).
    """
    suffix = uuid.uuid4().hex[:6]
    resolved_credential = credential or f"sk-extmod-{suffix}"  # pragma: allowlist secret
    resolved_provider_name = provider_name or f"{name}-provider"
    resolved_provider_namespace = provider_namespace or namespace
    secret_name = credential_secret_name or f"{resolved_provider_name}-cred"
    path_tag = f"{name}-{suffix}"

    with ExitStack() as stack:
        stack.enter_context(
            cm=Secret(
                client=admin_client,
                name=secret_name,
                namespace=resolved_provider_namespace,
                type="Opaque",
                string_data={CREDENTIAL_DATA_KEY: resolved_credential},
                teardown=teardown,
                wait_for_resource=True,
            )
        )
        stack.enter_context(
            cm=ExternalProvider(
                client=admin_client,
                name=resolved_provider_name,
                namespace=resolved_provider_namespace,
                provider="openai",
                endpoint=endpoint,
                auth={"type": "apikey", "secretRef": {"name": secret_name}},
                config=provider_config,
                teardown=teardown,
                wait_for_resource=True,
            )
        )

        override_secret_name_resolved: str | None = None
        if override_credential is not None:
            override_secret_name_resolved = override_secret_name or f"{name}-override-cred"
            stack.enter_context(
                cm=Secret(
                    client=admin_client,
                    name=override_secret_name_resolved,
                    namespace=namespace,
                    type="Opaque",
                    string_data={CREDENTIAL_DATA_KEY: override_credential},
                    teardown=teardown,
                    wait_for_resource=True,
                )
            )

        stack.enter_context(
            cm=ExternalModel(
                client=admin_client,
                name=name,
                namespace=namespace,
                external_provider_refs=[
                    external_model_binding(
                        provider_name=resolved_provider_name,
                        provider_namespace=resolved_provider_namespace
                        if resolved_provider_namespace != namespace
                        else None,
                        path_tag=path_tag,
                        provider_path=provider_path,
                        credential_secret=override_secret_name_resolved,
                    )
                ],
                gateway_refs=gateway_refs,
                model_name=model_name,
                teardown=teardown,
                wait_for_resource=True,
            )
        )

        model_refs: dict[str, str] = {}
        if tenant_refs:
            for tenant_ref in tenant_refs:
                ref_name = f"{name}-{tenant_ref}"
                stack.enter_context(
                    cm=TenantPinnedMaaSModelRef(
                        client=admin_client,
                        name=ref_name,
                        namespace=namespace,
                        model_ref={
                            "name": name,
                            "namespace": namespace,
                            "kind": "ExternalModel",
                            "apiGroup": "inference.opendatahub.io",
                        },
                        tenant_ref=tenant_ref,
                        teardown=teardown,
                        wait_for_resource=True,
                    )
                )
                model_refs[tenant_ref] = ref_name
        else:
            stack.enter_context(
                cm=MaaSModelRef(
                    client=admin_client,
                    name=name,
                    namespace=namespace,
                    model_ref={
                        "name": name,
                        "namespace": namespace,
                        "kind": "ExternalModel",
                        "apiGroup": "inference.opendatahub.io",
                    },
                    teardown=teardown,
                    wait_for_resource=True,
                )
            )

        yield {
            "name": name,
            "namespace": namespace,
            "client_name": model_name or name,
            "model_refs": model_refs,
            "provider": resolved_provider_name,
            "provider_namespace": resolved_provider_namespace,
            "secret_name": secret_name,
            "secret_namespace": resolved_provider_namespace,
            "credential": resolved_credential,
            "override_credential": override_credential,
            "override_secret": override_secret_name_resolved,
            "path_tag": path_tag,
        }


def patch_external_model(
    admin_client: DynamicClient,
    name: str,
    namespace: str,
    fields: dict[str, Any],
) -> None:
    """Merge-patch an ExternalModel's spec (gatewayRefs / provider set edits)."""
    external_model = ExternalModel(
        client=admin_client,
        name=name,
        namespace=namespace,
        ensure_exists=True,
    )
    ResourceEditor(patches={external_model: {"spec": fields}}).update()


def patch_external_model_gateway_refs(
    admin_client: DynamicClient,
    name: str,
    namespace: str,
    refs: list[dict[str, str]] | None,
) -> None:
    """Set (or remove, with None) an ExternalModel's ``spec.gatewayRefs``."""
    patch_external_model(admin_client=admin_client, name=name, namespace=namespace, fields={"gatewayRefs": refs})


def gateway_status_entry(
    admin_client: DynamicClient,
    model_name: str,
    model_namespace: str,
    gateway_name: str,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
) -> dict[str, Any] | None:
    """Return the status.gateways entry for one gateway attachment."""
    external_model = get_json_or_none(
        client=admin_client, kind="externalmodel", name=model_name, namespace=model_namespace
    )
    if not external_model:
        return None
    for entry in (external_model.get("status") or {}).get("gateways") or []:
        if entry.get("name") == gateway_name and entry.get("namespace") == gateway_namespace:
            return entry
    return None


def _attachment_conditions(
    external_model: dict[str, Any],
    gateway_name: str,
    gateway_namespace: str,
) -> tuple[list[dict[str, Any]], str]:
    """Conditions describing one gateway attachment (per-gateway entry preferred)."""
    entries = (external_model.get("status") or {}).get("gateways") or []
    if not entries:
        # Aggregate-status AGC builds report no per-gateway results at all;
        # there the model-level conditions are the only available signal.
        status = external_model.get("status") or {}
        return (status.get("conditions") or []), "aggregate model status"
    entry = next(
        (item for item in entries if item.get("name") == gateway_name and item.get("namespace") == gateway_namespace),
        None,
    )
    # Once per-gateway results exist, they are authoritative for the requested
    # gateway: falling back to the aggregate conditions could report a false
    # "ready" for a gateway that has no entry (or no conditions) yet. Callers
    # poll through wait_until, so these assertions retry until the entry
    # appears.
    assert entry is not None, (
        f"no status.gateways entry for {gateway_namespace}/{gateway_name} yet; entries: {redact_sensitive(entries)}"
    )
    conditions = entry.get("conditions") or []
    assert conditions, f"no conditions yet in the {gateway_namespace}/{gateway_name} entry: {redact_sensitive(entry)}"
    return conditions, f"attachment to {gateway_namespace}/{gateway_name}"


def _wait_for_gateway_attachment(
    admin_client: DynamicClient,
    model_name: str,
    model_namespace: str,
    gateway_name: str,
    *,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
    matcher: Callable[[list[dict[str, Any]], str], Any],
    what: str,
    timeout: int | None = None,
) -> Any:
    def _check() -> Any:
        external_model = get_json_or_none(
            client=admin_client, kind="externalmodel", name=model_name, namespace=model_namespace
        )
        assert external_model is not None, f"externalmodel/{model_namespace}/{model_name} not found"
        conditions, where = _attachment_conditions(
            external_model=external_model, gateway_name=gateway_name, gateway_namespace=gateway_namespace
        )
        assert conditions, (
            f"no attachment conditions for {gateway_namespace}/{gateway_name} on "
            f"{model_namespace}/{model_name}; status: {redact_sensitive(external_model.get('status'))}"
        )
        return matcher(conditions, where)  # noqa: FCN001 - Callable params take no keywords

    return wait_until(
        check=_check,
        timeout=timeout or TENANT_EXTERNAL_MODEL_READY_TIMEOUT,
        what=f"{model_namespace}/{model_name} attachment to {gateway_namespace}/{gateway_name}: {what}",
    )


def wait_for_gateway_attachment_ready(
    admin_client: DynamicClient,
    model_name: str,
    model_namespace: str,
    gateway_name: str,
    *,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Wait until one gateway attachment reports serving readiness."""

    def _matcher(conditions: list[dict[str, Any]], where: str) -> dict[str, Any]:
        by_type = {condition.get("type"): condition for condition in conditions}
        # The first condition type that is present is authoritative: an
        # explicit Ready=False must fail the wait instead of falling through
        # to Accepted/Attached.
        for condition_type in READY_CONDITION_TYPES:
            condition = by_type.get(condition_type)
            if condition is None:
                continue
            if condition.get("status") == "True":
                return condition
            raise AssertionError(
                f"attachment reports {condition_type}={condition.get('status')} on {where}; conditions: {conditions}"
            )
        raise AssertionError(f"attachment not ready on {where}; conditions: {conditions}")

    return _wait_for_gateway_attachment(
        admin_client=admin_client,
        model_name=model_name,
        model_namespace=model_namespace,
        gateway_name=gateway_name,
        gateway_namespace=gateway_namespace,
        matcher=_matcher,
        what="report Ready",
        timeout=timeout,
    )


def wait_for_gateway_attachment_reason(
    admin_client: DynamicClient,
    model_name: str,
    model_namespace: str,
    gateway_name: str,
    reason: str,
    *,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Wait until one gateway attachment reports a condition with *reason*.

    Used for NotAllowedByListeners (gateway-attachment authorization) and
    ModelNameConflict (client-facing name conflicts): the reason is the
    contract, matched on the per-gateway ``status.gateways`` entry when
    present, otherwise on the model's aggregate conditions.
    """

    def _matcher(conditions: list[dict[str, Any]], where: str) -> dict[str, Any]:
        matches = [condition for condition in conditions if condition.get("reason") == reason]
        assert matches, f"no condition with reason {reason!r} on {where}; conditions: {conditions}"
        return matches[0]

    return _wait_for_gateway_attachment(
        admin_client=admin_client,
        model_name=model_name,
        model_namespace=model_namespace,
        gateway_name=gateway_name,
        gateway_namespace=gateway_namespace,
        matcher=_matcher,
        what=f"report reason {reason}",
        timeout=timeout,
    )


def wait_for_gateway_attachment_absent(
    admin_client: DynamicClient,
    model_name: str,
    model_namespace: str,
    gateway_name: str,
    *,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
    timeout: int | None = None,
) -> None:
    """Wait until the status.gateways entry for a detached gateway is gone."""

    def _check() -> bool:
        entry = gateway_status_entry(
            admin_client=admin_client,
            model_name=model_name,
            model_namespace=model_namespace,
            gateway_name=gateway_name,
            gateway_namespace=gateway_namespace,
        )
        assert entry is None, f"status.gateways entry still present: {redact_sensitive(entry)}"
        return True

    wait_until(
        check=_check,
        timeout=timeout or TENANT_EXTERNAL_MODEL_READY_TIMEOUT,
        what=f"{model_namespace}/{model_name} detachment from {gateway_namespace}/{gateway_name}",
    )


def external_model_reports_maas_route(
    admin_client: DynamicClient,
    model_name: str,
    model_namespace: str,
) -> bool:
    """Whether the MaaS stack can resolve this ExternalModel's serving route.

    maas-controller resolves an attached model's HTTPRoute from the AGC's
    per-gateway ``status.gateways[]`` results or, on aggregate-status AGC
    builds, from ``status.httpRouteName`` — which those builds write only for
    single-attachment models (RHOAIENG-90802 gate).
    """
    external_model = get_json_or_none(
        client=admin_client, kind="externalmodel", name=model_name, namespace=model_namespace
    )
    if not external_model:
        return False
    status = external_model.get("status") or {}
    if status.get("httpRouteName"):
        return True
    for entry in status.get("gateways") or []:
        if not entry.get("httpRouteRef"):
            continue
        if any(
            condition.get("type") in READY_CONDITION_TYPES and condition.get("status") == "True"
            for condition in entry.get("conditions") or []
        ):
            return True
    return False


def route_is_for_model(route: dict[str, Any], model_name: str) -> bool:
    """Whether an AGC-managed HTTPRoute belongs to the given ExternalModel.

    Route names carry the model: ``external-model-<name>`` for tenant-local
    models and ``external-model-<name>-<hash>`` for cross-namespace
    attachments, where the hash is the controller's shortHash — exactly 12
    lowercase hex characters. Requiring that exact shape keeps models whose
    names share a prefix distinct.
    """
    route_name = (route.get("metadata") or {}).get("name") or ""
    base = external_model_route_name(model_name=model_name)
    if route_name == base:
        return True
    prefix = f"{base}-"
    if not route_name.startswith(prefix):
        return False
    return ATTACHMENT_ROUTE_HASH_RE.fullmatch(route_name.removeprefix(prefix)) is not None


def attachment_httproutes(
    admin_client: DynamicClient,
    route_namespace: str,
    gateway_name: str,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
) -> list[dict[str, Any]]:
    """HTTPRoutes in *route_namespace* whose parentRefs select the gateway."""
    routes: list[dict[str, Any]] = []
    for route in list_json(admin_client, "httproute", route_namespace):
        for parent in (route.get("spec") or {}).get("parentRefs") or []:
            # An omitted parentRef namespace means the Gateway in the ROUTE's
            # namespace (Gateway API default), not the target gateway's.
            if parent.get("name") == gateway_name and (parent.get("namespace") or route_namespace) == gateway_namespace:
                routes.append(route)
                break
    return routes


def httproute_parent_conditions(
    route: dict[str, Any],
    gateway_name: str,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
) -> list[dict[str, Any]]:
    """Route status conditions for the parentRef selecting the gateway."""
    route_namespace = route["metadata"]["namespace"]
    for parent in (route.get("status") or {}).get("parents") or []:
        ref = parent.get("parentRef") or {}
        # An omitted parentRef namespace means the Gateway in the ROUTE's
        # namespace (Gateway API default), not the target gateway's.
        if ref.get("name") == gateway_name and (ref.get("namespace") or route_namespace) == gateway_namespace:
            return parent.get("conditions") or []
    return []


def wait_for_attachment_httproute(
    admin_client: DynamicClient,
    route_namespace: str,
    gateway_name: str,
    *,
    model_name: str | None = None,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
    accepted: bool = True,
    reason: str | None = None,
    timeout: int | None = None,
) -> list[dict[str, Any]]:
    """Wait for a gateway-attached HTTPRoute in the route namespace.

    ``accepted=False`` waits for the gateway to refuse the route (e.g.
    NotAllowedByListeners). ``model_name`` scopes the wait to one
    ExternalModel's route: a namespace can host other models attached to the
    same gateway, and an unscoped wait could pass on their routes.
    """
    expected_reason = reason
    scope = f" for model {model_name}" if model_name else ""

    def _check() -> list[dict[str, Any]]:
        routes = attachment_httproutes(
            admin_client=admin_client,
            route_namespace=route_namespace,
            gateway_name=gateway_name,
            gateway_namespace=gateway_namespace,
        )
        if model_name is not None:
            routes = [route for route in routes if route_is_for_model(route, model_name)]
        assert routes, f"no HTTPRoute in {route_namespace} attached to {gateway_namespace}/{gateway_name}{scope}"
        for route in routes:
            conditions = {
                condition.get("type"): condition
                for condition in httproute_parent_conditions(route, gateway_name, gateway_namespace)
            }
            accepted_condition = conditions.get("Accepted")
            if accepted:
                if accepted_condition and accepted_condition.get("status") == "True":
                    return routes
            elif (
                accepted_condition
                and accepted_condition.get("status") == "False"
                and (expected_reason is None or accepted_condition.get("reason") == expected_reason)
            ):
                return routes
        raise AssertionError(
            f"HTTPRoute(s) in {route_namespace} not in expected state "
            f"(accepted={accepted}, reason={expected_reason!r}): "
            f"{[route.get('status') for route in routes]}"
        )

    return wait_until(
        check=_check,
        timeout=timeout or TENANT_EXTERNAL_MODEL_READY_TIMEOUT,
        what=f"HTTPRoute in {route_namespace} attached to {gateway_namespace}/{gateway_name}{scope} "
        f"(accepted={accepted}, reason={expected_reason!r})",
    )


def wait_for_no_attachment_httproute(
    admin_client: DynamicClient,
    route_namespace: str,
    gateway_name: str,
    *,
    model_name: str | None = None,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
    timeout: int | None = None,
) -> None:
    """Wait until no HTTPRoute remains attached to the gateway (model-scoped)."""
    scope = f" for model {model_name}" if model_name else ""

    def _check() -> bool:
        routes = attachment_httproutes(
            admin_client=admin_client,
            route_namespace=route_namespace,
            gateway_name=gateway_name,
            gateway_namespace=gateway_namespace,
        )
        if model_name is not None:
            routes = [route for route in routes if route_is_for_model(route, model_name)]
        assert not routes, f"stale HTTPRoute(s) still attached: {[route['metadata']['name'] for route in routes]}"
        return True

    wait_until(
        check=_check,
        timeout=timeout or TENANT_EXTERNAL_MODEL_READY_TIMEOUT,
        what=f"no HTTPRoute in {route_namespace} attached to {gateway_namespace}/{gateway_name}{scope}",
    )


def _find_credential_copy(
    admin_client: DynamicClient,
    tenant_namespace: str,
    expected: str,
    *,
    exclude_uid: str | None,
) -> dict[str, Any] | None:
    """Find a Secret in *tenant_namespace* carrying *expected*, excluding the source object."""
    for secret in list_json(admin_client, "secret", tenant_namespace):
        metadata = secret.get("metadata") or {}
        if metadata.get("uid") == exclude_uid:
            continue
        carried = (secret.get("data") or {}).get(CREDENTIAL_DATA_KEY) or ""
        if hmac.compare_digest(carried, expected):
            return secret
    return None


def credential_copy_in(
    admin_client: DynamicClient,
    tenant_namespace: str,
    source_secret_name: str,
    source_namespace: str,
) -> dict[str, Any] | None:
    """Find the controller-owned copy of a credential Secret in a tenant namespace.

    AGC delivers credentials from outside the tenant namespace as
    controller-owned copies, identified by content (matching ``api-key``
    data) rather than name.
    """
    source = get_json_or_none(client=admin_client, kind="secret", name=source_secret_name, namespace=source_namespace)
    if not source:
        return None
    expected = (source.get("data") or {}).get(CREDENTIAL_DATA_KEY)
    if not expected:
        return None
    return _find_credential_copy(
        admin_client=admin_client,
        tenant_namespace=tenant_namespace,
        expected=expected,
        exclude_uid=(source.get("metadata") or {}).get("uid"),
    )


def wait_for_credential_copy(
    admin_client: DynamicClient,
    tenant_namespace: str,
    source_secret_name: str,
    source_namespace: str,
    *,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Wait for the controller-owned credential copy in a tenant namespace."""

    def _check() -> dict[str, Any]:
        copy = credential_copy_in(
            admin_client=admin_client,
            tenant_namespace=tenant_namespace,
            source_secret_name=source_secret_name,
            source_namespace=source_namespace,
        )
        assert copy is not None, (
            f"no controller-owned credential copy of {source_namespace}/{source_secret_name} "
            f"found in tenant namespace {tenant_namespace}"
        )
        return copy

    return wait_until(
        check=_check,
        timeout=timeout or TENANT_EXTERNAL_MODEL_READY_TIMEOUT,
        what=f"credential copy of {source_namespace}/{source_secret_name} in {tenant_namespace}",
    )


def wait_for_no_credential_copy(
    admin_client: DynamicClient,
    tenant_namespace: str,
    source_secret_name: str,
    source_namespace: str,
    *,
    timeout: int | None = None,
) -> None:
    """Wait until no controller-owned copy of the credential remains.

    The expected ``api-key`` is captured from the source Secret up front: an
    unreadable source fails the wait rather than passing vacuously.
    """
    source = get_json_or_none(client=admin_client, kind="secret", name=source_secret_name, namespace=source_namespace)
    if not source:
        pytest.fail(
            f"source secret {source_namespace}/{source_secret_name} is unreadable; "
            "the absence of a credential copy cannot be proven without it"
        )
    expected = (source.get("data") or {}).get(CREDENTIAL_DATA_KEY)
    if not expected:
        pytest.fail(
            f"source secret {source_namespace}/{source_secret_name} carries no "
            f"{CREDENTIAL_DATA_KEY} data; the absence of a credential copy cannot be proven without it"
        )
    assert isinstance(expected, str), f"unexpected {CREDENTIAL_DATA_KEY} data type in source secret"
    source_uid = (source.get("metadata") or {}).get("uid")

    def _check() -> bool:
        copy = _find_credential_copy(
            admin_client=admin_client, tenant_namespace=tenant_namespace, expected=expected, exclude_uid=source_uid
        )
        assert copy is None, (
            f"stale credential copy {copy.get('metadata', {}).get('name')} still present in {tenant_namespace}"
        )
        return True

    wait_until(
        check=_check,
        timeout=timeout or TENANT_EXTERNAL_MODEL_READY_TIMEOUT,
        what=f"no credential copy of {source_namespace}/{source_secret_name} in {tenant_namespace}",
    )


def destination_rules_referencing(
    admin_client: DynamicClient,
    model_name: str,
    *namespaces: str,
    also_matching: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """DestinationRules that reference *model_name* (by name or labels).

    AGC creates the provider DestinationRules in the Gateway namespace under
    hashed names that are not part of the shared contract — discovery matches
    on references to the model (test provider names embed the model name, and
    the rules carry them in names and ``external-provider`` labels).
    *also_matching* additionally requires every string to appear in the same
    metadata (e.g. a gateway name, to scope to one attachment's rules).
    """
    matches: list[dict[str, Any]] = []
    needles = [model_name, *also_matching]
    for namespace in namespaces:
        for rule in list_json(admin_client, "destinationrule", namespace):
            metadata = rule.get("metadata") or {}
            owners = [(owner or {}).get("name") or "" for owner in metadata.get("ownerReferences") or []]
            blob = json.dumps({
                "name": metadata.get("name") or "",
                "labels": metadata.get("labels") or {},
                "ownerReferences": owners,
            })
            if all(needle in blob for needle in needles):
                matches.append(rule)
    return matches


def wait_for_no_destination_rules_referencing(
    admin_client: DynamicClient,
    model_name: str,
    *namespaces: str,
    also_matching: tuple[str, ...] = (),
    timeout: int | None = None,
) -> None:
    """Wait until no DestinationRule referencing the model remains."""

    def _check() -> bool:
        rules = destination_rules_referencing(  # noqa: FCN001 - *namespaces must follow positionally
            admin_client,
            model_name,
            *namespaces,
            also_matching=also_matching,
        )
        assert not rules, (
            f"stale DestinationRules referencing {model_name} (also matching {also_matching}): "
            f"{[rule['metadata']['name'] for rule in rules]}"
        )
        return True

    wait_until(
        check=_check,
        timeout=timeout or TENANT_EXTERNAL_MODEL_READY_TIMEOUT,
        what=f"no DestinationRules referencing {model_name} in {namespaces}",
    )


def external_chat_url(
    gateway_url: str,
    model_namespace: str,
    model_client_name: str,
    provider_path: str | None = None,
) -> str:
    """Chat-completions URL via the AGC path convention.

    ``provider_path`` overrides the client-side path suffix: the AGC's
    attachment route forwards whatever follows the model name to the
    provider verbatim, so requesting ``/status/503`` against an echo
    upstream fails that request deterministically (fault injection).
    """
    suffix = provider_path if provider_path is not None else "/v1/chat/completions"
    return f"{gateway_url}/{model_namespace}/{model_client_name}{suffix}"


def _transient_inference_response(response: requests.Response) -> bool:
    """Whether an inference response is retryable gateway/rollout noise."""
    if response.status_code >= 500:
        return True
    if response.status_code == 403 and not (response.text or "").strip():
        return True
    if response.status_code == 429 and "retry" in (response.text or "").lower():
        return True
    return response.status_code == 404


def infer_external_chat(
    request_session_http: requests.Session,
    gateway_url: str,
    model_namespace: str,
    model_client_name: str,
    api_key: str,
    *,
    path_tag: str,
    retries: int = 6,
    delay: int = 5,
    request_timeout: int = 60,
    verify: bool | str = True,
    provider_path: str | None = None,
) -> requests.Response:
    """Run one chat completion through a gateway and return the response.

    Retries transient gateway errors (auth propagation, route rollout) with a
    bounded count; non-transient failures return immediately. TLS is verified
    against the router CA bundle exported by the ``router_ca_bundle_env``
    fixture (``REQUESTS_CA_BUNDLE``); pass ``verify=False`` to opt out on
    clusters whose Route certificates cannot be resolved that way.
    """
    if verify is True:
        # The shared request session disables trust_env so that its own
        # verify=False is honored; requests then ignores REQUESTS_CA_BUNDLE
        # for per-request verify=True calls, so resolve the bundle here.
        bundle_path = os.environ.get("REQUESTS_CA_BUNDLE")
        if bundle_path:
            verify = bundle_path
    if provider_path is None:
        provider_path = echo_provider_path(path_tag) if path_tag else "/v1/chat/completions"
    url = external_chat_url(
        gateway_url=gateway_url,
        model_namespace=model_namespace,
        model_client_name=model_client_name,
        provider_path=provider_path,
    )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model_client_name, "messages": [{"role": "user", "content": path_tag}]}
    last_response: requests.Response | None = None
    try:
        for response in TimeoutSampler(
            wait_timeout=(retries + 1) * delay,
            sleep=delay,
            func=request_session_http.post,
            exceptions_dict={requests.RequestException: []},
            url=url,
            headers=headers,
            json=payload,
            timeout=request_timeout,
            verify=verify,
        ):
            last_response = response
            if not _transient_inference_response(response):
                return response
    except TimeoutExpiredError:
        if last_response is not None:
            return last_response
        raise AssertionError(f"inference to {url} kept failing transiently") from None
    raise AssertionError(f"inference to {url} kept failing transiently")


def infer_for_served_model(
    request_session_http: requests.Session,
    case: dict[str, str],
    served_model: dict[str, Any],
    *,
    retries: int = 6,
    provider_path: str | None = None,
) -> requests.Response:
    """One chat completion for a served tenant model through its own gateway."""
    model = served_model["model"]
    return infer_external_chat(
        request_session_http=request_session_http,
        gateway_url=case["gateway_url"],
        model_namespace=model["namespace"],
        model_client_name=model["name"],
        api_key=served_model["api_key"],
        path_tag=model["path_tag"],
        retries=retries,
        provider_path=provider_path,
    )


def infer_for_shared_model(
    request_session_http: requests.Session,
    case: dict[str, str],
    model: dict[str, Any],
    api_key: str,
    *,
    retries: int = 6,
    provider_path: str | None = None,
) -> requests.Response:
    """One chat completion for a shared model through the given tenant's gateway."""
    return infer_external_chat(
        request_session_http=request_session_http,
        gateway_url=case["gateway_url"],
        model_namespace=model["namespace"],
        model_client_name=model["client_name"],
        api_key=api_key,
        path_tag=model["path_tag"],
        retries=retries,
        provider_path=provider_path,
    )


def exhaust_tenant_token_budget(
    request_session_http: requests.Session,
    case: dict[str, str],
    model: dict[str, Any],
    api_key: str,
    *,
    provider_path: str,
    timeout: int,
    interval: int = 1,
) -> tuple[requests.Response, int]:
    """Drive a tenant's inference until its token budget returns 429.

    Returns the limiting response and the count of successful (200)
    inferences before it. Transient transport errors and 5xx responses are
    retried within *timeout*; a 429 before any success keeps polling (the
    limits may not be live yet); any other status fails the wait with the
    response body redacted.
    """
    successes = 0

    def _attempt() -> requests.Response:
        nonlocal successes
        response = infer_for_shared_model(
            request_session_http=request_session_http,
            case=case,
            model=model,
            api_key=api_key,
            provider_path=provider_path,
            retries=0,
        )
        if response.status_code == 200:
            successes += 1
            raise AssertionError("token budget not exhausted yet")
        if response.status_code == 429:
            assert successes > 0, "tenant hit 429 before any successful inference (limits not live?)"
            return response
        if response.status_code < 500:
            raise AssertionError(
                f"unexpected inference response on tenant: {response.status_code}: {redact_sensitive(response.text)}"
            )
        raise AssertionError(f"transient server error {response.status_code}; retrying")

    limited = wait_until(
        check=_attempt,
        timeout=timeout,
        interval=interval,
        what=f"tenant {case.get('gateway_name', '')} to hit its token budget",
    )
    return limited, successes


def maas_model_ref_phase(
    admin_client: DynamicClient,
    name: str,
    namespace: str,
) -> tuple[str, dict[str, Any]]:
    """Return the (phase, status) of a MaaSModelRef, requiring both keys.

    A missing status or phase is reported as an assertion — a polled,
    not-yet-reconciled state — instead of silently surfacing as ``None``,
    so callers can treat the keys as required once this returns.
    """
    ref = get_json_or_none(client=admin_client, kind="maasmodelref", name=name, namespace=namespace)
    assert ref is not None, f"MaaSModelRef {namespace}/{name} not found"
    status = ref.get("status") or {}
    phase = status.get("phase")
    assert phase is not None, f"MaaSModelRef {namespace}/{name} reports no phase yet: {redact_sensitive(status)}"
    return phase, status


def echo_headers(response: requests.Response) -> dict[str, str]:
    """Headers echoed back by the header-echo provider in a response body."""
    try:
        body = response.json()
    except ValueError as exc:
        raise AssertionError(f"non-JSON response body: {redact_sensitive(response.text)}") from exc
    headers = body.get("headers") or {}
    return {str(key): str(value) for key, value in headers.items()}


def assert_served_by_provider(
    response: requests.Response,
    path_tag: str,
    *,
    credential: str | None = None,
) -> dict[str, str]:
    """Assert the echo provider served the request, optionally with the credential injected.

    The echo endpoint reflects the request URL (proving which provider's
    binding served it, via the unique path tag) and the injected
    Authorization header (proving which credential Secret the gateway used).
    """
    assert response.status_code == 200, (
        f"expected 200 from echo provider, got {response.status_code}: {redact_sensitive(response.text)}"
    )
    try:
        body = response.json()
    except ValueError as exc:
        raise AssertionError(f"non-JSON echo response: {redact_sensitive(response.text)}") from exc
    url = body.get("url") or ""
    assert path_tag in url, f"request was not served by the provider bound to {path_tag!r}; echoed url: {url}"
    headers = echo_headers(response=response)
    if credential is not None:
        authorization = (headers.get("Authorization") or headers.get("authorization") or "").strip()
        # Compare the whole header, not a substring: a longer secret that
        # contains the expected one must not pass. hmac.compare_digest keeps
        # the comparison constant-time.
        injected = any(
            hmac.compare_digest(candidate, authorization) for candidate in (f"Bearer {credential}", credential)
        )
        assert injected, (
            f"expected credential was not injected; Authorization header: {redact_sensitive(authorization)}"
        )
    return headers


@contextmanager
def serve_external_model_in_tenant(
    admin_client: DynamicClient,
    request_session_http: requests.Session,
    ocp_token: str,
    tenant: dict[str, str],
    model_name: str,
    model_namespace: str,
    *,
    policy_name: str | None = None,
    subscription_name: str | None = None,
    token_limit: int = 100,
    window: str = "1m",
    key_name: str | None = None,
    wait_model_ready: bool = True,
    teardown: bool = True,
    timeout: int | None = None,
) -> Generator[dict[str, str], Any, Any]:
    """Apply authorization/subscription semantics for one tenant and mint a key.

    Creates MaaSAuthPolicy + MaaSSubscription in the tenant namespace
    referencing a model that may live in a user namespace, waits for
    reconciliation and mints a tenant API key for the subscription.
    Everything is created under one ExitStack, so a failure partway through
    still cleans up whatever was already created.
    """
    resolved_timeout = timeout or TENANT_EXTERNAL_MODEL_READY_TIMEOUT
    tenant_namespace = tenant["tenant_ns"]
    resolved_policy_name = policy_name or f"{model_name}-access"
    resolved_subscription_name = subscription_name or f"{model_name}-sub"

    with ExitStack() as stack:
        auth_policy = stack.enter_context(
            cm=MaaSAuthPolicy(
                client=admin_client,
                name=resolved_policy_name,
                namespace=tenant_namespace,
                model_refs=[{"name": model_name, "namespace": model_namespace}],
                subjects={"groups": [{"name": "system:authenticated"}]},
                teardown=teardown,
                wait_for_resource=True,
            )
        )
        auth_policy.wait_for_condition(condition="Ready", status="True", timeout=resolved_timeout)
        subscription = stack.enter_context(
            cm=MaaSSubscription(
                client=admin_client,
                name=resolved_subscription_name,
                namespace=tenant_namespace,
                owner={"groups": [{"name": "system:authenticated"}]},
                model_refs=[
                    {
                        "name": model_name,
                        "namespace": model_namespace,
                        "tokenRateLimits": [{"limit": token_limit, "window": window}],
                    }
                ],
                teardown=teardown,
                wait_for_resource=True,
            )
        )
        subscription.wait_for_condition(condition="Ready", status="True", timeout=resolved_timeout)
        if wait_model_ready:
            wait_for_maas_model_ref_ready(
                admin_client=admin_client,
                ref_name=model_name,
                namespace=model_namespace,
                timeout=resolved_timeout,
            )
        api_key, key_id = mint_tenant_api_key(
            request_session_http=request_session_http,
            ocp_token=ocp_token,
            tenant=tenant,
            key_name=key_name or f"e2e-extmod-{generate_random_name()}",
            subscription_name=resolved_subscription_name,
        )
        if teardown:

            def _revoke_key() -> None:
                revoke_tenant_api_key(
                    request_session_http=request_session_http,
                    ocp_token=ocp_token,
                    tenant=tenant,
                    key_id=key_id,
                )

            stack.callback(_revoke_key)  # noqa: FCN001
        yield {
            "policy": resolved_policy_name,
            "subscription": resolved_subscription_name,
            "api_key": api_key,
        }


def wait_for_maas_model_ref_ready(
    admin_client: DynamicClient,
    ref_name: str,
    namespace: str,
    *,
    timeout: int | None = None,
) -> None:
    """Wait until a MaaSModelRef reports ``status.phase: Ready``.

    Source-suite semantics: governance helpers key on the MaaSModelRef name
    (per-tenant tenant-pinned refs for shared models, the model name for
    single-attachment models), not on the ExternalModel name.
    """
    resolved_timeout = timeout or TENANT_EXTERNAL_MODEL_READY_TIMEOUT

    def _check() -> None:
        ref = get_json_or_none(client=admin_client, kind="maasmodelref", name=ref_name, namespace=namespace)
        assert ref is not None, f"MaaSModelRef {namespace}/{ref_name} not found"
        phase = (ref.get("status") or {}).get("phase")
        assert phase == "Ready", f"MaaSModelRef {namespace}/{ref_name} reports phase {phase!r}, want Ready"

    wait_until(check=_check, timeout=resolved_timeout, what=f"MaaSModelRef {namespace}/{ref_name} phase Ready")


def mint_tenant_api_key(
    request_session_http: requests.Session,
    ocp_token: str,
    tenant: dict[str, str],
    key_name: str,
    subscription_name: str,
) -> tuple[str, str]:
    """Mint an API key from a tenant's own maas-api endpoint.

    Returns the key value and the key id (needed to revoke the key during
    teardown so runs do not leave ``e2e-extmod-*`` keys in the tenant store).
    """
    response, body = create_api_key(
        base_url=tenant["maas_api_url"],
        ocp_user_token=ocp_token,
        request_session_http=request_session_http,
        api_key_name=key_name,
        subscription=subscription_name,
    )
    assert response.status_code in (200, 201), (
        f"failed to mint API key on tenant maas-api {tenant['maas_api_url']}: {response.status_code}"
    )
    try:
        api_key = body["key"]
        key_id = body["id"]
    except KeyError as exc:
        raise AssertionError(f"API key response missing field {exc}") from exc
    return api_key, key_id


def revoke_tenant_api_key(
    request_session_http: requests.Session,
    ocp_token: str,
    tenant: dict[str, str],
    key_id: str,
) -> None:
    """Best-effort revocation of a tenant maas-api key during teardown."""
    try:
        response, _body = revoke_api_key(
            request_session_http=request_session_http,
            base_url=tenant["maas_api_url"],
            key_id=key_id,
            ocp_user_token=ocp_token,
        )
        assert response.status_code in (200, 202, 204), (
            f"failed to revoke API key {key_id} on tenant maas-api {tenant['maas_api_url']}: {response.status_code}"
        )
    except Exception as exc:  # noqa: BLE001 - teardown must not mask test failures
        LOGGER.warning(f"best-effort API key revocation failed for key {key_id}", error_type=type(exc).__name__)


@contextmanager
def user_namespace(
    admin_client: DynamicClient,
    name: str,
    *,
    gateway_names: tuple[str, ...] = (),
    teardown: bool = True,
) -> Generator[Namespace | Project, Any, Any]:
    """Create a namespace whose HTTPRoutes the given tenant gateways admit.

    Tenant gateways ship with an allowedRoutes namespace selector keyed on
    the gateway-access label; labeling the user namespace admits routes
    created in it (the user-namespace serving path). No labels keeps that
    gate closed (the unadmitted-namespace scenarios).
    """
    labels: dict[str, str] = {}
    for gateway_name in gateway_names:
        labels.update(gateway_access_namespace_labels(gateway_name=gateway_name))
    with create_ns(
        admin_client=admin_client,
        name=name,
        labels=labels or None,
        teardown=teardown,
    ) as namespace:
        yield namespace


def new_user_namespace_name(prefix: str) -> str:
    """Unique user namespace name for a scenario prefix."""
    return f"{prefix}-{generate_random_name()}"


@contextmanager
def admitted_user_namespace(
    admin_client: DynamicClient,
    prefix: str,
    *gateway_names: str,
    teardown: bool = True,
) -> Generator[Namespace | Project, Any, Any]:
    """Create a user namespace whose routes the given gateways admit."""
    with user_namespace(
        admin_client,
        new_user_namespace_name(prefix),
        gateway_names=gateway_names,
        teardown=teardown,
    ) as namespace:
        yield namespace


@contextmanager
def unadmitted_user_namespace(
    admin_client: DynamicClient,
    prefix: str,
    *,
    teardown: bool = True,
) -> Generator[Namespace | Project, Any, Any]:
    """Create a namespace no tenant gateway's allowedRoutes selector admits."""
    with user_namespace(
        admin_client,
        new_user_namespace_name(prefix),
        teardown=teardown,
    ) as namespace:
        yield namespace


def namespace_name(namespace: Namespace | Project) -> str:
    """Assert and return a namespace wrapper's name (never None at runtime)."""
    assert namespace.name is not None, f"namespace object {namespace!r} carries no name"
    return namespace.name


def provider_external_name_service(provider_name: str) -> str:
    """The AGC's ExternalName Service name for a provider.

    The attachment HTTPRoute's backendRefs reference this Service (created in
    the model namespace, resolving to the provider endpoint), alongside the
    provider-selection sink Service.
    """
    return f"provider-{provider_name}"


def remove_gateway_access_label(
    admin_client: DynamicClient,
    namespace_name: str,
    gateway_name: str,
) -> None:
    """Remove a namespace's gateway-access label (narrow allowedRoutes admission)."""
    namespace = Namespace(client=admin_client, name=namespace_name, ensure_exists=True)
    assert namespace.exists, f"namespace {namespace_name} not found"
    ResourceEditor(
        patches={
            namespace: {
                "metadata": {"labels": {gateway_access_label_key(gateway_name=gateway_name): None}},
            },
        },
    ).update()
    LOGGER.info(f"Removed gateway-access label of {gateway_name!r} from namespace '{namespace_name}'")


def apply_credential_secret(
    admin_client: DynamicClient,
    name: str,
    namespace: str,
    credential: str,
) -> None:
    """Rewrite a provider credential Secret in place (data key ``api-key``).

    Used by the rotation scenario: the source Secret keeps its identity
    (name/namespace) while its key value changes, which is what the
    controller-owned tenant copy must follow.
    """
    secret = Secret(client=admin_client, name=name, namespace=namespace, ensure_exists=True)
    assert secret.exists, f"secret {namespace}/{name} not found"
    ResourceEditor(
        patches={secret: {"stringData": {CREDENTIAL_DATA_KEY: credential}}},
    ).update()
    LOGGER.info(f"Rotated credential secret {namespace}/{name}")


@contextmanager
def serve_in_tenant_and_cleanup(
    admin_client: DynamicClient,
    request_session_http: requests.Session,
    ocp_token: str,
    case: dict[str, str],
    model: dict[str, Any],
    *,
    teardown: bool = True,
) -> Generator[dict[str, str], Any, Any]:
    """Serve one model in a tenant and clean the MaaS stack up afterwards."""
    with serve_external_model_in_tenant(
        admin_client,
        request_session_http,
        ocp_token,
        case,
        model["name"],
        model["namespace"],
        policy_name=f"{model['name']}-access",
        subscription_name=f"{model['name']}-sub",
        teardown=teardown,
    ) as access:
        yield access


def assert_not_served(response: requests.Response) -> None:
    """Assert a response proves the model was not served.

    A successful serve is a 200 from the echo provider. Only the documented
    refusal statuses pass: 401/403/404 (auth denial / no route) and, with a
    gateway-generated (non-provider) body, 502/503 — envoy-side failures on
    a gateway that holds no route for the model, observed under load while
    filters reload, and equally proof that no serve happened. Any other
    status (e.g. 429, 500) proves nothing about routing and fails.
    """
    assert response.status_code in (401, 403, 404, 502, 503), (
        f"expected a documented refusal (401/403/404, or 502/503 from the gateway), got {response.status_code}"
    )
    if response.status_code in (502, 503):
        assert (
            not (response.text or "").strip()
            or "Internal Server Error" in response.text
            or "Service Unavailable" in response.text
        ), "5xx carries a provider-shaped body, expected a gateway-generated refusal/failure"


def provision_and_serve_in_tenant(
    admin_client: DynamicClient,
    request_session_http: requests.Session,
    ocp_token: str,
    stack: ExitStack,
    user_namespace: Namespace | Project,
    case: dict[str, str],
    teardown: bool,
) -> dict[str, Any]:
    """Provision one external model attached to this tenant's gateway and serve it."""
    model = stack.enter_context(
        cm=external_model_stack(
            admin_client=admin_client,
            name=f"e2e-fail-{case['tenant_label_name']}",
            namespace=namespace_name(user_namespace),
            gateway_refs=[gateway_ref(gateway_name=case["gateway_name"])],
            teardown=teardown,
        )
    )
    access = stack.enter_context(
        cm=serve_external_model_in_tenant(
            admin_client=admin_client,
            request_session_http=request_session_http,
            ocp_token=ocp_token,
            tenant=case,
            model_name=model["name"],
            model_namespace=model["namespace"],
            policy_name=f"{model['name']}-access",
            subscription_name=f"{model['name']}-sub",
            teardown=teardown,
        )
    )
    return {"model": model, "api_key": access["api_key"]}


def serve_shared_model_in_tenant(
    admin_client: DynamicClient,
    request_session_http: requests.Session,
    ocp_token: str,
    stack: ExitStack,
    case: dict[str, str],
    model: dict[str, Any],
    teardown: bool,
    **kwargs: Any,
) -> dict[str, str]:
    """Serve one shared model in one tenant and mint its API key."""
    ref_name = model["model_refs"][case["tenant_label_name"]]
    return stack.enter_context(
        cm=serve_external_model_in_tenant(
            admin_client,
            request_session_http,
            ocp_token,
            tenant=case,
            model_name=ref_name,
            model_namespace=model["namespace"],
            policy_name=f"{model['name']}-access",
            subscription_name=f"{model['name']}-sub",
            teardown=teardown,
            **kwargs,
        )
    )


def wait_shared_attachments_ready(
    admin_client: DynamicClient, cases: tuple[dict[str, str], ...], model: dict[str, Any]
) -> None:
    """Wait until both tenants' gateway attachments report ready."""
    for case in cases:
        wait_for_gateway_attachment_ready(
            admin_client=admin_client,
            model_name=model["name"],
            model_namespace=model["namespace"],
            gateway_name=case["gateway_name"],
        )


@contextmanager
def two_gateway_model(
    admin_client: DynamicClient,
    *,
    name: str,
    namespace: str,
    cases: dict[str, dict[str, str]],
    teardown: bool,
    **stack_kwargs: Any,
) -> Generator[dict[str, Any], Any, Any]:
    """A model attached to both tenants' gateways, waited until ready.

    Bundles the two-gateway ``external_model_stack`` shape shared by the
    class fixtures (both gateway refs, per-tenant refs, ``<name>-client``
    alias) with ``wait_shared_attachments_ready``; extra ``stack_kwargs``
    (endpoint, provider_path, credential, provider_config) pass through to
    ``external_model_stack``.
    """
    with external_model_stack(
        admin_client,
        name=name,
        namespace=namespace,
        gateway_refs=[
            gateway_ref(cases["tenant_a"]["gateway_name"]),
            gateway_ref(cases["tenant_b"]["gateway_name"]),
        ],
        model_name=f"{name}-client",
        tenant_refs=[
            cases["tenant_a"]["tenant_label_name"],
            cases["tenant_b"]["tenant_label_name"],
        ],
        teardown=teardown,
        **stack_kwargs,
    ) as model:
        wait_shared_attachments_ready(admin_client=admin_client, cases=tuple(cases.values()), model=model)
        yield model


def require_maas_route(admin_client: DynamicClient, model: dict[str, Any]) -> None:
    """Skip when the MaaS stack cannot resolve this multi-gateway model.

    Aggregate-status AGC builds report no per-gateway ``status.gateways``
    results and no ``status.httpRouteName`` for multi-attachment models, so
    maas-controller cannot select a route: the MaaSModelRef never reports an
    endpoint and the subscription TRLP backend stays not-ready, which makes
    per-tenant auth keys and quotas impossible to wire for the shared model.
    """
    if not external_model_reports_maas_route(admin_client, model["name"], model["namespace"]):
        pytest.skip(
            "AGC reports aggregate model status only (no per-gateway status.gateways "
            "results); the MaaS stack cannot resolve a multi-gateway model's route, so "
            "per-tenant auth/quota serving for a shared model cannot be exercised"
        )


def require_client_name_alias(
    admin_client: DynamicClient,
    model: dict[str, Any],
    case: dict[str, str],
) -> None:
    """Skip until the published model identity is the client-facing name.

    The per-tenant stacks go green with per-gateway status alone, but a
    path-routed request carries ``{namespace}/{client-facing name}`` while
    the modelRefs are named after their tenants: requests only authorize
    once the controller publishes ``spec.modelName`` as
    ``status.resolvedModelAlias`` and maas-api matches pairs against it.
    Both halves ship together, so one alias check gates both.
    """
    ref_name = model["model_refs"][case["tenant_label_name"]]
    alias = maas_model_ref_alias(admin_client=admin_client, namespace=model["namespace"], name=ref_name)
    if alias != model["client_name"]:
        pytest.skip(
            "client-facing model names are not resolvable through the gateway "
            f"yet: MaaSModelRef {model['namespace']}/{ref_name} "
            f"status.resolvedModelAlias={alias!r} != {model['client_name']!r}"
        )


def usage_provider(
    admin_client: DynamicClient,
    stack: ExitStack,
    namespace: str,
    client_model_name: str,
    teardown: bool,
) -> dict[str, Any]:
    """Deploy the usage-returning provider the quota model binds to.

    A dedicated in-cluster ``llm-d-inference-sim`` (``--mode random``): its
    responses carry ``usage.total_tokens``, which token budgets are
    accounted from. The gateway envoy verifies provider certificates, so
    the simulator cannot serve a self-signed cert: its Service is annotated
    for an OpenShift service serving certificate (signed by the cluster
    service CA) and the provider points envoy at the CA bundle mounted in
    every pod through ``spec.config["tls.caCertificates"]``. The simulator
    validates the request body's ``model`` field, so it is configured with
    the client-facing model name the requests will carry.

    The simulator is deployed in the model's namespace and is dedicated to
    that model, so no rate limits apply on the provider hop: a 429 seen
    through a tenant gateway can only be that tenant's own budget.

    An echo endpoint cannot stand in even with a synthetic usage field: the
    telemetry pipeline drops reports whose body does not name the model and
    cannot read gzip-compressed responses (which httpbingo serves) — the
    simulator's plain OpenAI-shaped bodies satisfy all of it.
    """
    suffix = uuid.uuid4().hex[:6]
    sim = f"e2e-usage-sim-{suffix}"
    cert_secret = f"{sim}-cert"
    dns_name = f"{sim}.{namespace}.svc.cluster.local"
    stack.enter_context(
        cm=Service(
            client=admin_client,
            name=sim,
            namespace=namespace,
            annotations={SERVING_CERT_ANNOTATION: cert_secret},
            selector={"app": sim},
            ports=[{"port": 443, "targetPort": 8000, "name": "https"}],
            teardown=teardown,
            wait_for_resource=True,
        )
    )

    def _cert_injected() -> None:
        secret = get_json_or_none(admin_client, "secret", cert_secret, namespace) or {}
        assert (secret.get("data") or {}).get("tls.crt"), (
            f"secret {namespace}/{cert_secret} has no tls.crt yet (service-ca injection pending)"
        )

    wait_until(check=_cert_injected, timeout=120, what=f"service-ca cert secret {namespace}/{cert_secret}")

    stack.enter_context(
        cm=Deployment(
            client=admin_client,
            name=sim,
            namespace=namespace,
            replicas=1,
            selector={"matchLabels": {"app": sim}},
            template={
                "metadata": {"labels": {"app": sim}},
                "spec": {
                    "containers": [
                        {
                            "name": "sim",
                            "image": LLM_D_INFERENCE_SIM_IMAGE,
                            "command": ["/app/llm-d-inference-sim"],
                            "args": [
                                "--port",
                                "8000",
                                "--model",
                                client_model_name,
                                "--mode",
                                "random",
                                "--no-mm-encoder-only",
                                "--ssl-certfile",
                                "/etc/sim/tls/tls.crt",
                                "--ssl-keyfile",
                                "/etc/sim/tls/tls.key",
                            ],
                            "ports": [{"containerPort": 8000, "name": "https", "protocol": "TCP"}],
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": "https",
                                    "scheme": "HTTPS",
                                }
                            },
                            "volumeMounts": [{"name": "tls", "mountPath": "/etc/sim/tls", "readOnly": True}],
                        }
                    ],
                    "volumes": [{"name": "tls", "secret": {"secretName": cert_secret}}],
                },
            },
            teardown=teardown,
        )
    )

    def _ready() -> None:
        deployment = get_json_or_none(admin_client, "deployment", sim, namespace) or {}
        status = deployment.get("status") or {}
        available = status.get("availableReplicas") or 0
        assert available >= 1, f"usage simulator {namespace}/{sim} has {available} available replicas, want >= 1"

    wait_until(check=_ready, timeout=240, what=f"usage simulator deployment {namespace}/{sim}")
    return {
        "endpoint": dns_name,
        "provider_path": "/v1/chat/completions",
        "credential": f"sk-usage-{suffix}",
        "provider_config": {"tls.caCertificates": SA_SERVICE_CA},
    }


def assert_returns_usage(response: requests.Response) -> None:
    """Assert an inference response carries OpenAI-style usage tokens.

    Token quotas are accounted from ``usage.total_tokens`` in the response,
    so a provider that returns no usage (e.g. the header-echo endpoint)
    can never exhaust a budget — quota assertions are only meaningful
    against responses that pass this check.
    """
    assert response.status_code == 200, (
        f"expected 200 from external model inference, got {response.status_code}: {redact_sensitive(response.text)}"
    )
    try:
        usage = (response.json() or {}).get("usage") or {}
    except ValueError:
        usage = {}
    total_tokens = usage.get("total_tokens")
    assert isinstance(total_tokens, int) and total_tokens > 0, (
        f"response carries no usage.total_tokens for token accounting: {redact_sensitive(response.text)}"
    )
