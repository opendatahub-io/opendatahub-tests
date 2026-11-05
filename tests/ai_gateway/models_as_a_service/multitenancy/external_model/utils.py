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

import json
import os
import re
import time
import uuid
from collections.abc import Callable, Generator
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
                if not aitenant.delete(wait=True, timeout=AITENANT_DELETE_TIMEOUT):
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
                tenant_namespace = Namespace(
                    client=admin_client,
                    name=expected_tenant_namespace_name(aitenant_name=aitenant_name),
                    ensure_exists=True,
                )
                if tenant_namespace.exists:
                    tenant_namespace.delete(wait=False)
                    LOGGER.info(f"Deleted tenant namespace '{tenant_namespace.name}'")


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
    """Skip unless the installed cluster can serve multi-tenant external models.

    Requires the additive schema (gatewayRefs + status.gateways) in the
    *installed* ExternalModel CRD and a deployed ai-gateway-controller. Keeps
    default CI green while AGC images catch up; the checks auto-activate the
    suite once the schema and controller ship.
    """
    crd = get_json_or_none(client=admin_client, kind="crd", name=EXTERNAL_MODEL_CRD_NAME)
    if crd is None:
        pytest.skip(f"ExternalModel CRD {EXTERNAL_MODEL_CRD_NAME} is not installed")
    capabilities = installed_crd_capabilities(admin_client=admin_client, crd=crd)
    if not (capabilities["gatewayRefs"] and capabilities["statusGateways"]):
        pytest.skip(
            "installed ExternalModel CRD lacks the gatewayRefs additive schema "
            f"(capabilities: {capabilities}); AGC gatewayRefs support not distributed"
        )
    if find_agc_deployment(admin_client) is None:
        pytest.skip(
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
        authorization = headers.get("Authorization") or headers.get("authorization") or ""
        assert credential in authorization, (
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
        f"failed to mint API key on tenant maas-api {tenant['maas_api_url']}: "
        f"{response.status_code} {redact_sensitive(response.text)}"
    )
    try:
        api_key = body["key"]
        key_id = body["id"]
    except KeyError as exc:
        raise AssertionError(f"API key response missing field {exc}: {redact_sensitive(body)}") from exc
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
        LOGGER.warning(f"best-effort API key revocation failed for key {key_id}: {exc}")


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


def namespace_name(namespace: Namespace | Project) -> str:
    """Assert and return a namespace wrapper's name (never None at runtime)."""
    assert namespace.name is not None, f"namespace object {namespace!r} carries no name"
    return namespace.name


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
