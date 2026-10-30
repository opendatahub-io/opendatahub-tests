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

from tests.ai_gateway.models_as_a_service.multitenancy.utils import (
    gateway_access_label_key as _gateway_access_label_key,
)
from tests.ai_gateway.models_as_a_service.utils import create_api_key, revoke_api_key
from utilities.constants import MAAS_GATEWAY_NAMESPACE
from utilities.general import generate_random_name
from utilities.infra import create_ns
from utilities.resources.destination_rule import DestinationRule
from utilities.resources.external_model import ExternalModel
from utilities.resources.external_provider import ExternalProvider
from utilities.resources.http_route import HTTPRoute

LOGGER = structlog.get_logger(name=__name__)

# ─── Constants ──────────────────────────────────────────────────────────────

TENANT_EXTERNAL_MODEL_READY_TIMEOUT = int(os.environ.get("E2E_TENANT_EXTERNAL_MODEL_READY_TIMEOUT", "300"))
TENANT_EXTERNAL_MODEL_POLL_INTERVAL = 5

# Header-echo endpoint: responses under /anything/* echo the request headers
# and URL back in the JSON body, proving which provider served the request
# (unique path tag) and which credential was injected (unique marker value).
EXTERNAL_ENDPOINT = os.environ.get("E2E_EXTERNAL_ENDPOINT", "httpbingo.org")

# Secret data key holding the provider credential (contract in the
# ExternalProvider/ExternalModel CRD descriptions).
CREDENTIAL_DATA_KEY = "api-key"

EXTERNAL_TARGET_MODEL = "gpt-3.5-turbo"
EXTERNAL_API_FORMAT = "openai-chat"

AGC_DEPLOYMENT_NAME = os.environ.get("AGC_DEPLOYMENT_NAME", "ai-gateway-controller")
# The AGC Deployment's namespace varies by deployment type: the MaaS e2e
# companion deploy puts it in "opendatahub", the RHOAI operator in
# "redhat-ods-applications", ODH in "odh-applications". An explicit
# AGC_DEPLOYMENT_NAMESPACE wins; otherwise find_agc_deployment searches the
# candidates below. (Only the Gateway and the praxis payload-processing
# Deployment live in MAAS_GATEWAY_NAMESPACE.)
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

# Condition types accepted as "this attachment is serving-ready". AGC's exact
# condition naming is not frozen in the shared CRD; match across plausible
# spellings until it is.
READY_CONDITION_TYPES = ("Ready", "Accepted", "Attached")

# AGC renames per-tenant praxis resources to "<resource>-<tenant-name>". The
# longest base name is the per-tenant EnvoyFilter
# ("payload-processing-external-model-filters-"), which leaves 20 characters
# for the tenant name before the Kubernetes 63-character name limit makes AGC
# refuse to render the tenant's whole praxis stack.
AGC_LONGEST_RESOURCE_BASE = "payload-processing-external-model-filters-"
MAX_AITENANT_NAME_LENGTH = 63 - len(AGC_LONGEST_RESOURCE_BASE)

# The AGC's attachment routes append shortHash — exactly 12 lowercase hex
# characters (sha256 over ns/name/gateway-namespace/gateway-name) — after the
# model name; see attachmentRouteName in the AGC controller.
ATTACHMENT_ROUTE_HASH_RE = re.compile(r"[0-9a-f]{12}")

# Per-tenant payload-processing deployment prefix (AGC renames per-tenant
# praxis resources to "<resource>-<tenant-name>").
PER_TENANT_PROCESSING_DEPLOYMENT_PREFIX = "payload-processing-"
PRAXIS_IMAGE_MARKER = "praxis"


# ─── Generic resource JSON access ───────────────────────────────────────────

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
    except NotFoundError:
        return None
    return _instance_to_dict(instances[0].instance) if instances else None


def list_json(
    client: DynamicClient,
    kind: str,
    namespace: str | None = None,
) -> list[dict[str, Any]]:
    """Return raw JSON dicts for every resource of *kind* in *namespace*."""
    wrapper = _JSON_WRAPPERS[kind]
    instances = list(wrapper.get(client=client, namespace=namespace))
    return [_instance_to_dict(instance.instance) for instance in instances]


def wait_until(
    check: Callable[[], Any],
    timeout: int,
    what: str,
    interval: int = TENANT_EXTERNAL_MODEL_POLL_INTERVAL,
) -> Any:
    """Poll *check* until it stops raising, within a bounded time budget."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return check()
        except AssertionError as exc:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out after {timeout}s waiting for: {what}") from exc
            time.sleep(interval)


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


def redact_sensitive(value: Any, *, max_length: int = 300) -> str:
    """Redact secret-looking fields and truncate for log/assert messages."""
    text = repr(value)
    for needle in ("sk-", "api-key", "password", "token"):
        text = re.sub(rf"{needle}[^'\"\\s,}}]*", f"{needle}***", text, flags=re.IGNORECASE)
    return text if len(text) <= max_length else f"{text[:max_length]}..."


# ─── Capability detection ───────────────────────────────────────────────────


def _crd_schema_properties(crd: dict[str, Any], section: str | None) -> dict[str, Any]:
    """Merge the schema properties of *section* ("spec"/"status") across versions."""
    merged: dict[str, Any] = {}
    for version in (crd.get("spec") or {}).get("versions") or []:
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
        # Fall back to a name match in case the companion deploy renames it.
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


def require_crossns_provider_support(admin_client: DynamicClient) -> None:
    """Gate P1 cross-namespace provider scenarios (RHOAIENG-90804)."""
    if not env_bool("TENANT_EXTERNAL_MODEL_CROSSNS_E2E"):
        pytest.skip(
            "cross-namespace provider E2E is gated until AGC ships "
            "cross-namespace provider support; set TENANT_EXTERNAL_MODEL_CROSSNS_E2E=true to enable"
        )
    if get_json_or_none(client=admin_client, kind="crd", name=REFERENCE_GRANT_CRD_NAME) is None:
        pytest.skip(f"ReferenceGrant CRD {REFERENCE_GRANT_CRD_NAME} is not installed")


def require_legacy_ipp_coexistence_support() -> None:
    """Gate the legacy IPP coexistence scenario (RHOAIENG-99761 image pin)."""
    if not env_bool("TENANT_EXTERNAL_MODEL_LEGACY_IPP_E2E"):
        pytest.skip(
            "legacy IPP coexistence E2E requires the MaaS IPP image pin containing "
            "the legacy-controller exclusion guard; set "
            "TENANT_EXTERNAL_MODEL_LEGACY_IPP_E2E=true to enable"
        )


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
            "TENANT_EXTERNAL_MODEL_CROSSNS_E2E": env_bool("TENANT_EXTERNAL_MODEL_CROSSNS_E2E"),
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


# ─── Resource builders ──────────────────────────────────────────────────────


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


def wait_for_model_generation_observed(
    admin_client: DynamicClient,
    name: str,
    namespace: str,
    timeout: int = TENANT_EXTERNAL_MODEL_READY_TIMEOUT,
) -> None:
    """Wait until the controller has observed the model's current generation.

    Readiness waits can return on the first poll while the status still
    reflects the previous (already Ready) generation; a recreation a few
    seconds later would slip through an immediate comparison.
    """

    def _check() -> bool:
        assert _generation_observed(get_json_or_none(admin_client, "externalmodel", name, namespace)), (
            f"observedGeneration of {namespace}/{name} not caught up"
        )
        return True

    wait_until(
        check=_check,
        timeout=timeout,
        what=f"observedGeneration of {namespace}/{name}",
    )


def _generation_observed(external_model: dict[str, Any] | None) -> bool:
    if not external_model:
        return False
    metadata = external_model.get("metadata") or {}
    status = external_model.get("status") or {}
    observed = status.get("observedGeneration") or 0
    return observed >= (metadata.get("generation") or 1)


# ─── ExternalModel per-gateway status ───────────────────────────────────────


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
    entry = next(
        (item for item in entries if item.get("name") == gateway_name and item.get("namespace") == gateway_namespace),
        None,
    )
    if entry is not None and entry.get("conditions"):
        return entry["conditions"], f"attachment to {gateway_namespace}/{gateway_name}"
    status = external_model.get("status") or {}
    return (status.get("conditions") or []), "aggregate model status"


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
        for condition_type in READY_CONDITION_TYPES:
            condition = by_type.get(condition_type)
            if condition and condition.get("status") == "True":
                return condition
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


def wait_for_gateway_attachment_not_ready(
    admin_client: DynamicClient,
    model_name: str,
    model_namespace: str,
    gateway_name: str,
    *,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
    message_pattern: str | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Wait until an attachment reports a failure (not serving).

    ``message_pattern`` (case-insensitive regex) additionally requires the
    failure to mention e.g. the missing ReferenceGrant authorization or Secret.
    """
    pattern = re.compile(message_pattern, re.IGNORECASE) if message_pattern else None

    def _match_condition(conditions: list[dict[str, Any]], where: str) -> dict[str, Any]:
        failures = [condition for condition in conditions if condition.get("status") == "False"]
        assert failures, f"no False condition on {where}; conditions: {conditions}"
        if pattern is None:
            return failures[0]
        matched = [
            condition
            for condition in failures
            if pattern.search(condition.get("message") or "") or pattern.search(condition.get("reason") or "")
        ]
        assert matched, f"no failure on {where} matching {message_pattern!r}; conditions: {conditions}"
        return matched[0]

    def _check() -> dict[str, Any]:
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
        return _match_condition(conditions=conditions, where=where)

    return wait_until(
        check=_check,
        timeout=timeout or TENANT_EXTERNAL_MODEL_READY_TIMEOUT,
        what=f"{model_namespace}/{model_name} attachment to {gateway_namespace}/{gateway_name} "
        f"report failure ({message_pattern or 'any reason'})",
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


# ─── HTTPRoute discovery per attachment ─────────────────────────────────────


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
            if (
                parent.get("name") == gateway_name
                and (parent.get("namespace") or gateway_namespace) == gateway_namespace
            ):
                routes.append(route)
                break
    return routes


def httproute_parent_conditions(
    route: dict[str, Any],
    gateway_name: str,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
) -> list[dict[str, Any]]:
    """Route status conditions for the parentRef selecting the gateway."""
    for parent in (route.get("status") or {}).get("parents") or []:
        ref = parent.get("parentRef") or {}
        if ref.get("name") == gateway_name and (ref.get("namespace") or gateway_namespace) == gateway_namespace:
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


# ─── Tenant-published resources (credential copies, DestinationRules) ───────


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
        if (secret.get("data") or {}).get(CREDENTIAL_DATA_KEY) == expected:
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
        rules = destination_rules_referencing(
            admin_client, model_name, *namespaces, also_matching=also_matching
        )  # noqa: FCN001 - namespaces feed *args
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


# ─── Inference through a specific gateway ───────────────────────────────────


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


def infer_external_chat(
    gateway_url: str,
    model_namespace: str,
    model_client_name: str,
    api_key: str,
    *,
    path_tag: str,
    retries: int = 6,
    delay: int = 5,
    request_timeout: int = 60,
    verify: bool = False,
    provider_path: str | None = None,
) -> requests.Response:
    """Run one chat completion through a gateway and return the response.

    Retries transient gateway errors (auth propagation, route rollout) with a
    bounded count; non-transient failures return immediately. TLS verification
    is off by default (the repo convention for gateway-route requests, cf.
    request_session_http): tenant-gateway Route certs are signed by the
    cluster CA, which is not in the runner's trust store.
    """
    url = external_chat_url(
        gateway_url=gateway_url,
        model_namespace=model_namespace,
        model_client_name=model_client_name,
        provider_path=provider_path,
    )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model_client_name, "messages": [{"role": "user", "content": path_tag}]}
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                url=url,
                headers=headers,
                json=payload,
                timeout=request_timeout,
                verify=verify,
            )
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(delay)
                continue
            raise
        if response.status_code >= 500 and attempt < retries:
            time.sleep(delay)
            continue
        if response.status_code == 429 and "retry" in (response.text or "").lower() and attempt < retries:
            time.sleep(delay)
            continue
        return response
    raise AssertionError(f"inference to {url} kept failing transiently: {last_exc}")


def echo_headers(response: requests.Response) -> dict[str, str]:
    """Headers echoed back by the header-echo provider in a response body."""
    try:
        body = response.json()
    except ValueError as exc:
        raise AssertionError(f"non-JSON response body: {redact_sensitive(response.text)}") from exc
    headers = ((body.get("json") or {}).get("headers") or {}) if "json" in body else (body.get("headers") or {})
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


# ─── MaaS governance + API keys per tenant ──────────────────────────────────


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
                priority=0,
                teardown=teardown,
                wait_for_resource=True,
            )
        )
        subscription.wait_for_condition(condition="Ready", status="True", timeout=resolved_timeout)
        if wait_model_ready:
            # Source-suite semantics (require_model_ready): the ref's phase
            # reads Ready only once governance is paired, so this wait must
            # follow the policy/subscription creation. model_name is the
            # MaaSModelRef name (per-tenant tenant-pinned ref for shared
            # models, the model name for single-attachment models).
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
            # Registered last so it unwinds first: revoke while the
            # subscription still exists (a revoked key is inert once the
            # subscription is gone, but revoking avoids tenant-store residue).
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
    api_key = body.get("key")
    assert api_key, f"API key missing in response: {redact_sensitive(body)}"
    key_id = body.get("id")
    assert key_id, f"API key id missing in response: {redact_sensitive(body)}"
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


# ─── User namespaces ────────────────────────────────────────────────────────


def gateway_access_label_key(gateway_name: str) -> str:
    """Namespace label key that admits routes for one tenant gateway."""
    return _gateway_access_label_key(gateway_name=gateway_name)


def gateway_access_namespace_labels(*gateway_names: str) -> dict[str, str]:
    """Namespace labels admitting routes for the given tenant gateways."""
    return {gateway_access_label_key(gateway_name): "true" for gateway_name in gateway_names}


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
    with create_ns(
        admin_client=admin_client,
        name=name,
        labels=gateway_access_namespace_labels(*gateway_names) or None,
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
