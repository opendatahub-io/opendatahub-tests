"""Persona authorization checks using Kubernetes SubjectAccessReview."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tests.observability.personas import Persona, TokenIdentity
from utilities.resources.subject_access_review import SubjectAccessReview

if TYPE_CHECKING:
    from kubernetes.dynamic import DynamicClient


@dataclass(frozen=True)
class SubjectAccessReviewResult:
    """Complete sanitized SAR outcome."""

    persona: str
    principal: str
    namespace: str | None
    verb: str
    api_group: str
    resource: str
    subresource: str | None
    allowed: bool
    denied: bool
    reason: str | None
    evaluation_error: str | None

    def to_dict(self) -> dict[str, object]:
        """Return SAR evidence without authentication material."""
        return {
            "persona": self.persona,
            "principal": self.principal,
            "namespace": self.namespace,
            "verb": self.verb,
            "api_group": self.api_group,
            "resource": self.resource,
            "subresource": self.subresource,
            "allowed": self.allowed,
            "denied": self.denied,
            "reason": self.reason,
            "evaluation_error": self.evaluation_error,
        }


def check_persona_access(
    admin_client: DynamicClient,
    persona: Persona,
    verb: str,
    resource: str,
    namespace: str | None,
    api_group: str = "",
    subresource: str | None = None,
    api_version: str = "v1",
) -> SubjectAccessReviewResult:
    """Run one SAR for a persona and preserve allowed, denied, and error fields."""
    review_name = "observability-subject-access-review"
    resource_attributes = {
        "namespace": namespace,
        "verb": verb,
        "group": api_group,
        "resource": resource,
        "version": api_version,
    }
    if subresource:
        resource_attributes["subresource"] = subresource
    review = SubjectAccessReview(
        client=admin_client,
        kind_dict={
            "apiVersion": SubjectAccessReview.api_version,
            "kind": "SubjectAccessReview",
            "metadata": {"name": review_name},
            "spec": {
                "user": persona.principal,
                "groups": list(persona.groups),
                "resourceAttributes": resource_attributes,
            },
        },
    )
    review.create()
    status = getattr(review.instance, "status", None)
    allowed = bool(getattr(status, "allowed", False))
    denied = bool(getattr(status, "denied", False))
    return SubjectAccessReviewResult(
        persona=persona.name,
        principal=persona.principal,
        namespace=namespace,
        verb=verb,
        api_group=api_group,
        resource=resource,
        subresource=subresource,
        allowed=allowed,
        denied=denied,
        reason=_optional_string(getattr(status, "reason", None)),
        evaluation_error=_optional_string(getattr(status, "evaluationError", None)),
    )


def authenticate_persona_tokens(admin_client: DynamicClient, tokens: Mapping[str, str]) -> dict[str, TokenIdentity]:
    """Authenticate injected bearer tokens through Kubernetes TokenReview."""
    token_review_api = admin_client.resources.get(
        api_version="authentication.k8s.io/v1",
        kind="TokenReview",
    )
    identities: dict[str, TokenIdentity] = {}
    for persona_name, token in tokens.items():
        review = token_review_api.create(
            body={
                "apiVersion": "authentication.k8s.io/v1",
                "kind": "TokenReview",
                "spec": {"token": token},
            }
        )
        status = _object_value(source=review, key="status")
        if _object_value(source=status, key="authenticated") is not True:
            raise ValueError(f"token for {persona_name} was not authenticated")
        user = _object_value(source=status, key="user")
        principal = _object_value(source=user, key="username")
        groups = _object_value(source=user, key="groups")
        if not isinstance(principal, str) or not principal:
            raise ValueError(f"token for {persona_name} returned no authenticated principal")
        if not isinstance(groups, (list, tuple)) or not all(isinstance(group, str) for group in groups):
            raise ValueError(f"token for {persona_name} returned invalid authenticated groups")
        identities[persona_name] = TokenIdentity(principal=principal, groups=tuple(groups))
    return identities


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


def _object_value(source: object, key: str) -> object:
    """Read a field from either a Kubernetes resource object or a test mapping."""
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)
