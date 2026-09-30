"""Persona definitions and independence validation."""

from collections.abc import Mapping
from dataclasses import dataclass

REQUIRED_PERSONAS = (
    "cluster-admin",
    "namespace-admin",
    "namespace-contributor",
    "regular-user",
)


class PersonaValidationError(ValueError):
    """Raised when persona evidence cannot prove independent authorization."""


@dataclass(frozen=True)
class Persona:
    """Sanitized identity and scope metadata for a test persona."""

    name: str
    principal: str
    groups: tuple[str, ...]
    namespaces: tuple[str, ...]


@dataclass(frozen=True)
class TokenIdentity:
    """Identity returned by Kubernetes after authenticating a persona token."""

    principal: str
    groups: tuple[str, ...]


def validate_personas(personas: list[Persona]) -> tuple[Persona, ...]:
    """Validate that all required personas have distinct authenticated principals."""
    by_name = {persona.name: persona for persona in personas}
    missing = tuple(name for name in REQUIRED_PERSONAS if name not in by_name)
    if missing:
        raise PersonaValidationError(f"missing required personas: {', '.join(sorted(missing))}")

    principals: dict[str, str] = {}
    for persona in personas:
        if not persona.principal:
            raise PersonaValidationError(f"{persona.name} has no authenticated principal")
        previous_name = principals.get(persona.principal)
        if previous_name and previous_name != persona.name:
            raise PersonaValidationError(
                f"{persona.name} and {previous_name} do not have independent principal identities"
            )
        principals[persona.principal] = persona.name

    return tuple(by_name[name] for name in REQUIRED_PERSONAS)


def validate_token_identities(personas: tuple[Persona, ...], identities: Mapping[str, TokenIdentity]) -> None:
    """Require each injected token to authenticate as its configured principal and groups."""
    for persona in personas:
        identity = identities.get(persona.name)
        if identity is None:
            raise PersonaValidationError(f"missing authenticated token identity for {persona.name}")
        if identity.principal != persona.principal:
            raise PersonaValidationError(
                f"{persona.name} authenticated as {identity.principal!r}, expected {persona.principal!r}"
            )
        if set(identity.groups) != set(persona.groups):
            raise PersonaValidationError(
                f"{persona.name} authenticated with groups {sorted(identity.groups)!r}, "
                f"expected {sorted(persona.groups)!r}"
            )
