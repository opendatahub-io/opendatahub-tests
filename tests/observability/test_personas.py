import pytest

from tests.observability.personas import (
    REQUIRED_PERSONAS,
    Persona,
    PersonaValidationError,
    TokenIdentity,
    select_persona_tokens,
    validate_personas,
    validate_token_identities,
)

pytestmark = pytest.mark.tier1


def test_persona_validation_requires_independent_principals() -> None:
    """Given four persona definitions, reject a restricted persona sharing the administrator principal."""
    personas = [Persona(name=name, principal="cluster-admin", groups=(), namespaces=()) for name in REQUIRED_PERSONAS]

    with pytest.raises(PersonaValidationError, match="independent principal"):
        validate_personas(personas=personas)


def test_persona_validation_requires_all_release_personas() -> None:
    """Given an incomplete persona set, report the missing required identities."""
    with pytest.raises(PersonaValidationError, match="namespace-contributor"):
        validate_personas(
            personas=[
                Persona(name="cluster-admin", principal="admin", groups=(), namespaces=()),
                Persona(name="namespace-admin", principal="ns-admin", groups=(), namespaces=("ns-a",)),
                Persona(name="regular-user", principal="user", groups=(), namespaces=("ns-a",)),
            ]
        )


def test_persona_validation_rejects_token_principal_mismatch() -> None:
    """Given a token review identity, reject a token that authenticates as another configured persona."""
    personas = [
        Persona(name=name, principal=f"{name}-principal", groups=(), namespaces=()) for name in REQUIRED_PERSONAS
    ]
    identities = {
        persona.name: TokenIdentity(principal=persona.principal, groups=persona.groups) for persona in personas
    }
    identities["regular-user"] = TokenIdentity(principal="cluster-admin-principal", groups=())

    with pytest.raises(PersonaValidationError, match="authenticated as"):
        validate_token_identities(personas=personas, identities=identities)


def test_persona_validation_rejects_token_group_mismatch() -> None:
    """Given a token review identity, reject groups that differ from the configured authorization evidence."""
    personas = [
        Persona(name=name, principal=f"{name}-principal", groups=(), namespaces=()) for name in REQUIRED_PERSONAS
    ]
    identities = {
        persona.name: TokenIdentity(principal=persona.principal, groups=persona.groups) for persona in personas
    }
    identities["namespace-admin"] = TokenIdentity(principal="namespace-admin-principal", groups=("system:masters",))

    with pytest.raises(PersonaValidationError, match="groups"):
        validate_token_identities(personas=personas, identities=identities)


def test_persona_token_selection_ignores_unconfigured_entries() -> None:
    """Given token input with an unnamed extra entry, select only configured persona tokens."""
    personas = (
        Persona(name="cluster-admin", principal="admin", groups=(), namespaces=("*",)),
        Persona(name="regular-user", principal="user", groups=(), namespaces=("ns-a",)),
    )

    tokens = select_persona_tokens(
        raw_tokens={
            "cluster-admin": "admin-token",
            "regular-user": "user-token",
            "unexpected": "admin-token",
        },
        personas=personas,
    )

    assert tokens == {"cluster-admin": "admin-token", "regular-user": "user-token"}
