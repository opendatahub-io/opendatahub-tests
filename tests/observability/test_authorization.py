from unittest.mock import Mock, patch

import pytest

from tests.observability.authorization import authenticate_persona_tokens, check_persona_access
from tests.observability.personas import Persona
from utilities.resources.subject_access_review import SubjectAccessReview

pytestmark = pytest.mark.tier1


def test_subject_access_review_preserves_groups_and_decision_fields() -> None:
    """Given a persona and SAR response, retain identity, scope, denial, and reason evidence."""
    status = Mock(allowed=False, denied=True, reason="forbidden", evaluationError=None)
    sar_resource = Mock()
    sar_resource.instance.status = status
    client = Mock()
    persona = Persona(
        name="namespace-admin",
        principal="alice",
        groups=("system:authenticated",),
        namespaces=("ns-a",),
    )

    with patch("tests.observability.authorization.SubjectAccessReview", return_value=sar_resource) as resource_factory:
        result = check_persona_access(
            admin_client=client,
            persona=persona,
            verb="get",
            resource="pods",
            namespace="ns-b",
            subresource="status",
        )

    assert result.allowed is False
    assert result.denied is True
    assert result.reason == "forbidden"
    assert result.resource == "pods"
    assert result.subresource == "status"
    assert result.to_dict()["subresource"] == "status"
    kind_dict = resource_factory.call_args.kwargs["kind_dict"]
    assert kind_dict["spec"]["groups"] == ["system:authenticated"]
    assert kind_dict["spec"]["resourceAttributes"]["resource"] == "pods"
    assert kind_dict["spec"]["resourceAttributes"]["subresource"] == "status"
    sar_resource.create.assert_called_once_with()


def test_subject_access_review_has_createable_resource_metadata() -> None:
    """Given a SAR specification, build the metadata and API version required by the wrapper create path."""
    review = SubjectAccessReview(
        client=Mock(),
        kind_dict={
            "apiVersion": SubjectAccessReview.api_version,
            "kind": "SubjectAccessReview",
            "metadata": {"name": "observability-review"},
            "spec": {"resourceAttributes": {"resource": "pods"}},
        },
    )

    review.to_dict()

    assert review.res["apiVersion"] == "authorization.k8s.io/v1"
    assert review.res["kind"] == "SubjectAccessReview"
    assert review.res["metadata"]["name"] == "observability-review"
    assert review.res["spec"]["resourceAttributes"]["resource"] == "pods"


def test_token_review_returns_authenticated_principal_and_groups() -> None:
    """Given a valid token review response, retain only the authenticated identity fields used by SAR checks."""
    client = Mock()
    token_review_api = Mock()
    review = Mock()
    review.status.authenticated = True
    review.status.user.username = "alice"
    review.status.user.groups = ["system:authenticated"]
    token_review_api.create.return_value = review
    client.resources.get.return_value = token_review_api

    identities = authenticate_persona_tokens(admin_client=client, tokens={"namespace-admin": "secret-token"})

    assert identities["namespace-admin"].principal == "alice"
    assert identities["namespace-admin"].groups == ("system:authenticated",)
    client.resources.get.assert_called_once()
    assert client.resources.get.call_args.kwargs["api_version"] == "authentication.k8s.io/v1"
    assert token_review_api.create.call_args.kwargs["body"]["spec"]["token"] == "secret-token"
