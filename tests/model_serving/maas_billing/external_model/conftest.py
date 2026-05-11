from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.maas_auth_policy import MaaSAuthPolicy
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.maas_subscription import MaaSSubscription
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret

from tests.model_serving.maas_billing.external_model.utils import (
    EXTERNAL_AUTH_POLICY_NAME,
    EXTERNAL_ENDPOINT,
    EXTERNAL_MODEL_NAME,
    EXTERNAL_SECRET_NAME,
    EXTERNAL_SUBSCRIPTION_NAME,
)
from tests.model_serving.maas_billing.utils import create_api_key, revoke_api_key
from utilities.general import generate_random_name
from utilities.resources.external_model import ExternalModel

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="class")
def external_model_credential_secret(
    admin_client: DynamicClient,
    maas_unprivileged_model_namespace: Namespace,
) -> Generator[Secret, Any, Any]:
    """Opaque secret holding a dummy API key required by ExternalModel.credentialRef."""
    with Secret(
        client=admin_client,
        name=EXTERNAL_SECRET_NAME,
        namespace=maas_unprivileged_model_namespace.name,
        type="Opaque",
        string_data={"api-key": "e2e-test-key"},
        teardown=True,
        wait_for_resource=True,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def external_model_cr(
    admin_client: DynamicClient,
    maas_unprivileged_model_namespace: Namespace,
    external_model_credential_secret: Secret,
) -> Generator[ExternalModel, Any, Any]:
    """ExternalModel CR pointing to the external endpoint (httpbin.org by default)."""
    with ExternalModel(
        client=admin_client,
        name=EXTERNAL_MODEL_NAME,
        namespace=maas_unprivileged_model_namespace.name,
        provider="openai",
        target_model="gpt-3.5-turbo",
        endpoint=EXTERNAL_ENDPOINT,
        credential_ref={
            "name": external_model_credential_secret.name,
        },
        teardown=True,
        wait_for_resource=True,
    ) as external_model:
        yield external_model


@pytest.fixture(scope="class")
def external_model_ref(
    admin_client: DynamicClient,
    maas_unprivileged_model_namespace: Namespace,
    external_model_cr: ExternalModel,
) -> Generator[MaaSModelRef, Any, Any]:
    """MaaSModelRef linking to the ExternalModel CR."""
    with MaaSModelRef(
        client=admin_client,
        name=EXTERNAL_MODEL_NAME,
        namespace=maas_unprivileged_model_namespace.name,
        model_ref={
            "name": external_model_cr.name,
            "kind": "ExternalModel",
        },
        teardown=True,
        wait_for_resource=True,
    ) as model_ref:
        model_ref.wait_for_condition(condition="Ready", status="True", timeout=600)
        yield model_ref


@pytest.fixture(scope="class")
def external_model_auth_policy(
    admin_client: DynamicClient,
    maas_subscription_namespace: Namespace,
    external_model_ref: MaaSModelRef,
) -> Generator[MaaSAuthPolicy, Any, Any]:
    """MaaSAuthPolicy granting system:authenticated access to the external model."""
    with MaaSAuthPolicy(
        client=admin_client,
        name=EXTERNAL_AUTH_POLICY_NAME,
        namespace=maas_subscription_namespace.name,
        model_refs=[
            {
                "name": external_model_ref.name,
                "namespace": external_model_ref.namespace,
            }
        ],
        subjects={"groups": [{"name": "system:authenticated"}]},
        teardown=True,
        wait_for_resource=True,
    ) as auth_policy:
        auth_policy.wait_for_condition(condition="Ready", status="True", timeout=300)
        yield auth_policy


@pytest.fixture(scope="class")
def external_model_subscription(
    admin_client: DynamicClient,
    maas_subscription_namespace: Namespace,
    external_model_ref: MaaSModelRef,
) -> Generator[MaaSSubscription, Any, Any]:
    """MaaSSubscription for the external model with generous token limits."""
    with MaaSSubscription(
        client=admin_client,
        name=EXTERNAL_SUBSCRIPTION_NAME,
        namespace=maas_subscription_namespace.name,
        owner={"groups": [{"name": "system:authenticated"}]},
        model_refs=[
            {
                "name": external_model_ref.name,
                "namespace": external_model_ref.namespace,
                "tokenRateLimits": [{"limit": 10000, "window": "1h"}],
            }
        ],
        priority=0,
        teardown=True,
        wait_for_resource=True,
    ) as subscription:
        subscription.wait_for_condition(condition="Ready", status="True", timeout=300)
        yield subscription


@pytest.fixture(scope="class")
def external_model_api_key(
    request_session_http: requests.Session,
    base_url: str,
    ocp_token_for_actor: str,
    external_model_subscription: MaaSSubscription,
) -> Generator[str, Any, Any]:
    """API key bound to the external model subscription. Revoked on teardown."""
    key_name = f"e2e-external-model-{generate_random_name()}"
    _, body = create_api_key(
        base_url=base_url,
        ocp_user_token=ocp_token_for_actor,
        request_session_http=request_session_http,
        api_key_name=key_name,
        subscription=external_model_subscription.name,
        expires_in="1h",
    )
    LOGGER.info(f"Created external model API key: id={body['id']}")
    yield body["key"]
    revoke_response, _ = revoke_api_key(
        request_session_http=request_session_http,
        base_url=base_url,
        key_id=body["id"],
        ocp_user_token=ocp_token_for_actor,
    )
    assert revoke_response.status_code in (200, 404), (
        f"Unexpected revocation status for key id={body['id']}: {revoke_response.status_code}"
    )


@pytest.fixture(scope="class")
def external_model_inference_url(
    maas_scheme: str,
    maas_host: str,
    maas_unprivileged_model_namespace: Namespace,
) -> str:
    """Chat completions URL for the external model on the MaaS gateway."""
    return (
        f"{maas_scheme}://{maas_host}"
        f"/{maas_unprivileged_model_namespace.name}"
        f"/{EXTERNAL_MODEL_NAME}/v1/chat/completions"
    )
