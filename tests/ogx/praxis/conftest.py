import pytest
from kubernetes.dynamic import DynamicClient

from utilities.infra import get_openshift_token


@pytest.fixture
def tenant_authorization_header(admin_client: DynamicClient) -> dict[str, str]:
    """Authorization header carrying the OpenShift token of the authenticated tenant."""
    return {
        "Authorization": f"Bearer {get_openshift_token(client=admin_client)}",
        "Content-Type": "application/json",
    }
