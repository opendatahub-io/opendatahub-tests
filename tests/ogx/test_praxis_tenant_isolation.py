import tempfile
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from typing import Self

import httpx
import pytest
import structlog
from ocp_resources.route import Route
from ogx_client import APIError, OgxClient

from tests.ogx.constants import OGX_CLIENT_VERIFY_SSL

LOGGER = structlog.get_logger(name=__name__)


@contextmanager
def _build_tenant_client(
    base_url: str,
    user_id: str,
    tenant_id: str,
    token: str | None = None,
) -> Generator[OgxClient]:
    """Helper context manager building an OgxClient scoped to a specific user and tenant identity."""
    headers = {
        "x-user-id": user_id,
        "x-tenant-id": tenant_id,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    http_client = httpx.Client(
        verify=OGX_CLIENT_VERIFY_SSL,
        timeout=60,
        headers=headers,
    )
    try:
        client = OgxClient(
            base_url=base_url,
            max_retries=2,
            http_client=http_client,
            timeout=60,
        )
        yield client
    finally:
        http_client.close()


def _get_status_code(exc: Exception) -> int | None:
    """Extract HTTP status code from APIError or httpx.HTTPStatusError."""
    status_code = getattr(exc, "status_code", None)
    if status_code is None and hasattr(exc, "response") and exc.response is not None:
        status_code = getattr(exc.response, "status_code", None)
    return status_code


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server",
    [
        pytest.param(
            {"name": "test-ogx-praxis-isolation", "randomize_name": True},
            {
                "vector_io_provider": "pgvector",
                "files_provider": "local",
            },
            id="praxis-tenant-isolation-pgvector-local",
        ),
    ],
    indirect=True,
)
@pytest.mark.ogx
class TestOgxPraxisTenantIsolation:
    """E2E System integration test suite for Praxis-to-OGX multi-tenant and cross-user isolation.

    Validates that identity context (x-user-id, x-tenant-id) forwarded through Praxis proxy
    strictly isolates Files and Vector Store resources across users and tenants.
    """

    @pytest.mark.tier1
    def test_e2e_file_upload_and_isolation(
        self: Self,
        ogx_test_route: Route,
    ) -> None:
        """Verify cross-user and cross-tenant isolation for Files API."""
        base_url = f"https://{ogx_test_route.host}"

        with ExitStack() as stack:
            client_tenant_a = stack.enter_context(
                cm=_build_tenant_client(base_url=base_url, user_id="user-a", tenant_id="tenant-alpha")
            )
            unauthorized_clients = [
                (
                    "cross-tenant/cross-user",
                    stack.enter_context(
                        cm=_build_tenant_client(base_url=base_url, user_id="user-b", tenant_id="tenant-beta")
                    ),
                ),
                (
                    "same-tenant/cross-user",
                    stack.enter_context(
                        cm=_build_tenant_client(base_url=base_url, user_id="user-b", tenant_id="tenant-alpha")
                    ),
                ),
                (
                    "cross-tenant/same-user",
                    stack.enter_context(
                        cm=_build_tenant_client(base_url=base_url, user_id="user-a", tenant_id="tenant-beta")
                    ),
                ),
            ]

            # 1. User A uploads file
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=True) as tmp_file:
                tmp_file.write("Sensitive tenant-alpha document content.")
                tmp_file.flush()

                uploaded_file = client_tenant_a.files.create(
                    file=tmp_file.name,
                    purpose="assistants",
                )
                file_id = uploaded_file.id
                LOGGER.info(f"User A created file {file_id} in tenant-alpha")

            try:
                # 2. User A can retrieve own file
                fetched_a = client_tenant_a.files.retrieve(file_id=file_id)
                assert fetched_a.id == file_id

                # 3. Verify unauthorized access attempts are rejected
                for boundary, unauth_client in unauthorized_clients:
                    with pytest.raises((APIError, httpx.HTTPStatusError)) as exc_info:
                        unauth_client.files.retrieve(file_id=file_id)

                    status_code = _get_status_code(exc=exc_info.value)
                    LOGGER.info(f"File retrieve ({boundary}) rejected with status: {status_code}")
                    assert status_code in (403, 404), f"Expected 403 or 404 on {boundary} file read, got {status_code}"

                    with pytest.raises((APIError, httpx.HTTPStatusError)) as exc_info_del:
                        unauth_client.files.delete(file_id=file_id)

                    del_status = _get_status_code(exc=exc_info_del.value)
                    LOGGER.info(f"File delete ({boundary}) rejected with status: {del_status}")
                    assert del_status in (403, 404), f"Expected 403 or 404 on {boundary} file delete, got {del_status}"

            finally:
                # Cleanup file via User A
                client_tenant_a.files.delete(file_id=file_id)

    @pytest.mark.tier1
    def test_e2e_vector_store_isolation(
        self: Self,
        ogx_test_route: Route,
    ) -> None:
        """Verify cross-user and cross-tenant isolation for Vector Stores API."""
        base_url = f"https://{ogx_test_route.host}"

        with ExitStack() as stack:
            client_tenant_a = stack.enter_context(
                cm=_build_tenant_client(base_url=base_url, user_id="user-a", tenant_id="tenant-alpha")
            )
            unauthorized_clients = [
                (
                    "cross-tenant/cross-user",
                    stack.enter_context(
                        cm=_build_tenant_client(base_url=base_url, user_id="user-b", tenant_id="tenant-beta")
                    ),
                ),
                (
                    "same-tenant/cross-user",
                    stack.enter_context(
                        cm=_build_tenant_client(base_url=base_url, user_id="user-b", tenant_id="tenant-alpha")
                    ),
                ),
                (
                    "cross-tenant/same-user",
                    stack.enter_context(
                        cm=_build_tenant_client(base_url=base_url, user_id="user-a", tenant_id="tenant-beta")
                    ),
                ),
            ]

            # 1. User A creates vector store
            vs_a = client_tenant_a.vector_stores.create(name="tenant-a-private-vs")
            vs_id = vs_a.id
            LOGGER.info(f"User A created vector store {vs_id} in tenant-alpha")

            try:
                # 2. User A sees vector store in list
                vs_list_a = client_tenant_a.vector_stores.list()
                assert any(item.id == vs_id for item in vs_list_a.data), "User A should see created vector store"

                # 3. Unauthorized users list vector stores - User A's vs_id MUST be absent
                for boundary, unauth_client in unauthorized_clients:
                    vs_list = unauth_client.vector_stores.list()
                    assert not any(item.id == vs_id for item in vs_list.data), (
                        f"Vector store {vs_id} belonging to tenant-alpha leaked into {boundary} list"
                    )

                    # 4. Unauthorized direct retrieve attempt
                    with pytest.raises((APIError, httpx.HTTPStatusError)) as exc_info:
                        unauth_client.vector_stores.retrieve(vector_store_id=vs_id)

                    status_code = _get_status_code(exc=exc_info.value)
                    assert status_code in (403, 404), (
                        f"Expected 403 or 404 on direct retrieve for {boundary}, got {status_code}"
                    )

            finally:
                client_tenant_a.vector_stores.delete(vector_store_id=vs_id)

    @pytest.mark.tier1
    def test_e2e_file_to_vector_store_attachment(
        self: Self,
        ogx_test_route: Route,
    ) -> None:
        """Verify that attaching files across tenants or modifying unauthorized vector stores is rejected."""
        base_url = f"https://{ogx_test_route.host}"

        with ExitStack() as stack:
            client_tenant_a = stack.enter_context(
                cm=_build_tenant_client(base_url=base_url, user_id="user-a", tenant_id="tenant-alpha")
            )
            client_tenant_b = stack.enter_context(
                cm=_build_tenant_client(base_url=base_url, user_id="user-b", tenant_id="tenant-beta")
            )
            client_same_tenant_diff_user = stack.enter_context(
                cm=_build_tenant_client(base_url=base_url, user_id="user-b", tenant_id="tenant-alpha")
            )

            # User A creates vector store
            vs_a = client_tenant_a.vector_stores.create(name="tenant-a-attach-test")
            vs_id_a = vs_a.id
            file_id_a: str | None = None
            vs_id_b: str | None = None

            try:
                # User B creates vector store in Tenant B
                vs_b = client_tenant_b.vector_stores.create(name="tenant-b-attach-test")
                vs_id_b = vs_b.id

                # User A uploads file in Tenant Alpha
                with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=True) as tmp_file:
                    tmp_file.write("Attachment file content.")
                    tmp_file.flush()
                    file_a = client_tenant_a.files.create(file=tmp_file.name, purpose="assistants")
                    file_id_a = file_a.id

                # Case 1: User B tries to attach User A's file to Tenant B's vector store (foreign file)
                with pytest.raises((APIError, httpx.HTTPStatusError)) as exc_info1:
                    client_tenant_b.vector_stores.files.create(
                        vector_store_id=vs_id_b,
                        file_id=file_id_a,
                    )
                status1 = _get_status_code(exc=exc_info1.value)
                assert status1 in (403, 404), f"Expected 403 or 404 on attaching foreign file, got {status1}"

                # Case 2: User B tries to attach User A's file to User A's vector store (foreign vector store)
                with pytest.raises((APIError, httpx.HTTPStatusError)) as exc_info2:
                    client_tenant_b.vector_stores.files.create(
                        vector_store_id=vs_id_a,
                        file_id=file_id_a,
                    )
                status2 = _get_status_code(exc=exc_info2.value)
                assert status2 in (
                    403,
                    404,
                ), f"Expected 403 or 404 on attaching to foreign vector store, got {status2}"

                # Case 3: Same tenant, different user tries to attach User A's file to User A's vector store
                with pytest.raises((APIError, httpx.HTTPStatusError)) as exc_info3:
                    client_same_tenant_diff_user.vector_stores.files.create(
                        vector_store_id=vs_id_a,
                        file_id=file_id_a,
                    )
                status3 = _get_status_code(exc=exc_info3.value)
                assert status3 in (
                    403,
                    404,
                ), f"Expected 403 or 404 for same-tenant cross-user attachment, got {status3}"

            finally:
                if vs_id_a:
                    client_tenant_a.vector_stores.delete(vector_store_id=vs_id_a)
                if vs_id_b:
                    client_tenant_b.vector_stores.delete(vector_store_id=vs_id_b)
                if file_id_a:
                    client_tenant_a.files.delete(file_id=file_id_a)
