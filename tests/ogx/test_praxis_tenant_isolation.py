import tempfile
from collections.abc import Generator
from typing import Self

import httpx
import pytest
import structlog
from ocp_resources.route import Route
from ogx_client import APIError, OgxClient

from tests.ogx.constants import OGX_CLIENT_VERIFY_SSL

LOGGER = structlog.get_logger(name=__name__)


def _build_tenant_client(
    base_url: str,
    user_id: str,
    tenant_id: str,
    token: str | None = None,
) -> Generator[OgxClient]:
    """Helper generator building an OgxClient scoped to a specific user and tenant identity."""
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

        client_tenant_a = next(_build_tenant_client(base_url, user_id="user-a", tenant_id="tenant-alpha"))
        client_tenant_b = next(_build_tenant_client(base_url, user_id="user-b", tenant_id="tenant-beta"))

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

                # 3. User B in Tenant B attempts to retrieve User A's file
                with pytest.raises((APIError, httpx.HTTPStatusError)) as exc_info:
                    client_tenant_b.files.retrieve(file_id=file_id)

                status_code = getattr(exc_info.value, "status_code", None)
                if status_code is None and hasattr(exc_info.value, "response"):
                    status_code = exc_info.value.response.status_code

                LOGGER.info(f"Cross-tenant file access rejected with status: {status_code}")
                assert status_code in (403, 404), f"Expected 403 or 404 on cross-tenant read, got {status_code}"

                # 4. User B attempts to delete User A's file
                with pytest.raises((APIError, httpx.HTTPStatusError)) as exc_info_del:
                    client_tenant_b.files.delete(file_id=file_id)

                del_status = getattr(exc_info_del.value, "status_code", None)
                assert del_status in (403, 404), f"Expected 403 or 404 on cross-tenant delete, got {del_status}"

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

        client_tenant_a = next(_build_tenant_client(base_url, user_id="user-a", tenant_id="tenant-alpha"))
        client_tenant_b = next(_build_tenant_client(base_url, user_id="user-b", tenant_id="tenant-beta"))

        # 1. User A creates vector store
        vs_a = client_tenant_a.vector_stores.create(name="tenant-a-private-vs")
        vs_id = vs_a.id
        LOGGER.info(f"User A created vector store {vs_id} in tenant-alpha")

        try:
            # 2. User A sees vector store in list
            vs_list_a = client_tenant_a.vector_stores.list()
            assert any(item.id == vs_id for item in vs_list_a.data), "User A should see created vector store"

            # 3. User B lists vector stores - User A's vs_id MUST be absent
            vs_list_b = client_tenant_b.vector_stores.list()
            assert not any(item.id == vs_id for item in vs_list_b.data), (
                f"Vector store {vs_id} belonging to tenant-alpha leaked into tenant-beta list"
            )

            # 4. User B attempts direct retrieve of User A's vector store
            with pytest.raises((APIError, httpx.HTTPStatusError)) as exc_info:
                client_tenant_b.vector_stores.retrieve(vector_store_id=vs_id)

            status_code = getattr(exc_info.value, "status_code", None)
            assert status_code in (403, 404), f"Expected 403 or 404 on direct retrieve, got {status_code}"

        finally:
            client_tenant_a.vector_stores.delete(vector_store_id=vs_id)

    @pytest.mark.tier1
    def test_e2e_file_to_vector_store_attachment(
        self: Self,
        ogx_test_route: Route,
    ) -> None:
        """Verify that attaching files across tenants or modifying unauthorized vector stores is rejected."""
        base_url = f"https://{ogx_test_route.host}"

        client_tenant_a = next(_build_tenant_client(base_url, user_id="user-a", tenant_id="tenant-alpha"))
        client_tenant_b = next(_build_tenant_client(base_url, user_id="user-b", tenant_id="tenant-beta"))

        # User A creates vector store and uploads file
        vs_a = client_tenant_a.vector_stores.create(name="tenant-a-attach-test")
        vs_id = vs_a.id

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=True) as tmp_file:
            tmp_file.write("Attachment file content.")
            tmp_file.flush()
            file_a = client_tenant_a.files.create(file=tmp_file.name, purpose="assistants")
            file_id_a = file_a.id

        try:
            # User B attempts to attach User A's file or attach to User A's vector store
            with pytest.raises((APIError, httpx.HTTPStatusError)) as exc_info:
                client_tenant_b.vector_stores.files.create(
                    vector_store_id=vs_id,
                    file_id=file_id_a,
                )

            status_code = getattr(exc_info.value, "status_code", None)
            assert status_code in (403, 404), f"Expected 403 or 404 on unauthorized attachment, got {status_code}"

        finally:
            client_tenant_a.vector_stores.delete(vector_store_id=vs_id)
            client_tenant_a.files.delete(file_id=file_id_a)
