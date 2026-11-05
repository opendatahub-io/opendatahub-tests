"""PostgreSQL storage backend, exercised end to end through the service API.

Every test here runs against a PostgreSQL deployment reached with
`sslmode=verify-full`, so the TLS path in `storage/sql/engine.py` is covered by
the suite as a whole rather than by a single test.
"""

import base64
from http import HTTPStatus
from typing import Any

import pytest
import structlog
from ocp_resources.deployment import Deployment
from ocp_resources.trustyai_service import TrustyAIService

from tests.ai_safety.trustyai_service.storage_backends.constants import (
    DATABASE_STORAGE_FORMAT,
    ENV_DATABASE_KIND,
    ENV_STORAGE_FORMAT,
    HEALTH_STATUS_OK,
    POSTGRES_DATABASE_KIND,
    STORAGE_CHECK_NAME,
)
from tests.ai_safety.trustyai_service.storage_backends.utils import (
    StorageBackendClient,
    build_upload_payload,
    run_python_in_service_pod,
    wait_for_model_observations,
)
from utilities.constants import TRUSTYAI_SERVICE_NAME

LOGGER = structlog.get_logger(name=__name__)

POSTGRES_SESSION_PROBE: str = """
import json
from sqlalchemy import text
from trustyai_service.service.data.storage import get_storage_interface

storage = get_storage_interface()
with storage._engine.connect() as conn:
    statement = text("select version(), current_database(), ssl from pg_stat_ssl where pid = pg_backend_pid()")
    row = conn.execute(statement).one()
print(json.dumps({"version": row[0], "database": row[1], "ssl": row[2]}))
"""


def _container_env(deployment: Deployment) -> dict[str, Any]:
    """Return the literal (non-secret) env of the service container."""
    container = deployment.instance.spec.template.spec.containers[0]
    return {var["name"]: var.get("value") for var in container.env}


@pytest.mark.ai_safety
@pytest.mark.tier1
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-trustyai-postgres-storage"})],
    indirect=True,
)
class TestPostgresStorageBackend:
    """Data written through the API round-trips through the PostgreSQL backend."""

    def test_service_reports_postgres_backend(
        self,
        postgres_backed_service: Deployment,
        postgres_storage_client: StorageBackendClient,
        postgres_db_credentials_secret,
        admin_client,
        model_namespace,
    ) -> None:
        """The CR keeps DATABASE and its Secret selects the PostgreSQL implementation."""
        env = _container_env(deployment=postgres_backed_service)
        assert env.get(ENV_STORAGE_FORMAT) == DATABASE_STORAGE_FORMAT, (
            f"Expected {ENV_STORAGE_FORMAT}={DATABASE_STORAGE_FORMAT}, got {env.get(ENV_STORAGE_FORMAT)!r}"
        )
        service = TrustyAIService(
            client=admin_client,
            namespace=model_namespace.name,
            name=TRUSTYAI_SERVICE_NAME,
            ensure_exists=True,
        )
        assert service.instance.spec.storage.format == DATABASE_STORAGE_FORMAT
        assert service.instance.spec.storage.databaseConfigurations == postgres_db_credentials_secret.name

        secret_kind = base64.b64decode(postgres_db_credentials_secret.instance.data["databaseKind"]).decode()
        assert secret_kind == POSTGRES_DATABASE_KIND
        kind_entry = next(
            item
            for item in postgres_backed_service.instance.spec.template.spec.containers[0].env
            if item["name"] == ENV_DATABASE_KIND
        )
        assert kind_entry["valueFrom"]["secretKeyRef"]["name"] == postgres_db_credentials_secret.name
        assert kind_entry["valueFrom"]["secretKeyRef"]["key"] == "databaseKind"

        settings = run_python_in_service_pod(
            client=admin_client,
            namespace=model_namespace.name,
            deployment=postgres_backed_service,
            code=POSTGRES_SESSION_PROBE,
        )
        assert "PostgreSQL" in settings["version"], f"Expected a PostgreSQL server, got {settings}"
        assert settings["database"] == "trustyai_db", f"Wrong database selected: {settings}"
        assert settings["ssl"] is True, f"PostgreSQL connection did not use TLS: {settings}"

        response = postgres_storage_client.readiness()
        assert response.status_code == HTTPStatus.OK, f"Readiness probe failed: {response.text}"

        checks = {check["name"]: check["status"] for check in response.json()["checks"]}
        assert checks.get(STORAGE_CHECK_NAME) == HEALTH_STATUS_OK, (
            f"Storage readiness is not ok against PostgreSQL: {response.json()}"
        )

    def test_upload_creates_datasets(self, postgres_storage_client: StorageBackendClient) -> None:
        """Uploaded rows are stored and reported back by the service."""
        model_name = "postgres-upload-model"
        n_rows = 5

        response = postgres_storage_client.upload(payload=build_upload_payload(model_name=model_name, n_rows=n_rows))
        assert response.status_code == HTTPStatus.OK, f"Upload failed: {response.text}"

        model_info = wait_for_model_observations(
            storage_client=postgres_storage_client, model_name=model_name, expected=n_rows
        )
        assert model_info["data"]["observations"] == n_rows

    def test_metadata_reports_uploaded_shape(self, postgres_storage_client: StorageBackendClient) -> None:
        """Column metadata written to the dynamic per-dataset tables is read back intact."""
        model_name = "postgres-shape-model"
        n_rows = 4

        assert (
            postgres_storage_client.upload(
                payload=build_upload_payload(model_name=model_name, n_rows=n_rows)
            ).status_code
            == HTTPStatus.OK
        )

        model_info = wait_for_model_observations(
            storage_client=postgres_storage_client, model_name=model_name, expected=n_rows
        )

        input_items = model_info["data"]["inputSchema"]["items"]
        output_items = model_info["data"]["outputSchema"]["items"]
        assert len(input_items) == 2, f"Expected 2 input columns, got {input_items}"
        assert len(output_items) == 1, f"Expected 1 output column, got {output_items}"

    def test_repeated_uploads_append_rows(self, postgres_storage_client: StorageBackendClient) -> None:
        """A second upload appends to the existing dataset instead of replacing it."""
        model_name = "postgres-append-model"
        n_rows = 3

        for offset in (0, n_rows):
            response = postgres_storage_client.upload(
                payload=build_upload_payload(model_name=model_name, n_rows=n_rows, offset=offset)
            )
            assert response.status_code == HTTPStatus.OK, f"Upload failed: {response.text}"

        model_info = wait_for_model_observations(
            storage_client=postgres_storage_client, model_name=model_name, expected=n_rows * 2
        )
        assert model_info["data"]["observations"] == n_rows * 2

    def test_inference_ids_listed_and_paged(self, postgres_storage_client: StorageBackendClient) -> None:
        """Inference ids are stored per row and paged with limit/offset."""
        model_name = "postgres-inference-ids-model"
        n_rows = 6

        assert (
            postgres_storage_client.upload(
                payload=build_upload_payload(model_name=model_name, n_rows=n_rows)
            ).status_code
            == HTTPStatus.OK
        )
        wait_for_model_observations(storage_client=postgres_storage_client, model_name=model_name, expected=n_rows)

        response = postgres_storage_client.inference_ids(model_name=model_name)
        assert response.status_code == HTTPStatus.OK, f"Inference id lookup failed: {response.text}"
        body = response.json()
        assert body["total"] == n_rows, f"Expected {n_rows} inference ids, got {body['total']}"

        paged = postgres_storage_client.inference_ids(model_name=model_name, params={"limit": 2, "offset": 2})
        assert paged.status_code == HTTPStatus.OK, f"Paged inference id lookup failed: {paged.text}"
        paged_body = paged.json()
        assert len(paged_body["ids"]) == 2, f"Expected 2 ids in the page, got {len(paged_body['ids'])}"
        assert paged_body["ids"] != body["ids"][:2], "Offset was ignored: the page repeats the first ids"

    def test_name_mapping_round_trip(self, postgres_storage_client: StorageBackendClient) -> None:
        """Applying and clearing a name mapping updates the stored dataset metadata."""
        model_name = "postgres-name-mapping-model"
        n_rows = 3

        assert (
            postgres_storage_client.upload(
                payload=build_upload_payload(model_name=model_name, n_rows=n_rows)
            ).status_code
            == HTTPStatus.OK
        )
        model_info = wait_for_model_observations(
            storage_client=postgres_storage_client, model_name=model_name, expected=n_rows
        )

        original_input = next(iter(model_info["data"]["inputSchema"]["items"]))
        mapping = {"modelId": model_name, "inputMapping": {original_input: "mapped_input"}, "outputMapping": {}}

        apply_response = postgres_storage_client.apply_name_mapping(payload=mapping)
        assert apply_response.status_code == HTTPStatus.OK, f"Name mapping failed: {apply_response.text}"

        mapped = postgres_storage_client.info().json()[model_name]
        assert mapped["data"]["inputSchema"]["nameMapping"].get(original_input) == "mapped_input", (
            f"Name mapping not reflected in metadata: {mapped['data']['inputSchema']}"
        )

        clear_response = postgres_storage_client.clear_name_mapping(model_name=model_name)
        assert clear_response.status_code == HTTPStatus.OK, f"Clearing name mapping failed: {clear_response.text}"

        cleared = postgres_storage_client.info().json()[model_name]
        assert not cleared["data"]["inputSchema"]["nameMapping"], (
            f"Name mapping survived the clear: {cleared['data']['inputSchema']['nameMapping']}"
        )

    def test_tags_reported_for_uploaded_data(self, postgres_storage_client: StorageBackendClient) -> None:
        """Rows uploaded with a tag are counted under that tag."""
        model_name = "postgres-tags-model"
        n_rows = 4
        tag = "TRAINING"

        assert (
            postgres_storage_client.upload(
                payload=build_upload_payload(model_name=model_name, n_rows=n_rows, data_tag=tag)
            ).status_code
            == HTTPStatus.OK
        )
        wait_for_model_observations(storage_client=postgres_storage_client, model_name=model_name, expected=n_rows)

        response = postgres_storage_client.tags(model_name=model_name)
        assert response.status_code == HTTPStatus.OK, f"Tag lookup failed: {response.text}"
        assert response.json().get(tag) == n_rows, f"Expected {n_rows} rows tagged {tag}, got {response.json()}"
