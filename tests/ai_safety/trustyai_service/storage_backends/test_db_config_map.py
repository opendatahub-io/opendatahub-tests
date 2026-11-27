"""SQL tuning ConfigMaps on TrustyAI services created through their CR."""

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager
from http import HTTPStatus
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.trustyai_service import TrustyAIService

from tests.ai_safety.trustyai_service.constants import TAI_METRICS_CONFIG
from tests.ai_safety.trustyai_service.storage_backends.constants import (
    ENV_APPLICATION_NAME,
    ENV_MAX_CONNECTIONS_BUDGET,
    ENV_MAX_OVERFLOW,
    ENV_POOL_RECYCLE,
    ENV_POOL_SIZE,
    ENV_POOL_TIMEOUT,
    ENV_SQLITE_BUSY_TIMEOUT_MS,
    ENV_SQLITE_CACHE_SIZE_KB,
    ENV_SQLITE_JOURNAL_MODE,
    ENV_SQLITE_SYNCHRONOUS,
    ENV_STATEMENT_TIMEOUT_MS,
    ENV_SYNCHRONOUS_COMMIT,
)
from tests.ai_safety.trustyai_service.storage_backends.utils import (
    StorageBackendClient,
    build_upload_payload,
    create_database_backed_trustyai_service,
    deployment_pod_selector,
    restart_trustyai_service_pod,
    run_python_in_service_pod,
    update_database_tuning_config_map,
    wait_for_deployment_unready,
    wait_for_model_observations,
    wait_for_pod_log_message,
    wait_for_trustyai_condition,
)
from utilities.constants import TRUSTYAI_SERVICE_NAME

LOGGER = structlog.get_logger(name=__name__)

CONFIGURED_DEPLOYMENT: str = TRUSTYAI_SERVICE_NAME
SQLITE_CONFIGURED_DEPLOYMENT: str = TRUSTYAI_SERVICE_NAME
DB_CONFIG_MAP_NAME: str = "trustyai-db-config"

# Values chosen to differ from every default so a silently ignored key is visible.
APPLICATION_NAME: str = "tas-db-config-test"
STATEMENT_TIMEOUT_MS: str = "45000"
POOL_SIZE: str = "3"
MAX_OVERFLOW: str = "2"
POOL_TIMEOUT: str = "13"
POOL_RECYCLE: str = "111"
MAX_CONNECTIONS_BUDGET: str = "5"
SYNCHRONOUS_COMMIT: str = "off"

# What a session's `SHOW statement_timeout` reports for 45000 ms.
EXPECTED_STATEMENT_TIMEOUT: str = "45s"

# Prints the settings the service's own engine ended up with, from inside the pod.
POSTGRES_SETTINGS_PROBE: str = """
import json
from sqlalchemy import text
from trustyai_service.service.data.storage import get_storage_interface

storage = get_storage_interface()
with storage._engine.connect() as conn:
    out = {
        "statement_timeout": conn.execute(text("show statement_timeout")).scalar(),
        "application_name": conn.execute(text("show application_name")).scalar(),
        "synchronous_commit": conn.execute(text("show synchronous_commit")).scalar(),
        "ssl": conn.execute(text("select ssl from pg_stat_ssl where pid = pg_backend_pid()")).scalar(),
    }
out["pool_size"] = storage._engine.pool.size()
out["max_overflow"] = storage._engine.pool._max_overflow
out["pool_timeout"] = storage._engine.pool._timeout
out["pool_recycle"] = storage._engine.pool._recycle
print(json.dumps(out))
"""

SQLITE_SETTINGS_PROBE: str = """
import json
from trustyai_service.service.data.storage import get_storage_interface

storage = get_storage_interface()
with storage._engine.connect() as conn:
    out = {
        "journal_mode": conn.exec_driver_sql("PRAGMA journal_mode").scalar(),
        "synchronous": conn.exec_driver_sql("PRAGMA synchronous").scalar(),
        "busy_timeout": conn.exec_driver_sql("PRAGMA busy_timeout").scalar(),
        "cache_size": conn.exec_driver_sql("PRAGMA cache_size").scalar(),
    }
print(json.dumps(out))
"""

# SQLite reports PRAGMA synchronous as an integer: OFF=0, NORMAL=1, FULL=2, EXTRA=3.
SQLITE_SYNCHRONOUS_FULL: int = 2


@pytest.mark.ai_safety
@pytest.mark.tier2
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-trustyai-db-config-pg"})],
    indirect=True,
)
class TestPostgresSettingsFromConfigMap:
    """Valid ConfigMap values reach the live PostgreSQL session and the pool."""

    @pytest.fixture(scope="class")
    def configured_service(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        postgres_db: Deployment,
        postgres_db_credentials_secret: Secret,
        postgres_service_ca_secret: Callable[[str], AbstractContextManager[Secret]],
        deploy_db_config_map: Callable[..., AbstractContextManager[ConfigMap]],
        teardown_resources: bool,
    ) -> Generator[Deployment, Any, Any]:
        """CR-created PostgreSQL service whose SQL tuning comes from its ConfigMap reference."""
        with (
            postgres_service_ca_secret(CONFIGURED_DEPLOYMENT),
            deploy_db_config_map(
                name=DB_CONFIG_MAP_NAME,
                data={},
            ) as tuning_config_map,
            create_database_backed_trustyai_service(
                client=admin_client,
                namespace=model_namespace.name,
                name=CONFIGURED_DEPLOYMENT,
                credentials_secret_name=postgres_db_credentials_secret.name,
                teardown=teardown_resources,
                database_tuning_config_map_name=DB_CONFIG_MAP_NAME,
            ) as (_, deployment),
        ):
            update_database_tuning_config_map(
                config_map=tuning_config_map,
                data={
                    ENV_POOL_SIZE: POOL_SIZE,
                    ENV_MAX_OVERFLOW: MAX_OVERFLOW,
                    ENV_POOL_TIMEOUT: POOL_TIMEOUT,
                    ENV_POOL_RECYCLE: POOL_RECYCLE,
                    ENV_MAX_CONNECTIONS_BUDGET: MAX_CONNECTIONS_BUDGET,
                    ENV_APPLICATION_NAME: APPLICATION_NAME,
                    ENV_STATEMENT_TIMEOUT_MS: STATEMENT_TIMEOUT_MS,
                    ENV_SYNCHRONOUS_COMMIT: SYNCHRONOUS_COMMIT,
                },
            )
            restart_trustyai_service_pod(
                client=admin_client,
                namespace=model_namespace.name,
                deployment=deployment,
            )
            yield deployment

    @pytest.fixture(scope="class")
    def configured_client(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        configured_service: Deployment,
        current_client_token: str,
    ) -> StorageBackendClient:
        """HTTP client for the ConfigMap-configured service."""
        return StorageBackendClient(
            client=admin_client,
            namespace=model_namespace.name,
            name=CONFIGURED_DEPLOYMENT,
            token=current_client_token,
        )

    def test_settings_reach_the_database_session(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        configured_service: Deployment,
    ) -> None:
        """The session the service opens carries the ConfigMap's tuning, over verified TLS."""
        settings = run_python_in_service_pod(
            client=admin_client,
            namespace=model_namespace.name,
            name=CONFIGURED_DEPLOYMENT,
            code=POSTGRES_SETTINGS_PROBE,
        )
        assert settings["application_name"] == APPLICATION_NAME, f"application_name not applied: {settings}"
        assert settings["statement_timeout"] == EXPECTED_STATEMENT_TIMEOUT, f"statement_timeout not applied: {settings}"
        assert settings["synchronous_commit"] == SYNCHRONOUS_COMMIT, f"synchronous_commit not applied: {settings}"
        assert settings["ssl"] is True, f"Connection is not using TLS: {settings}"

    def test_pool_sizing_comes_from_the_config_map(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        configured_service: Deployment,
    ) -> None:
        """The engine's pool is the size the ConfigMap asked for, not the default."""
        settings = run_python_in_service_pod(
            client=admin_client,
            namespace=model_namespace.name,
            name=CONFIGURED_DEPLOYMENT,
            code=POSTGRES_SETTINGS_PROBE,
        )
        assert settings["pool_size"] == int(POOL_SIZE), f"pool size not applied: {settings}"
        assert settings["max_overflow"] == int(MAX_OVERFLOW), f"max overflow not applied: {settings}"
        assert settings["pool_timeout"] == int(POOL_TIMEOUT), f"pool timeout not applied: {settings}"
        assert settings["pool_recycle"] == int(POOL_RECYCLE), f"pool recycle not applied: {settings}"

    def test_upload_round_trip_with_tuned_settings(self, configured_client: StorageBackendClient) -> None:
        """Tuned settings do not change behaviour: an upload is readable back through `/info`."""
        model_name = "db-config-round-trip-model"
        n_rows = 5

        response = configured_client.upload(payload=build_upload_payload(model_name=model_name, n_rows=n_rows))
        assert response.status_code == HTTPStatus.OK, f"Upload failed: {response.text}"

        model_info = wait_for_model_observations(
            storage_client=configured_client, model_name=model_name, expected=n_rows
        )
        assert model_info["data"]["observations"] == n_rows


@pytest.mark.ai_safety
@pytest.mark.tier3
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-trustyai-db-config-bad"})],
    indirect=True,
)
class TestInvalidSettingsFailFast:
    """The service refuses settings the operator cannot check, and the log names the key and the value."""

    @pytest.mark.parametrize(
        "case_id, config, expected_fragments",
        [
            pytest.param(
                "pool0",
                {ENV_POOL_SIZE: "0"},
                [f"{ENV_POOL_SIZE}='0'"],
                id="test_pool_size_zero_is_rejected_by_service",
            ),
            pytest.param(
                "budget",
                {ENV_POOL_SIZE: "5", ENV_MAX_OVERFLOW: "10", ENV_MAX_CONNECTIONS_BUDGET: "8"},
                [f"{ENV_MAX_CONNECTIONS_BUDGET}=8", "exceeds the per-pod budget"],
                id="test_pool_ceiling_over_connection_budget",
            ),
        ],
    )
    def test_invalid_config_map_value_is_rejected(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        postgres_db: Deployment,
        postgres_db_credentials_secret: Secret,
        postgres_service_ca_secret: Callable[[str], AbstractContextManager[Secret]],
        deploy_db_config_map: Callable[..., AbstractContextManager[ConfigMap]],
        teardown_resources: bool,
        case_id: str,
        config: dict[str, str],
        expected_fragments: list[str],
    ) -> None:
        """
        Given a tuning ConfigMap with a value only the service can reject
        When the service pod restarts
        Then the log names the key and value and the pod never becomes ready
        """
        name = f"{TRUSTYAI_SERVICE_NAME}-bad-{case_id}"
        config_map_name = f"{DB_CONFIG_MAP_NAME}-{case_id}"

        with (
            postgres_service_ca_secret(name),
            deploy_db_config_map(name=config_map_name, data={}) as tuning_config_map,
            create_database_backed_trustyai_service(
                client=admin_client,
                namespace=model_namespace.name,
                name=name,
                credentials_secret_name=postgres_db_credentials_secret.name,
                teardown=teardown_resources,
                database_tuning_config_map_name=config_map_name,
            ) as (_, deployment),
        ):
            update_database_tuning_config_map(config_map=tuning_config_map, data=config)
            restart_trustyai_service_pod(
                client=admin_client,
                namespace=model_namespace.name,
                deployment=deployment,
                wait_for_replicas=False,
            )
            logs = wait_for_pod_log_message(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector=deployment_pod_selector(deployment),
                message=expected_fragments[0],
            )
            for fragment in expected_fragments:
                assert fragment in logs, f"Error should mention {fragment!r}; logs: {logs[-2000:]}"
            assert wait_for_deployment_unready(
                client=admin_client,
                namespace=model_namespace.name,
                name=name,
            ), f"Service became ready with invalid settings {config}"


@pytest.mark.ai_safety
@pytest.mark.tier3
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-trustyai-db-config-operator"})],
    indirect=True,
)
class TestInvalidSettingsRejectedByOperator:
    """The operator rejects bad tuning values on the CR before it creates or updates any pod."""

    @pytest.mark.parametrize(
        "case_id, config, rejected_key, rejected_value",
        [
            pytest.param(
                "bool",
                {ENV_SYNCHRONOUS_COMMIT: "maybe"},
                ENV_SYNCHRONOUS_COMMIT,
                "maybe",
                id="test_invalid_synchronous_commit",
            ),
            pytest.param(
                "stmt",
                {ENV_STATEMENT_TIMEOUT_MS: "-5"},
                ENV_STATEMENT_TIMEOUT_MS,
                "-5",
                id="test_negative_statement_timeout",
            ),
            pytest.param(
                "both",
                {ENV_POOL_SIZE: "0", ENV_MAX_OVERFLOW: "-1"},
                # Keys are checked in sorted order and only the first bad one is reported.
                ENV_MAX_OVERFLOW,
                "-1",
                id="test_first_bad_key_is_reported",
            ),
        ],
    )
    def test_invalid_config_map_value_is_reported_on_the_cr(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        postgres_db: Deployment,
        postgres_db_credentials_secret: Secret,
        postgres_service_ca_secret: Callable[[str], AbstractContextManager[Secret]],
        deploy_db_config_map: Callable[..., AbstractContextManager[ConfigMap]],
        teardown_resources: bool,
        case_id: str,
        config: dict[str, str],
        rejected_key: str,
        rejected_value: str,
    ) -> None:
        """
        Given a tuning ConfigMap with a value the operator can validate
        When a TrustyAIService references it
        Then DatabaseTuningAvailable is False with reason InvalidConfiguration naming the key and value
        And the service is not Ready
        """
        name = f"{TRUSTYAI_SERVICE_NAME}-op-{case_id}"
        config_map_name = f"{DB_CONFIG_MAP_NAME}-op-{case_id}"
        storage = {
            "format": "DATABASE",
            "size": "1Gi",
            "databaseConfigurations": postgres_db_credentials_secret.name,
            "databaseTuning": config_map_name,
        }

        with (
            postgres_service_ca_secret(name),
            deploy_db_config_map(name=config_map_name, data=config),
            TrustyAIService(
                client=admin_client,
                name=name,
                namespace=model_namespace.name,
                storage=storage,
                metrics=TAI_METRICS_CONFIG,
                teardown=teardown_resources,
            ) as service,
        ):
            tuning = wait_for_trustyai_condition(
                service=service, condition_type="DatabaseTuningAvailable", status="False"
            )
            assert tuning["reason"] == "InvalidConfiguration", f"Unexpected reason: {tuning}"
            assert f'"{rejected_key}"' in tuning["message"], f"Message should name {rejected_key}: {tuning}"
            assert f'"{rejected_value}"' in tuning["message"], f"Message should show {rejected_value!r}: {tuning}"
            wait_for_trustyai_condition(service=service, condition_type="Ready", status="False")


@pytest.mark.ai_safety
@pytest.mark.tier2
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-trustyai-db-config-sqlite"})],
    indirect=True,
)
class TestSqlitePragmasFromConfigMap:
    """In-memory SQLite applies the ConfigMap's pragmas to every connection."""

    @pytest.fixture(scope="class")
    def configured_sqlite_service(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        sqlite_credentials_secret: Secret,
        deploy_db_config_map: Callable[..., AbstractContextManager[ConfigMap]],
        teardown_resources: bool,
    ) -> Generator[Deployment, Any, Any]:
        """CR-created SQLite service whose pragmas come from its tuning ConfigMap."""
        with (
            deploy_db_config_map(
                name=DB_CONFIG_MAP_NAME,
                data={},
            ) as tuning_config_map,
            create_database_backed_trustyai_service(
                client=admin_client,
                namespace=model_namespace.name,
                name=SQLITE_CONFIGURED_DEPLOYMENT,
                credentials_secret_name=sqlite_credentials_secret.name,
                teardown=teardown_resources,
                database_tuning_config_map_name=DB_CONFIG_MAP_NAME,
            ) as (_, deployment),
        ):
            update_database_tuning_config_map(
                config_map=tuning_config_map,
                data={
                    ENV_SQLITE_JOURNAL_MODE: "memory",
                    ENV_SQLITE_SYNCHRONOUS: "FULL",
                    ENV_SQLITE_BUSY_TIMEOUT_MS: "7000",
                    ENV_SQLITE_CACHE_SIZE_KB: "4000",
                },
            )
            restart_trustyai_service_pod(
                client=admin_client,
                namespace=model_namespace.name,
                deployment=deployment,
            )
            yield deployment

    def test_pragmas_are_applied(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        configured_sqlite_service: Deployment,
    ) -> None:
        """Each pragma on a fresh connection matches the ConfigMap."""
        pragmas = run_python_in_service_pod(
            client=admin_client,
            namespace=model_namespace.name,
            deployment=configured_sqlite_service,
            code=SQLITE_SETTINGS_PROBE,
        )
        assert pragmas["journal_mode"] == "memory", f"journal_mode not applied: {pragmas}"
        assert pragmas["synchronous"] == SQLITE_SYNCHRONOUS_FULL, f"synchronous not applied: {pragmas}"
        assert pragmas["busy_timeout"] == 7000, f"busy_timeout not applied: {pragmas}"
        # A negative cache_size is in KiB, so a 4000 KiB request reads back as -4000.
        assert pragmas["cache_size"] == -4000, f"cache_size not applied: {pragmas}"

    def test_upload_round_trip_with_pragmas(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        configured_sqlite_service: Deployment,
        current_client_token: str,
    ) -> None:
        """The configured pragmas do not break reads and writes through the API."""
        storage_client = StorageBackendClient(
            client=admin_client,
            namespace=model_namespace.name,
            name=SQLITE_CONFIGURED_DEPLOYMENT,
            token=current_client_token,
        )
        model_name = "sqlite-pragma-model"
        n_rows = 4

        response = storage_client.upload(payload=build_upload_payload(model_name=model_name, n_rows=n_rows))
        assert response.status_code == HTTPStatus.OK, f"Upload failed: {response.text}"

        model_info = wait_for_model_observations(storage_client=storage_client, model_name=model_name, expected=n_rows)
        assert model_info["data"]["observations"] == n_rows
