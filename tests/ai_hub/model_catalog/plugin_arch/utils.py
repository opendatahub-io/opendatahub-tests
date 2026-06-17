import requests
import structlog
from kubernetes.dynamic import DynamicClient
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_hub.model_catalog.utils import get_postgres_pod_in_namespace

LOGGER = structlog.get_logger(name=__name__)

READYZ_RECOVERY_TIMEOUT: int = 120
RESTORE_LOGIN_SQL: str = "ALTER USER catalog_user LOGIN;"
CLEAR_STALE_LOCK_SQL: str = "DELETE FROM locks WHERE name = 'catalog-leader';"


def run_superuser_sql(admin_client: DynamicClient, namespace: str, sql: str) -> str:
    """Execute SQL as the postgres superuser on the catalog database pod."""
    pod = get_postgres_pod_in_namespace(admin_client=admin_client, namespace=namespace)
    return pod.execute(
        command=["psql", "-U", "postgres", "-d", "model_catalog", "-c", sql],
        container="postgresql",
    )


def poll_readyz(url: str, headers: dict[str, str], expected_code: int, timeout: int) -> requests.Response:
    """Poll /readyz until the expected status code is returned."""
    try:
        for sample in TimeoutSampler(
            wait_timeout=timeout,
            sleep=10,
            func=requests.get,
            url=url,
            headers=headers,
            verify=False,
            timeout=10,
        ):
            if sample.status_code == expected_code:
                return sample
    except TimeoutExpiredError:
        raise AssertionError(f"/readyz did not return {expected_code} within {timeout}s")


def restore_catalog(admin_client: DynamicClient, namespace: str) -> None:
    """Restore DB login and clear stale leader locks."""
    for sql in (RESTORE_LOGIN_SQL, CLEAR_STALE_LOCK_SQL):
        run_superuser_sql(admin_client=admin_client, namespace=namespace, sql=sql)
