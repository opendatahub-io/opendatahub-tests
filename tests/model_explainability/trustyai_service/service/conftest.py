from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.maria_db import MariaDB
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.trustyai_service import TrustyAIService

from tests.model_explainability.trustyai_service.constants import (
    TAI_DB_STORAGE_CONFIG,
    TAI_METRICS_CONFIG,
)
from tests.model_explainability.trustyai_service.utils import (
    create_trustyai_service,
)
from utilities.constants import TRUSTYAI_SERVICE_NAME

INVALID_TLS_CERTIFICATE: str = "LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUJnRENDQVNlZ0F3SUJBZ0lRRGtTcXVuUWRzRmZwdi8zSm\
5TS2ZoVEFLQmdncWhrak9QUVFEQWpBVk1STXcKRVFZRFZRUURFd3B0WVhKcFlXUmlMV05oTUI0WERUSTFNRFF4TkRFME1EUXhOMW9YRFRJNE1EUXhNekUx\
TURReApOMW93RlRFVE1CRUdBMVVFQXhNS2JXRnlhV0ZrWWkxallUQlpNQk1HQnlxR1NNNDlBZ0VHQ0NxR1NNNDlBd0VICkEwSUFCQ2IxQ1IwUjV1akZ1QUR\
Gd1NsazQzUUpmdDFmTFVnOWNJNyttZ0w3bVd3MmVLUXowL04ybm9KMGpJaDYKN0NnQ2syUW1jNTdWM1podkFWQzJoU2NEbWg2aldUQlhNQTRHQTFVZER3RU\
Ivd1FFQXdJQ0JEQVBCZ05WSFJNQgpBZjhFQlRBREFRSC9NQjBHQTFVZERnUVdCQlNUa2tzSU9pL1pTbCtQRlJua2NQRlJ0QTRrMERBVkJnTlZIUkVFCkRqQ\
U1nZ3B0WVhKcFlXUmlMV05oTUFvR0NDcUdTTTQ5QkFNQ0EwY0FNRVFDSUI1Q2F6VW1WWUZQYTFkS2txUGkKbitKSEQvNVZTTGd4aHVPclgzUGcxQnlzQWlB\
RmcvTXlNWW9CZUNrUVRWdS9rUkIwK2N2Qy9RMDB4NExvVGpJaQpGdCtKMGc9PQotLS0tLUVORCBDRVJUSUZJQ0FURS0t\
LS0t"  # pragma: allowlist secret


@pytest.fixture(scope="class")
def trustyai_service_with_invalid_db_cert(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    cluster_monitoring_config: ConfigMap,
    user_workload_monitoring_config: ConfigMap,
    mariadb: MariaDB,
    trustyai_invalid_db_ca_secret: None,
) -> Generator[TrustyAIService]:
    """Create a TrustyAIService deployment with an invalid database certificate set as secret.

    Yields:
        A TrustyAIService with invalid database certificate set.
    """
    with create_trustyai_service(
        client=admin_client,
        namespace=model_namespace.name,
        storage=TAI_DB_STORAGE_CONFIG,
        metrics=TAI_METRICS_CONFIG,
        wait_for_replicas=False,
    ) as trustyai_service:
        yield trustyai_service


@pytest.fixture(scope="class")
def trustyai_invalid_db_ca_secret(
    admin_client: DynamicClient, model_namespace: Namespace, mariadb: MariaDB
) -> Generator[Secret, Any]:
    with Secret(
        client=admin_client,
        name=f"{TRUSTYAI_SERVICE_NAME}-db-ca",
        namespace=model_namespace.name,
        data_dict={"ca.crt": INVALID_TLS_CERTIFICATE},
    ) as secret:
        yield secret
