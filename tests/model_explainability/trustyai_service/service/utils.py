import re
from typing import Any

from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.trustyai_service import TrustyAIService
from timeout_sampler import retry

from tests.model_explainability.trustyai_service.trustyai_service_utils import TRUSTYAI_SERVICE_NAME
from utilities.constants import Timeout

db_migration_success_regex = re.compile(r".+INFO.+Migration complete, the PVC is now safe to remove\.")


@retry(wait_timeout=Timeout.TIMEOUT_5MIN, sleep=5)
def wait_for_trustyai_db_migration_complete_log(client: DynamicClient, trustyai_service: TrustyAIService) -> bool:
    trustyai_pod = list(
        Pod.get(
            dyn_client=client,
            namespace=trustyai_service.namespace,
            label_selector=f"app.kubernetes.io/instance={trustyai_service.name}",
        )
    )[0]
    return True if db_migration_success_regex.search(trustyai_pod.log(container=TRUSTYAI_SERVICE_NAME)) else False


def patch_trustyai_service_cr(trustyai_service: TrustyAIService, patches: dict[str, Any]) -> TrustyAIService:
    ResourceEditor(patches={trustyai_service: patches}).update()
    return trustyai_service
