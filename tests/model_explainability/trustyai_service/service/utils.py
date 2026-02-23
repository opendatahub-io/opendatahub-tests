import re
from typing import Any

from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.trustyai_service import TrustyAIService
from timeout_sampler import retry

from utilities.constants import TRUSTYAI_SERVICE_NAME, Timeout


@retry(wait_timeout=Timeout.TIMEOUT_5MIN, sleep=5)
def wait_for_trustyai_db_migration_complete_log(client: DynamicClient, trustyai_service: TrustyAIService) -> bool:
    trustyai_pod = list(
        Pod.get(
            client=client,
            namespace=trustyai_service.namespace,
            label_selector=f"app.kubernetes.io/instance={trustyai_service.name}",
        )
    )[0]
    return bool(
        re.search(
            r".+INFO.+Migration complete, the PVC is now safe to remove\.",
            trustyai_pod.log(container=TRUSTYAI_SERVICE_NAME),
        )
    )


def patch_trustyai_service_cr(trustyai_service: TrustyAIService, patches: dict[str, Any]) -> TrustyAIService:
    ResourceEditor(patches={trustyai_service: patches}).update()
    return trustyai_service
