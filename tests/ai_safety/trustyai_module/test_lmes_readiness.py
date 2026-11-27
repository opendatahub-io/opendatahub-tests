import time
from uuid import uuid4

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.lm_eval_job import LMEvalJob
from ocp_resources.namespace import Namespace

from tests.ai_safety.trustyai_module.utils import (
    get_condition_status,
    validate_trustyai_module_ready,
)
from utilities.resources.trustyai_module import TrustyAI


@pytest.mark.tier1
@pytest.mark.ai_safety
@pytest.mark.component_health
@pytest.mark.usefixtures("modular_trustyai_architecture")
def test_lmevaljob_does_not_block_module_readiness(
    admin_client: DynamicClient,
    trustyai_module: TrustyAI,
) -> None:
    """Given a Ready module, when an evaluation job exists, then the module stays Ready."""
    validate_trustyai_module_ready(trustyai_module=trustyai_module)
    assert trustyai_module.instance.spec.enabledServices.lmes, "This regression requires LMES enabled"

    # Suspension avoids creating a pod or downloading a model. The regression
    # is triggered by the CR's existence, regardless of its lifecycle state.
    with (
        Namespace(client=admin_client, name=f"lmes-readiness-{uuid4().hex[:8]}") as namespace,
        LMEvalJob(
            client=admin_client,
            name="module-readiness",
            namespace=namespace.name,
            model="hf",
            task_list={"taskNames": ["arc_easy"]},
            limit="1",
            suspend=True,
        ) as job,
    ):
        # Poll through multiple module reconciles rather than accepting a
        # stale Ready condition immediately after creating the job.
        for _ in range(15):
            time.sleep(2)
            assert job.exists, "Evaluation job disappeared during readiness observation"
            assert get_condition_status(trustyai_module=trustyai_module, condition_type="Ready") == "True", (
                f"LMEvalJob must not block module readiness: {trustyai_module.instance.status.conditions}"
            )
            assert get_condition_status(trustyai_module=trustyai_module, condition_type="Degraded") == "False", (
                f"LMEvalJob must not degrade the module: {trustyai_module.instance.status.conditions}"
            )
