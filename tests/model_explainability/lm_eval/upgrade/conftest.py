from typing import Any, Generator

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.lm_eval_job import LMEvalJob
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from pytest import Config, FixtureRequest

from tests.model_explainability.lm_eval.conftest import LMEVALJOB_NAME
from tests.model_explainability.lm_eval.utils import get_lmevaljob_pod
from utilities.exceptions import MissingParameter

@pytest.fixture(scope="class")
def lmevaljob_hf(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    patched_dsc_lmeval_allow_all: DataScienceCluster,
    lmeval_hf_access_token: Secret,
    pytestconfig: pytest.Config,
    teardown_resources: bool,
) -> Generator[LMEvalJob, None, None]:

    is_post_upgrade = pytestconfig.option.post_upgrade

    if is_post_upgrade:
        job = LMEvalJob(
            client=admin_client,
            name=LMEVALJOB_NAME,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        yield job
        job.clean_up()
        return

    job = LMEvalJob(
        client=admin_client,
        name=LMEVALJOB_NAME,
        namespace=model_namespace.name,
        model="hf",
        model_args=[{"name": "pretrained", "value": "rgeada/tiny-untrained-granite"}],
        task_list=request.param.get("task_list"),
        log_samples=True,
        allow_online=True,
        allow_code_execution=True,
        teardown=teardown_resources,
        system_instruction="Be concise. At every point give the shortest acceptable answer.",
        chat_template={"enabled": True},
        limit="0.01",
        pod={
            "container": {
                "resources": {
                    "limits": {"cpu": "1", "memory": "8Gi"},
                    "requests": {"cpu": "1", "memory": "8Gi"},
                },
                "env": [
                    {
                        "name": "HF_TOKEN",
                        "valueFrom": {
                            "secretKeyRef": {
                                "name": "hf-secret",
                                "key": "HF_ACCESS_TOKEN",
                            },
                        },
                    },
                    {"name": "HF_ALLOW_CODE_EVAL", "value": "1"},
                ],
            },
        },
    )
    job.create()
    yield job


@pytest.fixture(scope="class")
def lmevaljob_hf_pod(admin_client: DynamicClient, lmevaljob_hf: LMEvalJob) -> Generator[Pod, Any, Any]:
    yield get_lmevaljob_pod(client=admin_client, lmevaljob=lmevaljob_hf)



@pytest.fixture(scope="class")
def lmeval_hf_access_token(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    pytestconfig: Config,
) -> Secret:
    hf_access_token = pytestconfig.option.hf_access_token
    if not hf_access_token:
        raise MissingParameter(
            "HF access token is not set. "
            "Either pass with `--hf-access-token` or set `HF_ACCESS_TOKEN` environment variable"
        )
    with Secret(
        client=admin_client,
        name="hf-secret",
        namespace=model_namespace.name,
        string_data={
            "HF_ACCESS_TOKEN": hf_access_token,
        },
        wait_for_resource=True,
    ) as secret:
        yield secret
