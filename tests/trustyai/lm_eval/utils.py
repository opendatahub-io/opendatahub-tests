from kubernetes.dynamic import DynamicClient
from ocp_resources.lm_eval_job import LMEvalJob
from ocp_resources.pod import Pod

from timeout_sampler import retry


def verify_lmevaljob_running(client: DynamicClient, lmevaljob: LMEvalJob) -> None:
    # Verifies that the LMEvalJob Pod gets to Running and stays either Running or Succeeded for at least 2 minutes
    lmevaljob_pod = Pod(client=client, name=lmevaljob.name, namespace=lmevaljob.namespace, wait_for_resource=True)
    lmevaljob_pod.wait_for_status(status=lmevaljob_pod.Status.RUNNING)

    @retry(wait_timeout=10, sleep=1)
    def _check_pod_status() -> bool:
        assert lmevaljob_pod.status in (lmevaljob_pod.Status.RUNNING, lmevaljob_pod.Status.SUCCEEDED)
        return True

    _check_pod_status()
