from collections.abc import Generator
from subprocess import run

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.network_policy import NetworkPolicy
from ocp_resources.notebook import Notebook
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod

from tests.workbenches.notebooks_server.controller.utils import build_notebook_dict
from utilities import constants
from utilities.constants import Timeout

LOGGER = structlog.get_logger(name=__name__)

NOTEBOOK_PORT = 8888
KUBE_RBAC_PROXY_PORT = 8443
FIRST_NOTEBOOK_NAME = "test-nb-netpol-first"
SECOND_NOTEBOOK_NAME = "test-nb-netpol-second"
NOTEBOOK_PVC_SIZE = "1Gi"
# Enough for an idle Jupyter + kube-rbac-proxy sidecar; this test only curls endpoints.
NOTEBOOK_RESOURCES: dict[str, dict[str, str]] = {
    "limits": {"cpu": "500m", "memory": "1Gi"},
    "requests": {"cpu": "100m", "memory": "512Mi"},
}

# curl exit codes that indicate the remote host was unreachable / timed out at connect
CURL_FAILED_TO_CONNECT = 7
CURL_OPERATION_TIMEOUT = 28
CURL_NETWORK_BLOCK_EXIT_CODES = frozenset({CURL_FAILED_TO_CONNECT, CURL_OPERATION_TIMEOUT})


@pytest.fixture()
def second_notebook_pvc(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[PersistentVolumeClaim]:
    """PVC for the second notebook used in network-policy isolation tests."""
    with PersistentVolumeClaim(
        client=unprivileged_client,
        name=SECOND_NOTEBOOK_NAME,
        namespace=unprivileged_model_namespace.name,
        label={constants.Labels.OpenDataHub.DASHBOARD: "true"},
        accessmodes=PersistentVolumeClaim.AccessMode.RWO,
        size=NOTEBOOK_PVC_SIZE,
        volume_mode=PersistentVolumeClaim.VolumeMode.FILE,
    ) as pvc:
        yield pvc


@pytest.fixture()
def second_notebook_pod(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    second_notebook_pvc: PersistentVolumeClaim,
    notebook_image: str,
) -> Generator[Pod]:
    """Create a second notebook in the same namespace for network policy testing."""
    namespace = unprivileged_model_namespace.name

    notebook_dict = build_notebook_dict(
        namespace=namespace,
        name=SECOND_NOTEBOOK_NAME,
        image_path=notebook_image,
        resources=NOTEBOOK_RESOURCES,
    )

    with Notebook(client=unprivileged_client, kind_dict=notebook_dict):
        pod = Pod(
            client=unprivileged_client,
            namespace=namespace,
            name=f"{SECOND_NOTEBOOK_NAME}-0",
        )
        pod.wait(timeout=Timeout.TIMEOUT_1MIN)
        pod.wait_for_condition(
            condition=Pod.Condition.READY,
            status=Pod.Condition.Status.TRUE,
            timeout=Timeout.TIMEOUT_2MIN,
        )
        yield pod


def _notebook_api_url(namespace: str, notebook_name: str, host: str, port: int = NOTEBOOK_PORT) -> str:
    """HTTP URL for the Jupyter API, honoring ServerApp.base_url."""
    return f"http://{host}:{port}/notebook/{namespace}/{notebook_name}/api"


def _curl_from_pod(
    *,
    pod_name: str,
    namespace: str,
    url: str,
    connect_timeout: int = 5,
    max_time: int = 10,
) -> tuple[int, int, float, str]:
    """Run curl inside a pod and return network-relevant metrics.

    Returns:
        Tuple of (curl_or_oc_returncode, http_code, time_connect_seconds, stderr).
        ``http_code`` / ``time_connect_seconds`` are 0 when curl did not produce write-out
        (for example when ``oc exec`` itself failed before curl ran).
    """
    completed = run(
        args=[
            "oc",
            "exec",
            pod_name,
            "-n",
            namespace,
            "--",
            "curl",
            "-k",
            "-sS",
            "-o",
            "/dev/null",
            "--connect-timeout",
            str(connect_timeout),
            "--max-time",
            str(max_time),
            "-w",
            "%{http_code} %{time_connect}",
            url,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    http_code = 0
    time_connect = 0.0
    parts = completed.stdout.strip().split()
    if len(parts) >= 2:
        try:
            http_code = int(parts[0])
            time_connect = float(parts[1])
        except ValueError:
            LOGGER.warning(f"Unparseable curl write-out from {pod_name}: {completed.stdout!r}")

    LOGGER.info(
        f"curl from {namespace}/{pod_name} -> {url}: "
        f"rc={completed.returncode} http={http_code} time_connect={time_connect} stderr={completed.stderr!r}"
    )
    return completed.returncode, http_code, time_connect, completed.stderr


def _assert_curl_ran(*, returncode: int, http_code: int, time_connect: float, stderr: str, context: str) -> None:
    """Fail if oc exec never reached curl (avoids treating exec/RBAC failures as NetworkPolicy blocks)."""
    produced_write_out = http_code != 0 or time_connect > 0 or returncode in CURL_NETWORK_BLOCK_EXIT_CODES
    assert produced_write_out, (
        f"{context}: oc exec/curl did not produce usable curl metrics "
        f"(rc={returncode}, http={http_code}, time_connect={time_connect}, stderr={stderr!r}). "
        f"This is an exec failure, not evidence of NetworkPolicy enforcement."
    )


class TestNetworkPolicyIsolation:
    """Verify network policy isolation between notebook pods."""

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        "unprivileged_model_namespace,users_persistent_volume_claim,default_notebook,notebook_pod",
        [
            pytest.param(
                {
                    "name": "test-nb-netpol",
                    "add-dashboard-label": True,
                },
                # claimName in build_notebook_dict equals the notebook name
                {"name": FIRST_NOTEBOOK_NAME, "size": NOTEBOOK_PVC_SIZE},
                {
                    "namespace": "test-nb-netpol",
                    "name": FIRST_NOTEBOOK_NAME,
                    "resources": NOTEBOOK_RESOURCES,
                },
                {"timeout": Timeout.TIMEOUT_2MIN},
                id="test_notebooks_cannot_reach_each_other",
            )
        ],
        indirect=True,
    )
    def test_notebooks_cannot_reach_each_other(
        self,
        unprivileged_client: DynamicClient,
        notebook_pod: Pod,
        second_notebook_pod: Pod,
        unprivileged_model_namespace: Namespace,
        users_persistent_volume_claim: PersistentVolumeClaim,
        default_notebook: Notebook,
    ) -> None:
        """Verify peer notebooks are isolated on 8888 while local/proxy paths still work.

        Given two notebook pods are running in the same namespace with controller NetworkPolicies,
        When inspecting the first notebook's control NetworkPolicy,
        Then `{name}-ctrl-np` should exist.
        When the first notebook curls its own Jupyter API on localhost:8888,
        Then the request should succeed (workbench is healthy; networking is not globally broken).
        When the second notebook curls the first notebook's kube-rbac-proxy Service on 8443,
        Then TCP should connect (oauth NetworkPolicy allows ingress from any source).
        When the second notebook curls the first notebook pod IP on port 8888,
        Then the connection should fail at connect time (ctrl NetworkPolicy denies peer ingress).
        """
        namespace = default_notebook.namespace
        first_notebook_name = default_notebook.name

        ctrl_np = NetworkPolicy(
            client=unprivileged_client,
            name=f"{first_notebook_name}-ctrl-np",
            namespace=namespace,
        )
        assert ctrl_np.exists, (
            f"Expected NetworkPolicy '{ctrl_np.name}' in '{namespace}' "
            f"(controller should create per-notebook ingress allow-list on port {NOTEBOOK_PORT})"
        )

        # Positive: notebook is reachable to itself on the Jupyter port
        local_url = _notebook_api_url(
            namespace=namespace,
            notebook_name=first_notebook_name,
            host="127.0.0.1",
        )
        local_rc, local_http, local_connect, local_err = _curl_from_pod(
            pod_name=notebook_pod.name,
            namespace=namespace,
            url=local_url,
        )
        _assert_curl_ran(
            returncode=local_rc,
            http_code=local_http,
            time_connect=local_connect,
            stderr=local_err,
            context="localhost Jupyter API check",
        )
        assert local_rc == 0 and local_http in (200, 302), (
            f"Notebook '{first_notebook_name}' could not reach its own Jupyter API at {local_url}: "
            f"rc={local_rc} http={local_http} time_connect={local_connect} stderr={local_err!r}"
        )

        # Positive: peer can still reach kube-rbac-proxy (8443 allow-any ingress)
        proxy_svc_url = f"https://{first_notebook_name}-kube-rbac-proxy.{namespace}.svc:{KUBE_RBAC_PROXY_PORT}/"
        proxy_rc, proxy_http, proxy_connect, proxy_err = _curl_from_pod(
            pod_name=second_notebook_pod.name,
            namespace=namespace,
            url=proxy_svc_url,
        )
        _assert_curl_ran(
            returncode=proxy_rc,
            http_code=proxy_http,
            time_connect=proxy_connect,
            stderr=proxy_err,
            context="peer kube-rbac-proxy reachability check",
        )
        assert proxy_connect > 0 or proxy_http > 0, (
            f"Pod '{second_notebook_pod.name}' could not open TCP to kube-rbac-proxy at {proxy_svc_url}. "
            f"Peer networking may be broken independently of notebook isolation: "
            f"rc={proxy_rc} http={proxy_http} time_connect={proxy_connect} stderr={proxy_err!r}"
        )

        # Negative: peer must not reach Jupyter on 8888
        first_pod_ip = notebook_pod.instance.status.podIP
        assert first_pod_ip, f"Could not determine IP for pod {notebook_pod.name}"

        peer_url = _notebook_api_url(
            namespace=namespace,
            notebook_name=first_notebook_name,
            host=first_pod_ip,
        )
        peer_rc, peer_http, peer_connect, peer_err = _curl_from_pod(
            pod_name=second_notebook_pod.name,
            namespace=namespace,
            url=peer_url,
        )
        _assert_curl_ran(
            returncode=peer_rc,
            http_code=peer_http,
            time_connect=peer_connect,
            stderr=peer_err,
            context="peer Jupyter isolation check",
        )
        assert peer_rc in CURL_NETWORK_BLOCK_EXIT_CODES and peer_connect <= 0 and peer_http == 0, (
            f"Pod '{second_notebook_pod.name}' reached pod '{notebook_pod.name}' at {peer_url} "
            f"(rc={peer_rc}, http={peer_http}, time_connect={peer_connect}). "
            f"NetworkPolicy '{ctrl_np.name}' should deny peer ingress on port {NOTEBOOK_PORT}."
        )
