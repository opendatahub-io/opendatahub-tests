import json
from typing import Any, NamedTuple

import kubernetes.stream
from kubernetes.stream import ws_client
from ocp_resources.pod import Pod as _Pod
from timeout_sampler import TimeoutWatch


class PodExecResult(NamedTuple):
    stdout: str
    stderr: str
    rc: int


class Pod(_Pod):
    """Pod with an execute method that returns stdout, stderr, and return code."""

    def execute(
        self,
        command: list[str],
        timeout: int = 60,
        container: str = "",
        ignore_rc: bool = False,
    ) -> PodExecResult:
        """Execute a command on the pod and return stdout, stderr, and return code.

        Args:
            command: Command to run.
            timeout: Time to wait for the command.
            container: Container name where to exec the command.
            ignore_rc: Unused, kept for signature compatibility with base class.

        Returns:
            PodExecResult with stdout, stderr, and rc fields.
        """
        self.logger.info(f"Execute {command} on {self.name} ({self.node.name})")
        resp = kubernetes.stream.stream(
            api_method=self._kube_v1_api.connect_get_namespaced_pod_exec,
            name=self.name,
            namespace=self.namespace,
            command=command,
            container=container or self.instance.spec.containers[0].name,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _preload_content=False,
        )

        error_channel: dict[str, Any] = {}
        timeout_watch = TimeoutWatch(timeout=timeout)
        while resp.is_open():
            resp.run_forever(timeout=2)
            try:
                error_channel = json.loads(resp.read_channel(ws_client.ERROR_CHANNEL))
                break
            except json.decoder.JSONDecodeError:
                if timeout_watch.remaining_time() <= 0:
                    resp.close()
                    return PodExecResult(stdout="", stderr="command timed out", rc=-1)

        stdout = resp.read_stdout(timeout=5) or ""
        stderr = resp.read_stderr(timeout=5) or ""

        rc = 0
        if error_channel.get("status") == "Failure":
            causes = error_channel.get("details", {}).get("causes", [])
            rc = next((int(cause["message"]) for cause in causes if cause.get("reason") == "ExitCode"), -1)

        return PodExecResult(stdout=stdout, stderr=stderr, rc=rc)
