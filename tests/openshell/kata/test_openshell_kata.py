"""Kata Containers integration tests for OpenShell.

These tests validate that OpenShell correctly provisions and manages
sandboxes running in Kata VM boundaries rather than standard containers.

They are skipped by default unless explicitly opted-in via
OPENSHELL_RUN_KATA_TESTS=true.
"""

from time import sleep, time

import grpc
import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod
from openshell._proto import openshell_pb2, sandbox_pb2
from openshell.sandbox import SandboxClient, SandboxSession
from timeout_sampler import TimeoutSampler

from tests.openshell.kata.constants import OPENSHELL_RUN_KATA_TESTS

LOGGER = structlog.get_logger(name=__name__)

pytestmark = [
    pytest.mark.open_shell,
    pytest.mark.kata,
    pytest.mark.smoke,
    pytest.mark.skipif(
        condition=not OPENSHELL_RUN_KATA_TESTS,
        reason="Kata requires bare-metal/nested virtualization. Opt-in with OPENSHELL_RUN_KATA_TESTS=true",
    ),
]


@pytest.mark.usefixtures("kata_config")
class TestOpenShellKata:
    """Kata integration tests for OpenShell."""

    @pytest.mark.dependency(name="test_kata_sandbox_launch")
    def test_kata_sandbox_launch(self, admin_client: DynamicClient, kata_sandbox: SandboxSession) -> None:
        """Validate that sandbox pods launch with `runtimeClassName: kata`
        and successfully reach the Running state.
        """
        pod = Pod(client=admin_client, name=kata_sandbox.sandbox.name, namespace="openshell")
        assert pod.exists, f"Pod {kata_sandbox.sandbox.name} does not exist"
        assert pod.instance.spec.runtimeClassName == "kata", (
            f"Expected runtimeClassName 'kata', got {pod.instance.spec.runtimeClassName}"
        )
        assert pod.status == "Running", f"Expected Pod status Running, got {pod.status}"

    @pytest.mark.dependency(name="test_supervisor_sideload_injection", depends=["test_kata_sandbox_launch"])
    def test_supervisor_sideload_injection(self, admin_client: DynamicClient, kata_sandbox: SandboxSession) -> None:
        """Validate that the supervisor binary is delivered via the
        init-container sideload path.
        """
        pod = Pod(client=admin_client, name=kata_sandbox.sandbox.name, namespace="openshell")
        assert pod.exists, f"Pod {kata_sandbox.sandbox.name} does not exist"

        init_containers = pod.instance.spec.get("initContainers", [])
        init_names = [container.name for container in init_containers]
        assert "openshell-supervisor-install" in init_names, (
            f"Expected 'openshell-supervisor-install' init container, got: {init_names}"
        )

        volumes = pod.instance.spec.get("volumes", [])
        supervisor_vol = next((vol for vol in volumes if vol.name == "openshell-supervisor-bin"), None)
        assert supervisor_vol is not None, (
            f"Expected 'openshell-supervisor-bin' volume, got: {[vol.name for vol in volumes]}"
        )
        assert supervisor_vol.get("emptyDir") is not None, (
            f"Expected emptyDir volume for supervisor binary, got: {dict(supervisor_vol)}"
        )
        assert supervisor_vol.get("image") is None, (
            "Supervisor binary must use emptyDir (init-container sideload), not an image volume"
        )

    @pytest.mark.dependency(name="test_exec_in_kata_sandbox", depends=["test_kata_sandbox_launch"])
    def test_exec_in_kata_sandbox(self, kata_sandbox: SandboxSession) -> None:
        """Validate that commands can be executed inside a Kata sandbox."""
        result = kata_sandbox.exec(["echo", "hello from kata"], timeout_seconds=30)
        assert result.exit_code == 0
        assert "hello from kata" in result.stdout

    @pytest.mark.dependency(name="test_network_policy_enforcement", depends=["test_exec_in_kata_sandbox"])
    def test_network_policy_enforcement(self, sandbox_client: SandboxClient, kata_sandbox: SandboxSession) -> None:
        """Validate that the supervisor's network policy selectively allows and
        blocks outbound connections from within the Kata VM boundary.

        Adds github.com:443 as an allowed endpoint, then verifies that
        traffic to github.com succeeds while traffic to an unlisted
        host (example.com) is blocked.
        """
        rule = openshell_pb2.PolicyMergeOperation(
            add_rule=openshell_pb2.AddNetworkRule(
                rule_name="allow_github",
                rule=sandbox_pb2.NetworkPolicyRule(
                    name="allow_github",
                    endpoints=[
                        sandbox_pb2.NetworkEndpoint(
                            host="github.com",
                            port=443,
                            access="read-only",
                            protocol="rest",
                            enforcement="enforce",
                        )
                    ],
                    binaries=[sandbox_pb2.NetworkBinary(path="/usr/bin/curl")],
                ),
            ),
        )
        update_resp = sandbox_client._stub.UpdateConfig(  # noqa: FCN001
            openshell_pb2.UpdateConfigRequest(
                name=kata_sandbox.sandbox.name,
                merge_operations=[rule],
            ),
            timeout=sandbox_client._timeout,
        )
        target_version = update_resp.version
        LOGGER.info("Policy updated", name=kata_sandbox.sandbox.name, target_version=target_version)

        for sample in TimeoutSampler(
            wait_timeout=60,
            sleep=2,
            func=lambda: (
                sandbox_client._stub.GetSandboxPolicyStatus(
                    openshell_pb2.GetSandboxPolicyStatusRequest(name=kata_sandbox.sandbox.name),
                    timeout=sandbox_client._timeout,
                ).active_version
            ),
        ):
            if sample >= target_version:
                LOGGER.info("Supervisor loaded policy", name=kata_sandbox.sandbox.name, active_version=sample)
                break

        LOGGER.info("Testing allowed outbound connection to github.com")
        allowed = kata_sandbox.exec(
            ["curl", "-I", "-s", "--connect-timeout", "10", "https://github.com"],
            timeout_seconds=30,
        )
        assert allowed.exit_code == 0, (
            f"Connection to github.com was blocked (expected allowed): exit={allowed.exit_code} stderr={allowed.stderr}"
        )

        LOGGER.info("Testing denied outbound connection to example.com")
        denied = kata_sandbox.exec(
            ["curl", "-s", "--connect-timeout", "5", "https://example.com"],
            timeout_seconds=15,
        )
        assert (
            denied.exit_code != 0
        ), f"""Connection to example.com unexpectedly succeeded (expected blocked): exit={denied.exit_code}
            stderr={denied.stderr}"""

    @pytest.mark.dependency(name="test_workspace_persistence", depends=["test_network_policy_enforcement"])
    def test_workspace_persistence(
        self, admin_client: DynamicClient, kata_sandbox: SandboxSession, sandbox_client: SandboxClient
    ) -> None:
        """Validate that workspace data persists across pod restarts.

        Writes a file to the PVC-backed /sandbox directory, deletes the
        pod (the controller recreates it), then verifies the file survived.
        """
        test_file = "/sandbox/persistence_test.txt"
        test_content = "persisted_kata_data"

        LOGGER.info("Writing file to workspace")
        write_res = kata_sandbox.exec(["sh", "-c", f"echo '{test_content}' > {test_file}"], timeout_seconds=10)
        assert write_res.exit_code == 0, f"Write failed: {write_res.stderr}"

        LOGGER.info("Deleting sandbox pod to trigger recreation")
        pod = Pod(client=admin_client, name=kata_sandbox.sandbox.name, namespace="openshell")
        assert pod.exists
        pod.delete()

        LOGGER.info("Waiting for sandbox to become ready again")
        ready_ref = sandbox_client.wait_ready(sandbox_name=kata_sandbox.sandbox.name, timeout_seconds=90)

        session = SandboxSession(client=sandbox_client, sandbox=ready_ref)
        deadline = time() + 120
        while time() < deadline:
            try:
                session.exec(["true"], timeout_seconds=10)
                break
            except grpc.RpcError:
                sleep(5)
        else:
            pytest.fail("Sandbox never became exec-ready after pod restart")

        LOGGER.info("Verifying workspace file persisted")
        read_res = session.exec(["cat", test_file], timeout_seconds=10)
        assert read_res.exit_code == 0, f"File not found after restart: {read_res.stderr}"
        assert test_content in read_res.stdout
