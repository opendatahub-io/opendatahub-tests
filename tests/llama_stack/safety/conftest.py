import os
import subprocess
from base64 import b64encode
from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from pytest_testconfig import py_config


@pytest.fixture(scope="class")
def guardrails_orchestrator_ssl_cert(guardrails_orchestrator_route: Route):
    hostname = guardrails_orchestrator_route.host

    try:
        result = subprocess.run(
            args=["openssl", "s_client", "-showcerts", "-connect", f"{hostname}:443"],
            input="",
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        if result.returncode != 0 and "CONNECTED" not in result.stdout:
            raise RuntimeError(f"Failed to connect to {hostname}: {result.stderr}")

        cert_lines = []
        in_cert = False
        for line in result.stdout.splitlines():
            if "-----BEGIN CERTIFICATE-----" in line:
                in_cert = True
            if in_cert:
                cert_lines.append(line)
            if "-----END CERTIFICATE-----" in line:
                in_cert = False

        if not cert_lines:
            raise RuntimeError(f"No certificate found in response from {hostname}")

        filepath = os.path.join(py_config["tmp_base_dir"], "gorch_cert.crt")
        with open(filepath, "w") as f:
            f.write("\n".join(cert_lines))

        return filepath  # noqa: TRY300

    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Could not get certificate from {hostname}: {e}")


@pytest.fixture(scope="class")
def guardrails_orchestrator_ssl_cert_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    guardrails_orchestrator_ssl_cert: str,  # ← Add dependency and use correct cert
) -> Generator[Secret, Any]:
    with open(guardrails_orchestrator_ssl_cert, "r") as f:
        cert_content = f.read()

    with Secret(
        client=admin_client,
        name="orch-certificate",
        namespace=model_namespace.name,
        data_dict={"orch-certificate.crt": b64encode(cert_content.encode("utf-8")).decode("utf-8")},
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def patched_llamastack_deployment_tls_certs(llama_stack_distribution, guardrails_orchestrator_ssl_cert_secret):
    lls_deployment = Deployment(
        name=llama_stack_distribution.name,
        namespace=llama_stack_distribution.namespace,
        ensure_exists=True,
    )

    current_spec = lls_deployment.instance.spec.template.spec.to_dict()

    current_spec["volumes"].append({
        "name": "router-ca",
        "secret": {"secretName": "orch-certificate"},  # pragma: allowlist secret
    })

    for container in current_spec["containers"]:
        if container["name"] == "llama-stack":
            container["volumeMounts"].append({"name": "router-ca", "mountPath": "/etc/llama/certs", "readOnly": True})
            break

    with ResourceEditor(patches={lls_deployment: {"spec": {"template": {"spec": current_spec}}}}) as _:
        initial_replicas = lls_deployment.replicas
        lls_deployment.scale_replicas(replica_count=0)
        lls_deployment.scale_replicas(replica_count=initial_replicas)
        lls_deployment.wait_for_replicas()
        yield lls_deployment
