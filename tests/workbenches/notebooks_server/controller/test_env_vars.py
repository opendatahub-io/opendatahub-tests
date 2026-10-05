from typing import Any

import pytest
from ocp_resources.namespace import Namespace
from ocp_resources.notebook import Notebook
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from pyhelper_utils.shell import run_command

from utilities.constants import Timeout

CUSTOM_ENV_VARS: list[dict[str, str]] = [
    {"name": "MY_CUSTOM_VAR", "value": "hello-from-notebook"},
    {"name": "DATASET_PATH", "value": "/opt/app-root/data"},
    {"name": "DEBUG_MODE", "value": "true"},
]


class TestEnvironmentVariables:
    """Verify that environment variables from the Notebook CR propagate to the pod."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "unprivileged_model_namespace,users_persistent_volume_claim,default_notebook,notebook_pod",
        [
            pytest.param(
                {
                    "name": "test-nb-envvars",
                    "add-dashboard-label": True,
                },
                {"name": "test-nb-envvars"},
                {
                    "namespace": "test-nb-envvars",
                    "name": "test-nb-envvars",
                    "extra_env_vars": CUSTOM_ENV_VARS,
                },
                {"timeout": Timeout.TIMEOUT_2MIN},
                id="test_env_vars_propagation",
            )
        ],
        indirect=True,
    )
    def test_env_vars_propagation(
        self,
        notebook_pod: Pod,
        unprivileged_model_namespace: Namespace,
        users_persistent_volume_claim: PersistentVolumeClaim,
        default_notebook: Notebook,
    ) -> None:
        """Verify custom env vars propagate from Notebook CR to pod spec and runtime.

        Given a Notebook CR with custom env vars in the container spec,
        When the controller creates the pod,
        Then the notebook container spec should contain those env vars,
            and they should be accessible via printenv inside the running container.
        """
        notebook_container = self._find_notebook_container(pod=notebook_pod, notebook_name=default_notebook.name)
        assert notebook_container, (
            f"Notebook container '{default_notebook.name}' not found in pod. "
            f"Available: {[container.name for container in notebook_pod.instance.spec.containers]}"
        )

        pod_env = {env.name: env.value for env in (notebook_container.env or [])}

        for expected_var in CUSTOM_ENV_VARS:
            var_name = expected_var["name"]
            var_value = expected_var["value"]

            # Verify env var is present in pod spec
            assert var_name in pod_env, (
                f"Env var '{var_name}' not found in pod container spec. Present env vars: {list(pod_env.keys())}"
            )
            assert pod_env[var_name] == var_value, (
                f"Env var '{var_name}' value mismatch in spec: expected '{var_value}', got '{pod_env[var_name]}'"
            )

            # Verify env var is accessible at runtime via exec
            success, stdout, _ = run_command(
                command=[
                    "oc",
                    "exec",
                    notebook_pod.name,
                    "-n",
                    notebook_pod.namespace,
                    "-c",
                    default_notebook.name,
                    "--",
                    "printenv",
                    var_name,
                ],
                verify_stderr=False,
                check=False,
            )
            assert success, f"Failed to exec printenv {var_name} in pod {notebook_pod.name}"
            assert stdout.strip() == var_value, (
                f"Env var '{var_name}' runtime value mismatch: expected '{var_value}', got '{stdout.strip()}'"
            )

    @staticmethod
    def _find_notebook_container(pod: Pod, notebook_name: str) -> Any | None:
        """Find the notebook container in the pod by matching the notebook name."""
        for container in pod.instance.spec.containers:
            if container.name == notebook_name:
                return container
        return None
