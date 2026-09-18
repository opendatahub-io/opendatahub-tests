"""Tests for NeMo Guardrails default configmap behavior.

Covers the operator logic in mountNemoConfigs that:
  - Looks up default-prefixed ConfigMaps from the operator namespace.
  - Copies them into the CR namespace (with owner references for GC).
  - Patches the copied CM with the nemo-guardrails-config=true label.
  - Uses shortened volume names to avoid the 63-char K8s limit.
  - Garbage-collects the copied CMs when the NemoGuardrails CR is deleted.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.nemo_guardrails import NemoGuardrails
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from timeout_sampler import TimeoutSampler

from tests.ai_safety.nemo_guardrails.constants import (
    NEMO_DEFAULT_CONFIG_CM_PII,
    NEMO_DEFAULT_CONFIG_CM_PREFIX,
)

_CONFIG_NAME = "default-pii"
_TIMEOUT = 120
_SLEEP = 5


def _copied_cm_name(cr_name: str, source_cm_name: str) -> str:
    """Return the name the operator gives to a default CM copied into the CR namespace."""
    return f"{cr_name}-{source_cm_name}"


@pytest.mark.tier2
@pytest.mark.ai_safety
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-nemo-guardrails"})],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
class TestNemoGuardrailsDefaultConfig:
    """Structural tests for default-configmap behavior (no live traffic required)."""

    def test_default_config_cm_copied_to_cr_namespace(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_guardrails_default_config: NemoGuardrails,
    ) -> None:
        """The operator copies the default CM from the operator namespace into the CR namespace.

        The copy is named {cr-name}-{source-cm-name} to avoid collisions when multiple CRs
        in the same namespace reference the same default CM.

        Given: NemoGuardrails CR referencing a default-prefixed configmap
        When: Reconciliation runs
        Then: A ConfigMap named {cr-name}-{cm-name} appears in the CR namespace
        """
        expected_name = _copied_cm_name(
            cr_name=nemo_guardrails_default_config.name, source_cm_name=NEMO_DEFAULT_CONFIG_CM_PII
        )
        cm = ConfigMap(
            client=admin_client,
            name=expected_name,
            namespace=model_namespace.name,
        )
        assert cm.exists, (
            f"Expected default CM '{expected_name}' to be copied into CR namespace '{model_namespace.name}'"
        )

    def test_default_config_cm_labeled(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_guardrails_default_config: NemoGuardrails,
    ) -> None:
        """The copied CM in the CR namespace carries the nemo-guardrails-config=true label.

        Given: NemoGuardrails CR referencing a default-prefixed configmap
        When: The CM is copied to the CR namespace
        Then: It has the label nemo-guardrails-config=true
        """
        expected_name = _copied_cm_name(
            cr_name=nemo_guardrails_default_config.name, source_cm_name=NEMO_DEFAULT_CONFIG_CM_PII
        )
        cm = ConfigMap(
            client=admin_client,
            name=expected_name,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        labels = cm.instance.metadata.labels or {}
        assert labels.get("nemo-guardrails-config") == "true", (
            f"Expected label 'nemo-guardrails-config=true' on copied CM '{expected_name}', got: {labels}"
        )

    def test_default_config_volume_name_shortened(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_guardrails_default_config: NemoGuardrails,
    ) -> None:
        """Volume names for default-prefixed CMs are shortened so they fit within the K8s 63-char limit.

        Given: NemoGuardrails CR referencing a default-prefixed configmap
        When: The deployment is created
        Then: All volume names are ≤63 characters and do not embed the full CM name
        """
        pods = list(
            Pod.get(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector=f"app={nemo_guardrails_default_config.name}",
            )
        )
        assert pods, f"No pods found for {nemo_guardrails_default_config.name}"

        for volume in pods[0].instance.spec.volumes or []:
            assert len(volume.name) <= 63, f"Volume name '{volume.name}' exceeds the K8s 63-character limit"
            assert NEMO_DEFAULT_CONFIG_CM_PREFIX not in volume.name, (
                f"Volume name '{volume.name}' should not embed the full default CM prefix"
            )

    def test_default_config_mount_path(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_guardrails_default_config: NemoGuardrails,
    ) -> None:
        """The default CM is mounted at /app/config/{nemoConfig.name}.

        Given: NemoGuardrails CR with a nemoConfig named 'default-pii'
        When: The deployment is created
        Then: The main container has a volumeMount at /app/config/default-pii
        """
        pods = list(
            Pod.get(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector=f"app={nemo_guardrails_default_config.name}",
            )
        )
        assert pods, f"No pods found for {nemo_guardrails_default_config.name}"

        main_container = next(
            (c for c in pods[0].instance.spec.containers if "nemo" in c.name.lower()),
            pods[0].instance.spec.containers[0],
        )
        mount_paths = {vm.mountPath for vm in (main_container.volumeMounts or [])}
        expected = f"/app/config/{_CONFIG_NAME}"
        assert expected in mount_paths, f"Expected mount at '{expected}'; found: {sorted(mount_paths)}"

    def test_default_config_fallback_deploys(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_guardrails_default_config_fallback: NemoGuardrails,
    ) -> None:
        """A default-prefixed CM absent from the operator namespace is resolved from the CR namespace.

        Given: NemoGuardrails CR referencing a default-prefixed CM that only exists in the CR namespace
        When: Reconciliation runs
        Then: The deployment comes up successfully (fallback lookup succeeded)
        """
        assert nemo_guardrails_default_config_fallback.exists

        pods = list(
            Pod.get(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector=f"app={nemo_guardrails_default_config_fallback.name}",
            )
        )
        assert pods, f"No pods found for {nemo_guardrails_default_config_fallback.name}"
        assert pods[0].instance.status.phase in ("Running", "Pending"), (
            f"Unexpected pod phase: {pods[0].instance.status.phase}"
        )


@pytest.mark.tier2
@pytest.mark.ai_safety
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-nemo-guardrails"})],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
class TestNemoGuardrailsDefaultConfigCleanup:
    """Verify copied default CMs are garbage-collected when the CR is deleted."""

    def test_copied_cm_deleted_after_cr_deletion(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_api_token_secret: Secret,
    ) -> None:
        """Copied default CMs are removed from the CR namespace when the NemoGuardrails CR is deleted.

        Given: NemoGuardrails CR referencing a default-prefixed configmap
        When: The CR is deleted
        Then: The copied CM is garbage-collected from the CR namespace
        """
        with NemoGuardrails(
            client=admin_client,
            name="nemo-cleanup-test",
            namespace=model_namespace.name,
            nemo_configs=[
                {
                    "name": "cleanup-pii",
                    "configMaps": [NEMO_DEFAULT_CONFIG_CM_PII],
                    "default": True,
                }
            ],
            replicas=1,
            env=[
                {
                    "name": "OPENAI_API_KEY",
                    "valueFrom": {
                        "secretKeyRef": {
                            "name": nemo_api_token_secret.name,
                            "key": "token",
                        }
                    },
                }
            ],
        ) as nemo_cr:
            deployment = Deployment(
                client=admin_client,
                name=nemo_cr.name,
                namespace=nemo_cr.namespace,
                wait_for_resource=True,
            )
            deployment.wait_for_replicas()

            copied_cm_name = _copied_cm_name(cr_name=nemo_cr.name, source_cm_name=NEMO_DEFAULT_CONFIG_CM_PII)

            # Confirm the CM was copied into the CR namespace before deletion
            copied_cm = ConfigMap(
                client=admin_client,
                name=copied_cm_name,
                namespace=model_namespace.name,
            )
            assert copied_cm.exists, (
                f"Expected '{copied_cm_name}' to be present in '{model_namespace.name}' before CR deletion"
            )

        # CR is deleted when the `with` block exits; wait for the copied CM to be GC'd
        for sample in TimeoutSampler(
            wait_timeout=_TIMEOUT,
            sleep=_SLEEP,
            func=lambda: (
                ConfigMap(
                    client=admin_client,
                    name=copied_cm_name,
                    namespace=model_namespace.name,
                ).exists
            ),
        ):
            if not sample:
                break

        assert not ConfigMap(
            client=admin_client,
            name=copied_cm_name,
            namespace=model_namespace.name,
        ).exists, (
            f"CM '{copied_cm_name}' should have been garbage-collected from "
            f"'{model_namespace.name}' after the NemoGuardrails CR was deleted"
        )

    def test_shared_default_cm_isolation(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_api_token_secret: Secret,
    ) -> None:
        """Deleting one CR does not garbage-collect another CR's copy of the same default CM.

        Because each copy is named {cr-name}-{source-cm-name}, two CRs that reference the
        same default CM get independent copies. Deleting CR A must only remove A's copy.

        Given: Two NemoGuardrails CRs in the same namespace referencing the same default CM
        When: The first CR is deleted
        Then: Its copied CM is GC'd but the second CR's copied CM is unaffected
        """
        _env = [
            {
                "name": "OPENAI_API_KEY",
                "valueFrom": {"secretKeyRef": {"name": nemo_api_token_secret.name, "key": "token"}},
            }
        ]
        _nemo_configs = [{"name": "shared-pii", "configMaps": [NEMO_DEFAULT_CONFIG_CM_PII], "default": True}]

        with NemoGuardrails(
            client=admin_client,
            name="nemo-shared-cr-a",
            namespace=model_namespace.name,
            nemo_configs=_nemo_configs,
            replicas=1,
            env=_env,
        ) as cr_a:
            Deployment(
                client=admin_client,
                name=cr_a.name,
                namespace=cr_a.namespace,
                wait_for_resource=True,
            ).wait_for_replicas()

            with NemoGuardrails(
                client=admin_client,
                name="nemo-shared-cr-b",
                namespace=model_namespace.name,
                nemo_configs=_nemo_configs,
                replicas=1,
                env=_env,
            ) as cr_b:
                Deployment(
                    client=admin_client,
                    name=cr_b.name,
                    namespace=cr_b.namespace,
                    wait_for_resource=True,
                ).wait_for_replicas()

                cm_name_a = _copied_cm_name(cr_name=cr_a.name, source_cm_name=NEMO_DEFAULT_CONFIG_CM_PII)
                cm_name_b = _copied_cm_name(cr_name=cr_b.name, source_cm_name=NEMO_DEFAULT_CONFIG_CM_PII)

                assert cm_name_a != cm_name_b, "Each CR must get a distinct copy of the default CM"
                assert ConfigMap(client=admin_client, name=cm_name_a, namespace=model_namespace.name).exists
                assert ConfigMap(client=admin_client, name=cm_name_b, namespace=model_namespace.name).exists

            # cr_b deleted — wait for its copy to disappear
            for sample in TimeoutSampler(
                wait_timeout=_TIMEOUT,
                sleep=_SLEEP,
                func=lambda: ConfigMap(client=admin_client, name=cm_name_b, namespace=model_namespace.name).exists,
            ):
                if not sample:
                    break

            assert not ConfigMap(client=admin_client, name=cm_name_b, namespace=model_namespace.name).exists, (
                f"CR B's CM '{cm_name_b}' should be GC'd after CR B was deleted"
            )

            # cr_a is still alive — its copy must still exist
            assert ConfigMap(client=admin_client, name=cm_name_a, namespace=model_namespace.name).exists, (
                f"CR A's CM '{cm_name_a}' should still exist while CR A is alive"
            )
