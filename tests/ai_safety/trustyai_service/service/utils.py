from typing import Any

from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.resource import ResourceEditor
from ocp_resources.trustyai_service import TrustyAIService
from timeout_sampler import retry

from utilities.constants import TRUSTYAI_SERVICE_NAME
from utilities.resources.pod import Pod


def stage_trustyai_legacy_migration_file(
    client: DynamicClient,
    namespace: str,
    service_name: str,
    model_name: str,
) -> None:
    """Stage the model's PVC datasets in the legacy filename consumed by the migrator."""
    deployment = Deployment(client=client, namespace=namespace, name=service_name, ensure_exists=True)
    match_labels = deployment.instance.to_dict()["spec"]["selector"]["matchLabels"]
    selector = ",".join(f"{key}={value}" for key, value in match_labels.items())
    pods = list(Pod.get(client=client, namespace=namespace, label_selector=selector))
    assert pods, f"No pod found for Deployment {service_name}"

    python_code = f"""
from pathlib import Path
import h5py

model_name = {model_name!r}
pvc_path = Path("/inputs")
legacy_path = pvc_path / f"{{model_name}}_trustyai.hdf5"
if legacy_path.exists():
    print(f"Using existing legacy migration file {{legacy_path}}")
else:
    source_paths = sorted(pvc_path.glob(f"{{model_name}}_*_data.csv"))
    datasets_copied = 0

    with h5py.File(legacy_path, "w") as legacy_file:
        for source_path in source_paths:
            with h5py.File(source_path, "r") as source_file:
                for dataset_name in source_file:
                    dataset = source_file[dataset_name]
                    if len(dataset):
                        source_file.copy(dataset_name, legacy_file, name=dataset_name)
                        datasets_copied += 1

    if not datasets_copied:
        legacy_path.unlink(missing_ok=True)
        raise RuntimeError(f"No non-empty HDF5 datasets found for {{model_name}} in {{pvc_path}}")
    print(f"Staged {{datasets_copied}} datasets in {{legacy_path}}")
"""
    result = pods[0].execute(
        command=["python", "-c", python_code],
        container=TRUSTYAI_SERVICE_NAME,
        timeout=60,
    )
    assert result.rc == 0, f"Failed to stage legacy PVC migration file: {result.stderr or result.stdout}"


def prepare_trustyai_db_migration(
    client: DynamicClient,
    namespace: str,
    service_name: str,
    original_deployment_spec: dict[str, Any],
) -> None:
    """Enable PVC-to-database migration after the operator applies DATABASE storage."""
    source_volume = next(volume for volume in original_deployment_spec["volumes"] if "persistentVolumeClaim" in volume)
    source_mount = next(
        mount
        for container in original_deployment_spec["containers"]
        if container["name"] == TRUSTYAI_SERVICE_NAME
        for mount in container.get("volumeMounts", [])
        if mount["mountPath"] == "/inputs"
    )

    @retry(wait_timeout=300, sleep=5)
    def _database_storage_is_reconciled() -> bool:
        deployment = Deployment(client=client, namespace=namespace, name=service_name, ensure_exists=True)
        deployment_data = deployment.instance.to_dict()
        metadata = deployment_data.get("metadata", {})
        status = deployment_data.get("status", {})
        generation = metadata.get("generation", 0)
        observed_generation = status.get("observedGeneration", 0)
        containers = deployment_data.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
        service_container = next(
            (container for container in containers if container.get("name") == TRUSTYAI_SERVICE_NAME),
            {},
        )
        environment = {item.get("name"): item.get("value") for item in service_container.get("env", [])}
        return observed_generation >= generation and environment.get("SERVICE_STORAGE_FORMAT") == "DATABASE"

    _database_storage_is_reconciled()

    deployment = Deployment(client=client, namespace=namespace, name=service_name, ensure_exists=True)
    deployment_data = deployment.instance.to_dict()
    pod_spec = deployment_data["spec"]["template"]["spec"]
    volumes = pod_spec.get("volumes", [])
    volume_mounts = next(
        container.get("volumeMounts", [])
        for container in pod_spec["containers"]
        if container.get("name") == TRUSTYAI_SERVICE_NAME
    )

    patch: dict[str, Any] = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": TRUSTYAI_SERVICE_NAME,
                            "env": [{"name": "DATABASE_ATTEMPT_MIGRATION", "value": "true"}],
                        }
                    ]
                }
            }
        }
    }
    existing_source_volume = next(
        (volume for volume in volumes if volume.get("name") == source_volume["name"]),
        None,
    )
    if existing_source_volume != source_volume:
        patch["spec"]["template"]["spec"]["volumes"] = [source_volume]

    existing_source_mount = next(
        (mount for mount in volume_mounts if mount.get("mountPath") == "/inputs"),
        None,
    )
    if existing_source_mount is not None:
        assert existing_source_mount.get("name") == source_mount["name"], (
            "Operator mounted a different volume at /inputs; refusing to replace the migration source."
        )
    else:
        patch["spec"]["template"]["spec"]["containers"][0]["volumeMounts"] = [source_mount]

    deployment.api.patch(
        body=patch,
        name=service_name,
        namespace=namespace,
        content_type="application/strategic-merge-patch+json",
    )


@retry(wait_timeout=600, sleep=5)
def wait_for_trustyai_db_migration_rollout(client: DynamicClient, namespace: str, service_name: str) -> bool:
    """Wait for a ready, observed DATABASE rollout with migration enabled."""
    deployment = Deployment(client=client, namespace=namespace, name=service_name, ensure_exists=True)
    deployment_data = deployment.instance.to_dict()
    metadata = deployment_data.get("metadata", {})
    status = deployment_data.get("status", {})
    spec = deployment_data.get("spec", {})
    generation = metadata.get("generation", 0)
    observed_generation = status.get("observedGeneration", 0)
    replicas = spec.get("replicas", 1)
    containers = spec.get("template", {}).get("spec", {}).get("containers", [])
    service_container = next(
        (container for container in containers if container.get("name") == TRUSTYAI_SERVICE_NAME),
        {},
    )
    environment = {item.get("name"): item.get("value") for item in service_container.get("env", [])}
    return (
        observed_generation >= generation
        and status.get("updatedReplicas", 0) == replicas
        and status.get("readyReplicas", 0) == replicas
        and environment.get("SERVICE_STORAGE_FORMAT") == "DATABASE"
        and environment.get("DATABASE_ATTEMPT_MIGRATION") == "true"
    )


def patch_trustyai_service_cr(trustyai_service: TrustyAIService, patches: dict[str, Any]) -> TrustyAIService:
    ResourceEditor(patches={trustyai_service: patches}).update()
    return trustyai_service
