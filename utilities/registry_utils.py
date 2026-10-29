import requests
import structlog

LOGGER = structlog.get_logger(name=__name__)


def pull_manifest_from_oci_registry(registry_url: str, repo: str, tag: str) -> dict[str, str]:
    """Pull a manifest from an OCI registry."""
    response = requests.get(
        f"{registry_url}/v2/{repo}/manifests/{tag}",
        headers={"Accept": "application/vnd.oci.image.manifest.v1+json"},
        timeout=10,
    )
    LOGGER.info(f"Manifest pull: {response.status_code}")
    assert response.status_code == 200, f"Failed to pull manifest: {response.status_code}"
    return response.json()


def pull_blob_from_oci_registry(registry_url: str, repo: str, digest: str) -> bytes:
    """Pull a blob by digest from an OCI registry."""
    response = requests.get(
        f"{registry_url}/v2/{repo}/blobs/{digest}",
        timeout=10,
    )
    LOGGER.info(f"Blob pull: {response.status_code}")
    assert response.status_code == 200, f"Failed to pull blob {digest}: {response.status_code}"
    return response.content
