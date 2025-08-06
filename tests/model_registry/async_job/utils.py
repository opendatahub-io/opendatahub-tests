import hashlib
import requests
import json

from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)


def push_blob_to_oci_registry(registry_url: str, data: bytes, repo: str = "test/simple-artifact") -> str:
    """
    Push a blob to an OCI registry.
    https://specs.opencontainers.org/distribution-spec/?v=v1.0.0#pushing-blobs
    POST to /v2/<repo>/blobs/uploads/ in order to initiate the upload
    The response will contain a Location header that contains the upload URL
    PUT to the Location URL with the data to be uploaded
    """

    blob_digest = f"sha256:{hashlib.sha256(data).hexdigest()}"

    LOGGER.info(f"Pushing blob with digest: {blob_digest}")

    upload_response = requests.post(f"{registry_url}/v2/{repo}/blobs/uploads/", timeout=10)
    LOGGER.info(f"Blob upload initiation: {upload_response.status_code}")
    assert upload_response.status_code == 202, f"Failed to initiate blob upload: {upload_response.status_code}"

    upload_location = upload_response.headers.get("Location")
    LOGGER.info(f"Upload location: {upload_location}")
    base_url = f"{registry_url}{upload_location}"
    upload_url = f"{base_url}?digest={blob_digest}"
    response = requests.put(url=upload_url, data=data, headers={"Content-Type": "application/octet-stream"}, timeout=10)
    assert response.status_code == 201, f"Failed to upload blob: {response.status_code}"
    return blob_digest


def create_manifest(blob_digest: str, config_json: str, config_digest: str, data: bytes) -> bytes:
    """Create a manifest for an OCI registry."""

    manifest = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {
            "mediaType": "application/vnd.oci.image.config.v1+json",
            "size": len(config_json),
            "digest": config_digest,
        },
        "layers": [{"mediaType": "application/vnd.oci.image.layer.v1.tar", "size": len(data), "digest": blob_digest}],
    }

    return json.dumps(manifest, separators=(",", ":")).encode("utf-8")


def push_manifest_to_oci_registry(registry_url: str, manifest: bytes, repo: str, tag: str) -> None:
    """Push a manifest to an OCI registry."""
    response = requests.put(
        f"{registry_url}/v2/{repo}/manifests/{tag}",
        data=manifest,
        headers={"Content-Type": "application/vnd.oci.image.manifest.v1+json"},
        timeout=10,
    )
    assert response.status_code == 201, f"Failed to push manifest: {response.status_code}"


def pull_manifest_from_oci_registry(registry_url: str, repo: str, tag: str) -> dict:
    """Pull a manifest from an OCI registry."""
    response = requests.get(
        f"{registry_url}/v2/{repo}/manifests/{tag}",
        headers={"Accept": "application/vnd.oci.image.manifest.v1+json"},
        timeout=10,
    )
    LOGGER.info(f"Manifest pull: {response.status_code}")
    assert response.status_code == 200, f"Failed to pull manifest: {response.status_code}"
    return response.json()
