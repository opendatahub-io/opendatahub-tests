"""ConnectionsAPI helpers for the InferenceService storage tests (S3/OCI/URI)."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.secret import Secret
from ocp_resources.service import Service

from tests.model_serving.model_server.utils import (
    CONNECTION_TYPE_PROTOCOL_ANNOTATION,
    assert_service_account_exists,
    wait_for_cleared_predicate,
)
from utilities.constants import ApiGroups, Labels, MinIo, Protocols, Timeout
from utilities.general import get_s3_secret_dict


@contextmanager
def create_minio_connection_secret(
    minio_service: Service,
    model_namespace: str,
    aws_s3_bucket: str,
    client: DynamicClient,
) -> Generator[Secret, Any, Any]:
    """Create the in-cluster MinIO data-connection Secret, additionally typed for ConnectionsAPI.

    Mirrors `utilities.minio.create_minio_data_connection_secret` (same name, credentials, and
    `opendatahub.io/connection-type: s3` annotation for the `secret_controller`'s `storage-config`
    auto-creation) but also sets `opendatahub.io/connection-type-protocol: s3`, which
    `odh-model-controller`'s ConnectionsAPI webhook requires to accept `opendatahub.io/connections`
    references to this Secret (`ValidateConnectionType` rejects the admission otherwise). Kept as a
    local copy rather than editing the shared helper, to avoid changing behavior for its other
    (non-ConnectionsAPI) caller.

    Args:
        minio_service: The in-cluster MinIO Service (used to compute the S3 endpoint).
        model_namespace: Namespace to create the Secret in.
        aws_s3_bucket: MinIO bucket name.
        client: Kubernetes dynamic client.

    Yields:
        Secret: The created, ConnectionsAPI-typed MinIO data-connection Secret.
    """
    data_dict = get_s3_secret_dict(
        aws_access_key=MinIo.Credentials.ACCESS_KEY_VALUE,
        aws_secret_access_key=MinIo.Credentials.SECRET_KEY_VALUE,  # pragma: allowlist secret
        aws_s3_bucket=aws_s3_bucket,
        aws_s3_endpoint=f"{Protocols.HTTP}://{minio_service.instance.spec.clusterIP}:{MinIo.Metadata.DEFAULT_PORT!s}",
        aws_s3_region="us-south",
    )
    with Secret(
        client=client,
        name="aws-connection-minio-data-connection",
        namespace=model_namespace,
        data_dict=data_dict,
        label={
            Labels.OpenDataHub.DASHBOARD: "true",
            Labels.OpenDataHubIo.MANAGED: "true",
        },
        annotations={
            f"{ApiGroups.OPENDATAHUB_IO}/connection-type": "s3",
            CONNECTION_TYPE_PROTOCOL_ANNOTATION: "s3",
            "openshift.io/display-name": "Minio Data Connection",
        },
    ) as minio_secret:
        yield minio_secret


# ---------------------------------------------------------------------------
# InferenceService injection assertions
# ---------------------------------------------------------------------------
def assert_isvc_s3_injected(isvc: InferenceService, secret_name: str, expected_path: str) -> None:
    """Assert an S3 connection was injected into an InferenceService's predictor spec.

    Args:
        isvc: InferenceService to inspect (re-read via `.instance`).
        secret_name: Name of the S3 connection Secret that should have been injected.
        expected_path: Expected `storage.path` value (from `opendatahub.io/connection-path`).

    Raises:
        AssertionError: If the SA name, storage key, or storage path do not match.
    """
    predictor = isvc.instance.spec.predictor
    sa_name = predictor.get("serviceAccountName")
    assert sa_name == f"{secret_name}-sa", f"Expected predictor.serviceAccountName={secret_name}-sa, got {sa_name!r}"

    storage = predictor.model.get("storage") or {}
    assert storage.get("key") == secret_name, f"Expected predictor.model.storage.key={secret_name!r}, got {storage!r}"
    assert storage.get("path") == expected_path, (
        f"Expected predictor.model.storage.path={expected_path!r}, got {storage!r}"
    )


def assert_isvc_s3_fully_injected(
    client: DynamicClient, isvc: InferenceService, namespace: str, secret_name: str, expected_path: str
) -> None:
    """Assert both effects of S3 injection on an InferenceService: spec fields and SA creation.

    S3 injection is always expected to produce both effects together (the webhook sets the
    predictor spec fields *and* creates the `{secret}-sa` ServiceAccount as a side effect), so
    every S3 CREATE/UPDATE-inject test needs both checks. Combines `assert_isvc_s3_injected` and
    `assert_service_account_exists` so callers only need one call.

    Args:
        client: Kubernetes dynamic client.
        isvc: InferenceService to inspect (re-read via `.instance`).
        namespace: Namespace the `{secret}-sa` ServiceAccount is expected in.
        secret_name: Name of the S3 connection Secret that should have been injected.
        expected_path: Expected `storage.path` value (from `opendatahub.io/connection-path`).

    Raises:
        AssertionError: If the spec fields don't match, or the ServiceAccount doesn't exist.
    """
    assert_isvc_s3_injected(isvc=isvc, secret_name=secret_name, expected_path=expected_path)
    assert_service_account_exists(client=client, namespace=namespace, name=f"{secret_name}-sa")


def assert_isvc_uri_injected(isvc: InferenceService, expected_uri: str) -> None:
    """Assert a `uri` connection was injected as `predictor.model.storageUri`.

    Args:
        isvc: InferenceService to inspect (re-read via `.instance`).
        expected_uri: Expected `storageUri` value (the connection Secret's `URI` key).

    Raises:
        AssertionError: If `storageUri` does not match.
    """
    storage_uri = isvc.instance.spec.predictor.model.get("storageUri")
    assert storage_uri == expected_uri, f"Expected predictor.model.storageUri={expected_uri!r}, got {storage_uri!r}"


def assert_isvc_oci_injected(isvc: InferenceService, secret_name: str) -> None:
    """Assert an `oci` connection was injected as a `predictor.imagePullSecrets` entry.

    Args:
        isvc: InferenceService to inspect (re-read via `.instance`).
        secret_name: Name of the OCI connection Secret expected in `imagePullSecrets`.

    Raises:
        AssertionError: If the secret name is not present in `imagePullSecrets`.
    """
    pull_secrets = isvc.instance.spec.predictor.get("imagePullSecrets") or []
    names = [dict(entry).get("name") for entry in pull_secrets]
    assert secret_name in names, f"Expected {secret_name!r} in predictor.imagePullSecrets, got {names}"


def assert_isvc_connection_cleared(isvc: InferenceService) -> None:
    """Assert an ISVC's S3 injection fields were cleared by an UPDATE-remove action.

    Args:
        isvc: InferenceService to inspect (re-read via `.instance`).

    Raises:
        AssertionError: If `serviceAccountName` or `model.storage` are still populated.
    """
    predictor = isvc.instance.spec.predictor
    sa_name = predictor.get("serviceAccountName")
    assert not sa_name, f"Expected predictor.serviceAccountName to be cleared, got {sa_name!r}"

    storage = predictor.model.get("storage")
    assert not storage, f"Expected predictor.model.storage to be cleared, got {storage!r}"


def wait_for_isvc_connection_cleared(isvc: InferenceService, timeout: int = Timeout.TIMEOUT_2MIN) -> None:
    """Poll until an ISVC's connection fields are cleared after an UPDATE-remove action.

    The webhook's cleanup runs asynchronously relative to the annotation patch, so this polls
    instead of asserting immediately after `.update()`.

    Args:
        isvc: InferenceService to poll (re-read via `.instance` on every sample).
        timeout: Seconds to wait before giving up.

    Raises:
        TimeoutError: If the fields are not cleared within `timeout` seconds.
    """

    def _cleared() -> bool:
        predictor = isvc.instance.spec.predictor
        return not predictor.get("serviceAccountName") and not predictor.model.get("storage")

    wait_for_cleared_predicate(predicate=_cleared, timeout=timeout, resource_label=f"InferenceService {isvc.name}")
