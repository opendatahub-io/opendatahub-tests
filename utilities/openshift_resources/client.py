"""Async HTTP client for Kubernetes/OpenShift API servers."""

from __future__ import annotations

import asyncio
import json
import os
import random
import ssl
import tempfile
from pathlib import Path
from typing import Any, Self

import aiohttp
import structlog

from utilities.openshift_resources.oc import (
    ConflictError,
    CRDNotInstalledError,
    ForbiddenError,
    OCError,
    ResourceNotFoundError,
    run_oc,
)

logger = structlog.get_logger()

_IN_CLUSTER_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
_IN_CLUSTER_CA_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
_IN_CLUSTER_NAMESPACE_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"


def _k8s_plural(kind: str) -> str:
    """Derive the Kubernetes plural resource name from a kind."""
    lower = kind.lower()
    if lower.endswith(("s", "x", "z", "ch", "sh")):
        return f"{lower}es"
    if lower.endswith("y") and len(lower) > 1 and lower[-2] not in "aeiou":
        return f"{lower[:-1]}ies"
    return f"{lower}s"


def api_path(
    api_group: str,
    api_version: str,
    plural: str,
    name: str | None = None,
    namespace: str | None = None,
) -> str:
    """Build the Kubernetes REST API path for a resource."""
    base = f"/apis/{api_group}/{api_version}" if api_group else f"/api/{api_version}"
    if namespace:
        base = f"{base}/namespaces/{namespace}"
    path = f"{base}/{plural}"
    if name:
        path = f"{path}/{name}"
    return path


def _raise_for_status(method: str, path: str, status: int, body: dict[str, Any]) -> None:
    """Raise the appropriate OCError subclass for a failed API response."""
    message = body.get("message", f"HTTP {status}")
    command = [method, path]
    if status == 404:
        if "ensure CRDs are installed" in message or "no matches for kind" in message:
            raise CRDNotInstalledError(command=command, returncode=status, stderr=message)
        raise ResourceNotFoundError(command=command, returncode=status, stderr=message)
    if status == 403:
        raise ForbiddenError(command=command, returncode=status, stderr=message)
    if status == 409:
        raise ConflictError(command=command, returncode=status, stderr=message)
    raise OCError(command=command, returncode=status, stderr=message)


_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class ApiClient:
    """HTTP client bound to a single user identity on a Kubernetes API server."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        name: str,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
    ) -> None:
        self._session = session
        self.name = name
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff

    async def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Send an HTTP request with retry on transient errors."""
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                logger.debug(event="api_request", method=method, path=path, user=self.name, attempt=attempt)
                async with self._session.request(method, path, **kwargs) as resp:
                    if resp.content_type == "application/json":
                        body = await resp.json()
                    else:
                        text = await resp.text()
                        body = {"message": text} if text else {}

                    if resp.status in _RETRYABLE_STATUSES and attempt < self._max_retries:
                        wait = self._retry_backoff * (2**attempt) + random.uniform(0, 0.1)
                        logger.warning(event="api_retry", status=resp.status, attempt=attempt + 1, wait=f"{wait:.1f}s")
                        await asyncio.sleep(wait)
                        continue

                    if resp.status >= 400:
                        _raise_for_status(method=method, path=path, status=resp.status, body=body)

                    return body

            except (TimeoutError, aiohttp.ClientError) as exc:
                last_error = exc
                if attempt < self._max_retries:
                    wait = self._retry_backoff * (2**attempt) + random.uniform(0, 0.1)
                    logger.warning(event="api_retry_connection", error=str(exc), attempt=attempt + 1)
                    await asyncio.sleep(wait)
                    continue
                raise

        raise last_error  # type: ignore[misc]  # pragma: no cover

    async def get(self, path: str, **params: Any) -> dict[str, Any]:
        return await self.request(method="GET", path=path, params=params or None)

    async def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self.request(method="POST", path=path, json=body)

    async def patch(
        self,
        path: str,
        body: dict[str, Any],
        content_type: str = "application/merge-patch+json",
    ) -> dict[str, Any]:
        return await self.request(
            method="PATCH",
            path=path,
            data=json.dumps(body),
            headers={"Content-Type": content_type},
        )

    async def delete(self, path: str, **params: Any) -> dict[str, Any]:
        return await self.request(method="DELETE", path=path, params=params or None)

    async def server_side_apply(
        self,
        path: str,
        body: dict[str, Any],
        field_manager: str = "openshift-resources",
    ) -> dict[str, Any]:
        return await self.request(
            method="PATCH",
            path=path,
            data=json.dumps(body),
            headers={"Content-Type": "application/apply-patch+yaml"},
            params={"fieldManager": field_manager, "force": "true"},
        )

    async def watch(
        self,
        path: str,
        resource_version: str | None = None,
        timeout: int | None = None,
        field_selector: str | None = None,
        label_selector: str | None = None,
    ):
        """Stream watch events from the Kubernetes API.

        Yields dicts with keys: type (ADDED/MODIFIED/DELETED), object (resource dict).
        """
        params: dict[str, str] = {"watch": "true"}
        if resource_version:
            params["resourceVersion"] = resource_version
        if timeout:
            params["timeoutSeconds"] = str(timeout)
        if field_selector:
            params["fieldSelector"] = field_selector
        if label_selector:
            params["labelSelector"] = label_selector

        request_timeout = aiohttp.ClientTimeout(total=None, sock_read=(timeout or 300) + 10)
        async with self._session.get(path, params=params, timeout=request_timeout) as resp:
            async for line in resp.content:
                line = line.strip()
                if line:
                    yield json.loads(line)


class ClusterSession:
    """Manage connections to a Kubernetes/OpenShift cluster with multiple user identities.

    async with ClusterSession() as cluster:
        await cluster.login("admin", token=admin_token)
        await cluster.login("user1", token=user1_token)

        admin = cluster.client("admin")
        user1 = cluster.client("user1")
    """

    def __init__(
        self,
        server: str | None = None,
        verify_ssl: bool | str = True,
    ) -> None:
        self._server = server
        self._verify_ssl = verify_ssl
        self._clients: dict[str, ApiClient] = {}
        self._sessions: list[aiohttp.ClientSession] = []
        self._default_client: str | None = None

    @staticmethod
    def in_cluster() -> bool:
        """Check if running inside a Kubernetes pod."""
        return Path(_IN_CLUSTER_TOKEN_PATH).exists()

    @staticmethod
    def in_cluster_namespace() -> str:
        """Return the namespace of the pod we're running in."""
        return Path(_IN_CLUSTER_NAMESPACE_PATH).read_text().strip()

    async def login(
        self,
        name: str,
        token: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> ApiClient:
        """Register a user identity and return its client.

        Args:
            name: Identity name (e.g., "admin", "user1").
            token: Bearer token directly.
            username: Log in via oc with username/password.
            password: Required when username is set.

        If neither token nor username is given, reads the token from the current oc context.
        """
        if self._server is None:
            result = await run_oc(args=["whoami", "--show-server"])
            self._server = result.stdout.strip()

        if token is None and username is None:
            result = await run_oc(args=["whoami", "-t"])
            token = result.stdout.strip()
        elif username is not None:
            token = await self._login_with_credentials(username, password)

        ssl_ctx = self._build_ssl_context()
        connector = aiohttp.TCPConnector(ssl=ssl_ctx)

        session = aiohttp.ClientSession(
            base_url=self._server,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
            connector=connector,
        )
        self._sessions.append(session)

        client = ApiClient(session=session, name=name)
        self._clients[name] = client
        if self._default_client is None:
            self._default_client = name

        return client

    async def login_in_cluster(self, name: str = "in-cluster") -> ApiClient:
        """Log in using the pod's service account credentials.

        Reads the token and CA cert from the standard Kubernetes mount paths.
        """
        token_path = Path(_IN_CLUSTER_TOKEN_PATH)
        ca_path = Path(_IN_CLUSTER_CA_PATH)

        if not token_path.exists():
            raise FileNotFoundError(
                f"In-cluster token not found at {_IN_CLUSTER_TOKEN_PATH}. Are you running inside a Kubernetes pod?"
            )

        token = token_path.read_text().strip()

        if self._server is None:
            self._server = "https://kubernetes.default.svc"

        if ca_path.exists() and self._verify_ssl is True:
            self._verify_ssl = str(ca_path)

        return await self.login(name=name, token=token)

    async def _login_with_credentials(self, username: str, password: str | None) -> str:
        """Log in via oc using a temporary kubeconfig to avoid changing the current context."""
        fd, tmp_kubeconfig = tempfile.mkstemp(suffix=".kubeconfig")
        os.close(fd)
        try:
            assert self._server is not None
            login_args = ["login", self._server, "-u", username]
            if password:
                login_args.extend(["-p", password])
            if self._verify_ssl is False:
                login_args.append("--insecure-skip-tls-verify")
            login_args.extend(["--kubeconfig", tmp_kubeconfig])
            await run_oc(args=login_args)

            result = await run_oc(args=["whoami", "-t", "--kubeconfig", tmp_kubeconfig])
            return result.stdout.strip()
        finally:
            os.unlink(tmp_kubeconfig)

    def client(self, name: str | None = None) -> ApiClient:
        """Get a client by identity name. Returns the first registered if name is None."""
        key = name or self._default_client
        if key is None:
            raise ValueError("No client registered. Call login() first.")
        if key not in self._clients:
            raise KeyError(f"Unknown client: {key!r}. Registered: {list(self._clients)}")
        return self._clients[key]

    def _build_ssl_context(self) -> ssl.SSLContext | bool:
        if isinstance(self._verify_ssl, str):
            return ssl.create_default_context(cafile=self._verify_ssl)
        return self._verify_ssl

    async def close(self) -> None:
        for session in self._sessions:
            await session.close()
        self._sessions.clear()
        self._clients.clear()
        self._default_client = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
