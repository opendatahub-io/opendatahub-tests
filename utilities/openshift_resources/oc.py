"""Low-level wrapper around the oc binary."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

import structlog

logger = structlog.get_logger()


class OCError(Exception):
    """Raised when an oc command fails."""

    def __init__(self, command: list[str], returncode: int, stderr: str) -> None:
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        cmd_str = " ".join(command)
        super().__init__(f"oc failed (rc={returncode}): {cmd_str}\n{stderr}")


class CRDNotInstalledError(OCError):
    """Raised when a resource kind is not installed on the cluster."""


class ResourceNotFoundError(OCError):
    """Raised when a resource does not exist."""


class ForbiddenError(OCError):
    """Raised when the user lacks permission."""


class ConflictError(OCError):
    """Raised when a resource update conflicts with a newer version."""


_OC_BINARY: str | None = None


def _get_oc_binary() -> str:
    """Find the oc binary on PATH, caching the result."""
    global _OC_BINARY
    if _OC_BINARY is None:
        _OC_BINARY = shutil.which(cmd="oc")
        if not _OC_BINARY:
            raise FileNotFoundError("oc binary not found on PATH")
    return _OC_BINARY


def run_oc(
    *args: str,
    capture_output: bool = True,
    timeout: int = 120,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run an oc command and return the result."""
    command = [_get_oc_binary(), *args]
    logger.debug(event="run_oc", command=" ".join(command))
    result = subprocess.run(
        command,
        capture_output=capture_output,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        stderr = result.stderr.strip()
        error_cls = OCError
        if "ensure CRDs are installed first" in stderr or "no matches for kind" in stderr:
            error_cls = CRDNotInstalledError
        elif "NotFound" in stderr or "not found" in stderr:
            error_cls = ResourceNotFoundError
        elif "Forbidden" in stderr:
            error_cls = ForbiddenError
        elif "Conflict" in stderr or "the object has been modified" in stderr:
            error_cls = ConflictError
        raise error_cls(command=command, returncode=result.returncode, stderr=stderr)
    return result


def oc_login(
    server: str,
    token: str | None = None,
    username: str | None = None,
    password: str | None = None,
    insecure_skip_tls_verify: bool = False,
) -> str:
    """Log in to an OpenShift cluster.

    Args:
        server: API server URL (e.g., https://api.example.com:6443).
        token: Bearer token for authentication.
        username: Username for basic auth.
        password: Password for basic auth.
        insecure_skip_tls_verify: Skip TLS certificate verification.

    Returns:
        The oc login stdout output.
    """
    args = ["login", server]
    if token:
        args.extend(["--token", token])
    elif username:
        args.extend(["-u", username])
        if password:
            args.extend(["-p", password])
    if insecure_skip_tls_verify:
        args.append("--insecure-skip-tls-verify")
    return run_oc(*args).stdout.strip()


def oc_whoami(show_token: bool = False, show_server: bool = False) -> str:
    """Return the currently logged-in user, token, or server URL."""
    args = ["whoami"]
    if show_token:
        args.append("-t")
    elif show_server:
        args.append("--show-server")
    return run_oc(*args).stdout.strip()


def oc_logout() -> str:
    """Log out of the current session."""
    return run_oc("logout").stdout.strip()


def oc_config(subcommand: str, *args: str) -> str:
    """Run an oc config subcommand (view, use-context, get-contexts, etc.)."""
    return run_oc("config", subcommand, *args).stdout.strip()


def oc_get_json(
    kind: str,
    name: str | None = None,
    namespace: str | None = None,
    label_selector: str | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Run oc get with JSON output and return parsed result."""
    args = ["get", kind]
    if name:
        args.append(name)
    if namespace:
        args.extend(["-n", namespace])
    if label_selector:
        args.extend(["-l", label_selector])
    args.extend(["-o", "json"])

    result = run_oc(*args)
    data = json.loads(result.stdout)

    if data.get("kind") == "List":
        return data.get("items", [])
    return data
