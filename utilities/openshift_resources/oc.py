"""Async wrapper around the oc binary."""

from __future__ import annotations

import asyncio
import json
import random
import shutil
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger()


def _jitter() -> float:
    return random.uniform(a=0, b=0.1)


_SENSITIVE_FLAGS: set[str] = {"--token", "-p"}


def _redact_command(command: list[str]) -> str:
    """Redact sensitive values (tokens, passwords) from a command list."""
    redacted = []
    skip_next = False
    for arg in command:
        if skip_next:
            redacted.append("********")
            skip_next = False
        elif arg in _SENSITIVE_FLAGS:
            redacted.append(arg)
            skip_next = True
        else:
            redacted.append(arg)
    return " ".join(redacted)


class OCError(Exception):
    """Raised when an oc command fails."""

    def __init__(self, command: list[str], returncode: int, stderr: str) -> None:
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"oc failed (rc={returncode}): {_redact_command(command)}\n{stderr}")


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


@dataclass
class OCResult:
    """Result of an oc command."""

    stdout: str
    stderr: str
    returncode: int


def _classify_error(stderr: str) -> type[OCError]:
    if "ensure CRDs are installed first" in stderr or "no matches for kind" in stderr:
        return CRDNotInstalledError
    if "NotFound" in stderr or "not found" in stderr:
        return ResourceNotFoundError
    if "Forbidden" in stderr:
        return ForbiddenError
    if "Conflict" in stderr or "the object has been modified" in stderr:
        return ConflictError
    return OCError


_RETRYABLE_PATTERNS = {
    "etcd leader changed",
    "etcd cluster is unavailable",
    "context deadline exceeded",
    "connection refused",
    "connection reset by peer",
    "tls handshake timeout",
    "i/o timeout",
    "read: connection reset",
    "net/http: request canceled",
}


def _is_retryable(stderr: str) -> bool:
    lower = stderr.lower()
    return any(pattern in lower for pattern in _RETRYABLE_PATTERNS)


async def run_oc(
    args: list[str],
    timeout: int = 120,
    check: bool = True,
    input: str | None = None,
    retries: int = 3,
    retry_backoff: float = 0.5,
) -> OCResult:
    """Run an oc command asynchronously with retry on transient errors."""
    command = [_get_oc_binary(), *args]

    for attempt in range(retries + 1):
        logger.debug(event="run_oc", command=_redact_command(command), attempt=attempt)
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE if input else None,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(  # noqa: FCN001
                proc.communicate(input=input.encode() if input else None),
                timeout=timeout,
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise TimeoutError(f"oc command timed out after {timeout}s: {_redact_command(command)}")

        stdout = stdout_bytes.decode() if stdout_bytes else ""
        stderr = (stderr_bytes.decode() if stderr_bytes else "").strip()

        if check and proc.returncode != 0:
            if attempt < retries and _is_retryable(stderr):
                wait = retry_backoff * (2**attempt) + _jitter()
                logger.warning(event="run_oc_retry", stderr=stderr, attempt=attempt + 1, wait=f"{wait:.1f}s")
                await asyncio.sleep(wait)
                continue
            error_cls = _classify_error(stderr=stderr)
            raise error_cls(command=command, returncode=proc.returncode or 1, stderr=stderr)

        return OCResult(stdout=stdout, stderr=stderr, returncode=proc.returncode or 0)

    raise OCError(command=command, returncode=1, stderr="max retries exceeded")


async def oc_login(
    server: str,
    token: str | None = None,
    username: str | None = None,
    password: str | None = None,
    insecure_skip_tls_verify: bool = False,
) -> str:
    """Log in to an OpenShift cluster."""
    login_args = ["login", server]
    if token:
        login_args.extend(["--token", token])
    elif username:
        login_args.extend(["-u", username])
        if password:
            login_args.extend(["-p", password])
    if insecure_skip_tls_verify:
        login_args.append("--insecure-skip-tls-verify")
    result = await run_oc(args=login_args)
    return result.stdout.strip()


async def oc_whoami(show_token: bool = False, show_server: bool = False) -> str:
    """Return the currently logged-in user, token, or server URL."""
    whoami_args = ["whoami"]
    if show_token:
        whoami_args.append("-t")
    elif show_server:
        whoami_args.append("--show-server")
    result = await run_oc(args=whoami_args)
    return result.stdout.strip()


async def oc_logout() -> str:
    """Log out of the current session."""
    result = await run_oc(args=["logout"])
    return result.stdout.strip()


async def oc_config(subcommand: str, *extra_args: str) -> str:
    """Run an oc config subcommand (view, use-context, get-contexts, etc.)."""
    result = await run_oc(args=["config", subcommand, *extra_args])
    return result.stdout.strip()


async def oc_get_json(
    kind: str,
    name: str | None = None,
    namespace: str | None = None,
    all_namespaces: bool = False,
    label_selector: str | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Run oc get with JSON output and return parsed result."""
    get_args = ["get", kind]
    if name:
        get_args.append(name)
    if namespace:
        get_args.extend(["-n", namespace])
    elif all_namespaces:
        get_args.append("--all-namespaces")
    if label_selector:
        get_args.extend(["-l", label_selector])
    get_args.extend(["-o", "json"])

    result = await run_oc(args=get_args)
    data = json.loads(result.stdout)

    if data.get("kind") == "List":
        return data.get("items", [])
    return data
