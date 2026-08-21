"""Utilities for vLLM-Omni test validation and measurement."""

import re
import time
from typing import Any

import requests
import structlog
import urllib3
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from utilities.inference_utils import get_exposed_isvc_url

urllib3.disable_warnings(category=urllib3.exceptions.InsecureRequestWarning)

LOGGER = structlog.get_logger(name=__name__)


def log_loop_progress(
    logger: structlog.stdlib.BoundLogger,
    event: str,
    current: int,
    total: int,
    is_error: bool = False,
    every: int = 10,
    **extra: Any,
) -> None:
    """Log loop iteration progress at INFO every ``every`` steps, else DEBUG.

    Always logs at INFO for the final iteration and on errors.
    """
    if current % every == 0 or current == total or is_error:
        logger.info(event=event, turn=current, total=total, **extra)
    else:
        logger.debug(event=event, turn=current, total=total, **extra)


# Audio format magic byte signatures
_WAV_MAGIC: bytes = b"RIFF"
_FLAC_MAGIC: bytes = b"fLaC"
_PNG_MAGIC: bytes = b"\x89PNG"
_JPEG_MAGIC: bytes = b"\xff\xd8\xff"

# WAV header minimum size (44 bytes)
_WAV_HEADER_MIN_BYTES: int = 44

_FORMAT_CONTENT_TYPES: dict[str, set[str]] = {
    "wav": {"audio/wav", "audio/x-wav"},
    "mp3": {"audio/mpeg", "audio/mp3"},
    "flac": {"audio/flac"},
    "pcm": {"audio/pcm", "audio/x-pcm"},
    "opus": {"audio/ogg", "audio/opus"},
}

# Pre-compiled pattern for Prometheus "# TYPE <name> <kind>" header lines
_TYPE_LINE_PATTERN: re.Pattern[str] = re.compile(r"^# TYPE (\S+)")


def _fetch_raw_metrics(pod: Pod) -> str:
    """Curl http://localhost:<METRICS_PORT>/metrics inside the pod and return the body."""
    from tests.model_serving.model_runtime.vllm_omni.constant import METRICS_PATH, METRICS_PORT
    from utilities.constants import Containers

    cmd: list[str] = [
        "curl",
        "-s",
        f"http://localhost:{METRICS_PORT}{METRICS_PATH}",
        "--max-time",
        "30",
    ]
    return pod.execute(command=cmd, container=Containers.KSERVE_CONTAINER_NAME)


def _extract_metric_families(metrics_text: str, prefix: str) -> set[str]:
    """Return the set of distinct metric family names that start with *prefix*."""
    families: set[str] = set()
    for line in metrics_text.splitlines():
        match = _TYPE_LINE_PATTERN.match(string=line)
        if match:
            metric_name = match.group(1)
            if metric_name.startswith(prefix):
                families.add(metric_name)
    return families


def validate_tts_output(response: requests.Response, response_format: str = "wav") -> None:
    """Validate a /v1/audio/speech TTS response.

    Checks HTTP status, Content-Type, minimum body size, and magic bytes.

    Args:
        response: HTTP response from the TTS endpoint.
        response_format: Expected audio format (wav, mp3, flac, pcm, opus).

    Raises:
        AssertionError: If any validation check fails.
    """
    assert response.status_code == 200, (
        f"/v1/audio/speech returned HTTP {response.status_code}; expected 200. "
        f"Body (first 500 chars): {response.text[:500]}"
    )
    content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
    expected_types = _FORMAT_CONTENT_TYPES.get(response_format, {"audio/wav"})
    assert content_type in expected_types, (
        f"Content-Type {content_type!r} not in expected set {expected_types} for response_format={response_format!r}"
    )
    body = response.content
    assert len(body) > _WAV_HEADER_MIN_BYTES, (
        f"Audio body {len(body)} bytes does not exceed the minimum "
        f"{_WAV_HEADER_MIN_BYTES}-byte threshold (WAV header size)"
    )
    if response_format == "wav":
        assert body[:4] == _WAV_MAGIC, (
            f"Audio body does not begin with expected magic bytes {_WAV_MAGIC!r} "
            f"for format {response_format!r}; got {body[:4]!r}"
        )
    elif response_format == "flac":
        assert body[:4] == _FLAC_MAGIC, (
            f"Audio body does not begin with expected magic bytes {_FLAC_MAGIC!r} "
            f"for format {response_format!r}; got {body[:4]!r}"
        )
    elif response_format == "mp3":
        is_mpeg_sync = len(body) >= 2 and body[0] == 0xFF and (body[1] & 0xE0) == 0xE0
        assert is_mpeg_sync or body[:3] == b"ID3", (
            f"Audio body does not begin with expected magic bytes for MP3; got {body[:4]!r}"
        )


def timed_tts_request(
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 120,
    *,
    model: str | None = None,
    prompt: str | None = None,
    voice: str | None = None,
) -> tuple[int, float, float]:
    """Send a TTS request to /v1/audio/speech and measure TTFB and total latency.

    Supports two calling conventions:
      - Legacy: timed_tts_request(url, payload)
      - Keyword: timed_tts_request(url=..., model=..., prompt=..., voice=...)

    Args:
        url: Base URL of the InferenceService (from get_exposed_isvc_url).
            /v1/audio/speech is appended automatically.
        payload: Request body dict (used directly if provided).
        headers: Optional HTTP headers (defaults to JSON content-type).
        timeout: Request timeout in seconds.
        model: Model name (builds payload when payload is None).
        prompt: Input text for TTS (builds payload when payload is None).
        voice: Voice name for TTS (builds payload when payload is None).
            Pass None to omit the voice field (required for models without
            built-in voice presets, e.g. OmniVoice).

    Returns:
        Tuple of (status_code, ttfb_seconds, e2e_seconds).
    """
    if payload is None:
        payload = {
            "model": model or "",
            "input": prompt or "",
            "response_format": "wav",
        }
        if voice is not None:
            payload["voice"] = voice

    tts_url = f"{url.rstrip('/')}/v1/audio/speech"
    headers = headers or {"Content-Type": "application/json"}
    start = time.monotonic()
    first_byte_time: float | None = None

    with requests.post(tts_url, json=payload, headers=headers, stream=True, timeout=timeout, verify=False) as resp:
        for chunk in resp.iter_content(chunk_size=4096):
            if chunk and first_byte_time is None:
                first_byte_time = time.monotonic()
        end = time.monotonic()

    ttfb = (first_byte_time if first_byte_time is not None else end) - start
    total = end - start
    return resp.status_code, ttfb, total


def calculate_percentile(values: list[float], percentile: float) -> float:
    """Calculate the given percentile from a sorted list of floats.

    Args:
        values: Non-empty list of float measurements.
        percentile: Percentile value in [0, 100] (e.g. 95.0 for p95).

    Returns:
        Interpolated percentile value.

    Raises:
        ValueError: If values is empty or percentile is out of range.
    """
    if not values:
        raise ValueError("Cannot calculate percentile of empty list")
    if not 0 <= percentile <= 100:
        raise ValueError(f"Percentile must be 0-100, got {percentile}")
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * percentile / 100
    floor_k = int(k)
    ceil_k = min(floor_k + 1, len(sorted_vals) - 1)
    return sorted_vals[floor_k] + (sorted_vals[ceil_k] - sorted_vals[floor_k]) * (k - floor_k)


_voice_cache: dict[str, str] = {}


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(min=1, max=6),
    retry=retry_if_exception_type(requests.RequestException),
    reraise=True,
)
def run_tts_inference(
    isvc: InferenceService,
    text: str,
    voice: str | None = None,
    response_format: str = "wav",
    timeout: int = 120,
) -> requests.Response:
    """Send a TTS request to /v1/audio/speech, deriving model name from the ISVC.

    Mirrors vLLM's run_raw_inference pattern — model name = isvc.instance.metadata.name.
    Voice is auto-discovered on first call and cached per URL.
    Response is validated via validate_tts_output before returning.
    """
    url = get_exposed_isvc_url(isvc=isvc)
    if voice is None:
        voice = _voice_cache.get(url) or discover_tts_voice(isvc=isvc)
        _voice_cache[url] = voice
    model_name = isvc.instance.metadata.name
    response = requests.post(
        f"{url}/v1/audio/speech",
        json={
            "model": model_name,
            "input": text,
            "voice": voice,
            "response_format": response_format,
        },
        verify=False,
        timeout=timeout,
    )
    validate_tts_output(response=response, response_format=response_format)
    return response


def discover_tts_voice(isvc: InferenceService, preferred: str = "vivian") -> str:
    """Discover available TTS voices from the /v1/audio/voices endpoint."""
    url = get_exposed_isvc_url(isvc=isvc)
    resp = requests.get(f"{url}/v1/audio/voices", verify=False, timeout=30)
    if resp.status_code != 200:
        LOGGER.warning(event="voice discovery failed", status=resp.status_code)
        return preferred
    payload = resp.json()
    voices = payload.get("voices", payload) if isinstance(payload, dict) else payload
    names = [v.get("name", str(v)) if isinstance(v, dict) else str(v) for v in voices]
    if not names:
        return preferred
    return preferred if preferred in names else names[0]
