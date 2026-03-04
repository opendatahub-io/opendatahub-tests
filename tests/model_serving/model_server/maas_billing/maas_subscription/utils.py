from __future__ import annotations

import json
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlparse

import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.resource import ResourceEditor
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from utilities.constants import (
    MAAS_GATEWAY_NAME,
    MAAS_GATEWAY_NAMESPACE,
    ApiGroups,
)
from utilities.resources.maa_s_subscription import MaaSSubscription

LOGGER = get_logger(name=__name__)


@contextmanager
def patch_llmisvc_with_maas_router_and_tiers(
    llm_service: LLMInferenceService,
    tiers: Sequence[str],
    enable_auth: bool = True,
) -> Generator[None]:
    """
    Patch an LLMInferenceService to use MaaS router (gateway refs + route {})
    and set MaaS tier annotation.

    This is intended for MaaS subscription tests where you want distinct
    tiered models (e.g. free vs premium)

    Examples:
      - tiers=[]              -> open model
      - tiers=["premium"]     -> premium-only
    """
    router_spec = {
        "gateway": {"refs": [{"name": MAAS_GATEWAY_NAME, "namespace": MAAS_GATEWAY_NAMESPACE}]},
        "route": {},
    }

    tiers_val = list(tiers)
    patch_body = {
        "metadata": {
            "annotations": {
                f"alpha.{ApiGroups.MAAS_IO}/tiers": json.dumps(tiers_val),
                "security.opendatahub.io/enable-auth": "true" if enable_auth else "false",
            }
        },
        "spec": {"router": router_spec},
    }

    with ResourceEditor(patches={llm_service: patch_body}):
        yield


def model_id_from_chat_completions_url(model_url: str) -> str:
    path = urlparse(model_url).path.strip("/")
    parts = path.split("/")

    if len(parts) >= 2 and parts[0] == "llm":
        model_id = parts[1]
        if model_id:
            return model_id

    raise AssertionError(f"Cannot extract model id from url: {model_url!r} (path={path!r})")


def chat_payload_for_url(model_url: str, *, prompt: str = "Hello", max_tokens: int = 8) -> dict:
    model_id = model_id_from_chat_completions_url(model_url=model_url)
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }


def poll_expected_status(
    *,
    request_session_http: requests.Session,
    model_url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    expected_statuses: set[int],
    wait_timeout: int = 240,
    sleep: int = 5,
    request_timeout: int = 60,
) -> requests.Response:
    """
    Poll model endpoint until we see one of `expected_statuses` or timeout.

    Returns the response that matched expected status.
    """
    last_response: requests.Response | None = None
    observed_responses: list[tuple[int | None, str]] = []

    for response in TimeoutSampler(
        wait_timeout=wait_timeout,
        sleep=sleep,
        func=request_session_http.post,
        url=model_url,
        headers=headers,
        json=payload,
        timeout=request_timeout,
    ):
        last_response = response
        status_code = getattr(response, "status_code", None)
        response_text = (getattr(response, "text", "") or "")[:200]

        observed_responses.append((status_code, response_text))

        LOGGER.info(
            "Polling model_url=%s status=%s expected=%s",
            model_url,
            status_code,
            sorted(expected_statuses),
        )

        if status_code in expected_statuses:
            return response

    pytest.fail(
        "Timed out waiting for expected HTTP status. "
        f"model_url={model_url}, "
        f"expected={sorted(expected_statuses)}, "
        f"last_status={getattr(last_response, 'status_code', None)}, "
        f"last_body={(getattr(last_response, 'text', '') or '')[:200]}, "
        f"seen_count={len(observed_responses)}"
    )


def create_maas_subscription(
    *,
    admin_client: DynamicClient,
    subscription_name: str,
    owner_group_name: str,
    model_name: str,
    tokens_per_minute: int,
    window: str = "1m",
    priority: int = 0,
    teardown: bool = True,
    wait_for_resource: bool = True,
) -> MaaSSubscription:
    applications_namespace = py_config["applications_namespace"]

    return MaaSSubscription(
        client=admin_client,
        name=subscription_name,
        namespace=applications_namespace,
        owner={
            "groups": [{"name": owner_group_name}],
        },
        model_refs=[
            {
                "name": model_name,
                "tokenRateLimits": [{"limit": tokens_per_minute, "window": window}],
            }
        ],
        priority=priority,
        teardown=teardown,
        wait_for_resource=wait_for_resource,
    )
