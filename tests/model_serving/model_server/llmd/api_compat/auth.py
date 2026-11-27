from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.service_account import ServiceAccount

from utilities.infra import create_inference_token

if TYPE_CHECKING:
    from types import TracebackType

    from kubernetes.dynamic import DynamicClient

    from utilities.resources.llm_inference_service import LLMInferenceService


@runtime_checkable
class APIKeyProvider(Protocol):
    def get_api_key(self) -> str: ...


class BearerTokenProvider:
    def __init__(self, token: str) -> None:
        self._token = token

    def get_api_key(self) -> str:
        return self._token


class ServiceAccountTokenProvider:
    """Creates SA + Role + RoleBinding for an LLMInferenceService and provides an auth token.

    Use as a context manager to ensure cleanup of Kubernetes resources:

        with ServiceAccountTokenProvider(client=client, llmisvc=llmisvc) as provider:
            validator = OpenAICompatibilityValidator(
                base_url=url,
                model_name=model,
                api_key_provider=provider,
            )
            validator.run_all()
    """

    def __init__(
        self,
        client: DynamicClient,
        llmisvc: LLMInferenceService,
        sa_name: str | None = None,
    ) -> None:
        self._client = client
        self._llmisvc = llmisvc
        self._sa_name = sa_name or f"{llmisvc.name}-compat-sa"
        self._stack: ExitStack | None = None
        self._token: str | None = None

    def __enter__(self) -> Self:
        self._stack = ExitStack()
        sa = self._stack.enter_context(  # noqa: FCN001
            ServiceAccount(
                client=self._client,
                namespace=self._llmisvc.namespace,
                name=self._sa_name,
            ),
        )
        role = self._stack.enter_context(  # noqa: FCN001
            Role(
                client=self._client,
                name=f"{self._llmisvc.name}-compat-view",
                namespace=self._llmisvc.namespace,
                rules=[
                    {
                        "apiGroups": [self._llmisvc.api_group],
                        "resources": ["llminferenceservices"],
                        "verbs": ["get"],
                        "resourceNames": [self._llmisvc.name],
                    },
                ],
            ),
        )
        self._stack.enter_context(  # noqa: FCN001
            RoleBinding(
                client=self._client,
                namespace=self._llmisvc.namespace,
                name=f"{self._sa_name}-compat-view",
                role_ref_name=role.name,
                role_ref_kind=role.kind,
                subjects_kind="ServiceAccount",
                subjects_name=self._sa_name,
            ),
        )
        self._token = create_inference_token(model_service_account=sa)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._stack:
            self._stack.__exit__(exc_type, exc_val, exc_tb)  # noqa: FCN001
            self._stack = None
        self._token = None

    def get_api_key(self) -> str:
        if self._token is None:
            raise RuntimeError("ServiceAccountTokenProvider must be used as a context manager")
        return self._token


class NoAuthProvider:
    def get_api_key(self) -> str:
        return "unused"
