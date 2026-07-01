import base64
from types import SimpleNamespace

import pytest
from _pytest.config.argparsing import Parser

from tests.model_serving.model_runtime.vllm.modelcar.constant import (
    MODELCAR_REGISTRIES,
    QUAY_IO_REGISTRY_HOST,
    REGISTRY_REDHAT_IO_HOST,
    REGISTRY_STAGE_REDHAT_IO_HOST,
)
from tests.model_serving.model_runtime.vllm.modelcar.pytest_options import (
    register_modelcar_registry_pull_secret_options,
)
from tests.model_serving.model_runtime.vllm.modelcar.utils import collect_modelcar_registry_credentials

VALID_AUTH = base64.b64encode(b"user:pass").decode()


def _pytestconfig_with_secrets(**secrets: str | None) -> SimpleNamespace:
    option_values = {registry.option_dest: secrets.get(registry.host) for registry in MODELCAR_REGISTRIES}
    return SimpleNamespace(option=SimpleNamespace(**option_values))


def _parse_registry_options(args: list[str]) -> SimpleNamespace:
    parser = Parser(prog="pytest", usage="pytest [options]")
    register_modelcar_registry_pull_secret_options(parser=parser)
    return parser.parse(args=args)


class TestCollectModelcarRegistryCredentials:
    """Unit tests for modelcar registry credential collection."""

    def test_collects_single_registry_secret(self) -> None:
        """Given one configured registry pull secret, return matching host and secret."""
        pytestconfig = _pytestconfig_with_secrets(**{QUAY_IO_REGISTRY_HOST: VALID_AUTH})

        hosts, secrets = collect_modelcar_registry_credentials(pytestconfig=pytestconfig)  # type: ignore[arg-type]

        assert hosts == [QUAY_IO_REGISTRY_HOST]
        assert secrets == [VALID_AUTH]

    def test_collects_multiple_registry_secrets_in_definition_order(self) -> None:
        """Given multiple configured secrets, return hosts in registry definition order."""
        stage_auth = base64.b64encode(b"stage-user:stage-pass").decode()
        redhat_auth = base64.b64encode(b"redhat-user:redhat-pass").decode()
        pytestconfig = _pytestconfig_with_secrets(**{
            REGISTRY_REDHAT_IO_HOST: redhat_auth,
            QUAY_IO_REGISTRY_HOST: VALID_AUTH,
            REGISTRY_STAGE_REDHAT_IO_HOST: stage_auth,
        })

        hosts, secrets = collect_modelcar_registry_credentials(pytestconfig=pytestconfig)  # type: ignore[arg-type]

        assert hosts == [QUAY_IO_REGISTRY_HOST, REGISTRY_STAGE_REDHAT_IO_HOST, REGISTRY_REDHAT_IO_HOST]
        assert secrets == [VALID_AUTH, stage_auth, redhat_auth]

    def test_returns_empty_lists_when_no_registry_secret_configured(self) -> None:
        """Given no configured pull secrets and required=False, return empty lists."""
        pytestconfig = _pytestconfig_with_secrets()

        hosts, secrets = collect_modelcar_registry_credentials(pytestconfig=pytestconfig)  # type: ignore[arg-type]

        assert hosts == []
        assert secrets == []

    def test_raises_when_no_registry_secret_configured_and_required(self) -> None:
        """Given no configured pull secrets and required=True, raise a helpful ValueError."""
        pytestconfig = _pytestconfig_with_secrets()

        with pytest.raises(ValueError, match="No modelcar registry pull secret is configured"):
            collect_modelcar_registry_credentials(pytestconfig=pytestconfig, required=True)  # type: ignore[arg-type]


class TestModelcarRegistryPytestOptions:
    """Verify pytest CLI options read modelcar registry env vars."""

    @pytest.mark.parametrize(
        argnames=("env_var", "cli_option", "option_dest", "host"),
        argvalues=[
            (
                "QUAY_IO_REGISTRY_PULL_SECRET",
                "--quay-io-registry-pull-secret",
                "quay_io_registry_pull_secret",
                QUAY_IO_REGISTRY_HOST,
            ),
            (
                "REGISTRY_STAGE_REDHAT_IO_REGISTRY_PULL_SECRET",
                "--registry-stage-redhat-io-registry-pull-secret",
                "registry_stage_redhat_io_registry_pull_secret",
                REGISTRY_STAGE_REDHAT_IO_HOST,
            ),
            (
                "REGISTRY_REDHAT_IO_REGISTRY_PULL_SECRET",
                "--registry-redhat-io-registry-pull-secret",
                "registry_redhat_io_registry_pull_secret",
                REGISTRY_REDHAT_IO_HOST,
            ),
        ],
        ids=[
            "test_quay_io_registry_pull_secret",
            "test_registry_stage_redhat_io_registry_pull_secret",
            "test_registry_redhat_io_registry_pull_secret",
        ],
    )
    def test_registry_pull_secret_reads_env_var(
        self,
        monkeypatch: pytest.MonkeyPatch,
        env_var: str,
        cli_option: str,
        option_dest: str,
        host: str,
    ) -> None:
        """Given an env var for a registry pull secret, pytest option defaults to that value."""
        for registry in MODELCAR_REGISTRIES:
            monkeypatch.delenv(key=registry.env_var, raising=False)
        monkeypatch.setenv(key=env_var, value=VALID_AUTH)

        options = _parse_registry_options(args=[])

        pytestconfig = SimpleNamespace(option=options)
        hosts, secrets = collect_modelcar_registry_credentials(pytestconfig=pytestconfig)  # type: ignore[arg-type]

        assert getattr(options, option_dest) == VALID_AUTH
        assert hosts == [host]
        assert secrets == [VALID_AUTH]

    def test_registry_pull_secret_cli_overrides_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Given both env var and CLI option, CLI option takes precedence."""
        cli_auth = base64.b64encode(b"cli-user:cli-pass").decode()
        env_auth = base64.b64encode(b"env-user:env-pass").decode()
        monkeypatch.setenv(key="QUAY_IO_REGISTRY_PULL_SECRET", value=env_auth)

        options = _parse_registry_options(args=["--quay-io-registry-pull-secret", cli_auth])
        pytestconfig = SimpleNamespace(option=options)
        hosts, secrets = collect_modelcar_registry_credentials(pytestconfig=pytestconfig)  # type: ignore[arg-type]

        assert hosts == [QUAY_IO_REGISTRY_HOST]
        assert secrets == [cli_auth]
        assert secrets != [env_auth]
