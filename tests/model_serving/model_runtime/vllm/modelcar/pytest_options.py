import os

from _pytest.config.argparsing import Parser

from tests.model_serving.model_runtime.vllm.modelcar.constant import MODELCAR_REGISTRIES


def register_modelcar_registry_pull_secret_options(parser: Parser) -> None:
    """Register vLLM OCI registry pull-secret CLI options on the pytest parser."""
    ociregistry_group = parser.getgroup(name="OCI Registry")
    for registry in MODELCAR_REGISTRIES:
        ociregistry_group.addoption(
            registry.cli_option,
            default=os.environ.get(registry.env_var),
            help=f"Pull secret for {registry.host} modelcar images (base64 auth or JSON credentials)",
        )
