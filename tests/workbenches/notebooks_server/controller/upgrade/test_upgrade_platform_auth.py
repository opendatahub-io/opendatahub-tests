import json
from typing import Any

import pytest
from ocp_resources.deployment import Deployment
from ocp_resources.gateway_config import GatewayConfig
from ocp_resources.gateway_gateway_networking_k8s_io import Gateway
from ocp_resources.secret import Secret

from tests.workbenches.notebooks_server.controller.upgrade.platform_auth_constants import (
    KUBE_AUTH_PROXY_CONTAINER_NAME,
    PLATFORM_CLIENT_ID_SECRET_KEY,
    PLATFORM_CLIENT_SECRET_SECRET_KEY,
    PLATFORM_COOKIE_SECRET_KEY,
    PLATFORM_OAUTH_CALLBACK_PATH,
)
from tests.workbenches.notebooks_server.controller.utils import (
    OAuthClient,
    normalized_json,
    secret_data_digests,
    sha256_digest,
)

# Transport-level TLS args: their values are unordered lists and are allowed
# to evolve between versions (cipher suite and curve hardening);
# authentication-relevant args must not change across an upgrade.
TRANSPORT_TLS_ARGS: frozenset[str] = frozenset((
    "--tls-cipher-suite",
    "--tls-curve-preferences",
))


def _normalized_arg_set(args: list[str]) -> set[str]:
    """Return a set of CLI args, sorting comma-separated values of list flags.

    Allows benign reordering of unordered list-valued options (e.g. TLS cipher
    suites) while keeping scalar-valued options (e.g. --redirect-url) exact,
    preserving any comma order within them.

    Args:
        args (list[str]): Container args.

    Returns:
        set[str]: Normalized args.
    """
    normalized: set[str] = set()
    for arg in args:
        if "=" in arg:
            flag, value = arg.split("=", 1)
            if flag in TRANSPORT_TLS_ARGS and "," in value:
                value = ",".join(sorted(value.split(",")))
            arg = f"{flag}={value}"
        normalized.add(arg)
    return normalized


def _assert_settings_subset(baseline: Any, current: Any, resource: str, setting_path: str) -> None:
    """Recursively assert that baseline settings are preserved in current values.

    Mappings are compared as subsets (fields newly introduced by an upgrade are
    allowed inside existing settings); scalars and lists must match exactly.

    Args:
        baseline (Any): Value captured before upgrade.
        current (Any): Value observed after upgrade.
        resource (str): Resource identifier for failure messages.
        setting_path (str): Dotted path of the setting, for failure messages.
    """
    if isinstance(baseline, dict) and isinstance(current, dict):
        missing_keys = set(baseline) - set(current)
        assert not missing_keys, f"{resource} lost settings after upgrade at '{setting_path}': {sorted(missing_keys)}"
        for key, baseline_value in baseline.items():
            _assert_settings_subset(
                baseline=baseline_value,
                current=current[key],
                resource=resource,
                setting_path=f"{setting_path}.{key}",
            )
        return

    assert baseline == current, (
        f"{resource} setting '{setting_path}' changed after upgrade: '{current}' != baseline '{baseline}'"
    )


@pytest.mark.usefixtures("capture_platform_auth_baseline")
class TestPreUpgradePlatformAuth:
    """Verify the platform default-ingress auth stack before upgrade.

    Steps:
        1. Verify the GatewayConfig for the default platform ingress exists.
        2. Verify the default Gateway exists with a stable identity.
        3. Verify the platform OAuthClient exists with the oauth2 callback redirect.
        4. Verify the credentials Secret exists with client and cookie keys.
        5. Verify the kube-auth-proxy Deployment exists, has its container and is Available.
    """

    @pytest.mark.pre_upgrade
    def test_gateway_config_exists(
        self,
        default_gateway_config: GatewayConfig,
    ) -> None:
        """Given a RHOAI deployment with default ingress authentication,
        When the gateway operator reconciles,
        Then the GatewayConfig for the default ingress should exist with a non-empty spec.
        """
        assert default_gateway_config.exists, f"GatewayConfig '{default_gateway_config.name}' does not exist"
        assert default_gateway_config.instance.spec, f"GatewayConfig '{default_gateway_config.name}' has an empty spec"

    @pytest.mark.pre_upgrade
    def test_default_gateway_exists(
        self,
        default_gateway: Gateway,
    ) -> None:
        """Given a GatewayConfig for the default ingress,
        When the gateway operator reconciles,
        Then the default Gateway should exist with a stable identity (UID).
        """
        assert default_gateway.exists, f"Gateway '{default_gateway.name}' does not exist"
        assert default_gateway.instance.metadata.uid, f"Gateway '{default_gateway.name}' has no UID in metadata"

    @pytest.mark.pre_upgrade
    def test_platform_oauth_client_exists(
        self,
        platform_oauth_client: OAuthClient,
    ) -> None:
        """Given a RHOAI deployment with default ingress authentication,
        When kube-auth-proxy registers with the cluster OAuth server,
        Then the platform OAuthClient should exist with a grant method, a secret
        and a redirect URI ending in the oauth2 callback path.
        """
        assert platform_oauth_client.exists, f"OAuthClient '{platform_oauth_client.name}' does not exist"

        assert platform_oauth_client.instance.grantMethod, (
            f"OAuthClient '{platform_oauth_client.name}' has an empty grantMethod"
        )

        assert platform_oauth_client.instance.secret, f"OAuthClient '{platform_oauth_client.name}' has an empty secret"

        redirect_uris = platform_oauth_client.instance.redirectURIs or []
        assert redirect_uris, f"OAuthClient '{platform_oauth_client.name}' has no redirectURIs"
        assert all(uri.endswith(PLATFORM_OAUTH_CALLBACK_PATH) for uri in redirect_uris), (
            f"OAuthClient '{platform_oauth_client.name}' has redirect URIs not ending in "
            f"'{PLATFORM_OAUTH_CALLBACK_PATH}': {redirect_uris}"
        )

    @pytest.mark.pre_upgrade
    def test_platform_credentials_secret_exists(
        self,
        platform_credentials_secret: Secret,
    ) -> None:
        """Given kube-auth-proxy registered with the cluster OAuth server,
        When the gateway operator stores its credentials,
        Then the credentials Secret should exist with the client id, client secret
        and cookie secret keys.
        """
        assert platform_credentials_secret.exists, (
            f"Secret '{platform_credentials_secret.name}' does not exist in "
            f"namespace '{platform_credentials_secret.namespace}'"
        )

        data_keys = set(platform_credentials_secret.instance.data.keys())
        for secret_key in (
            PLATFORM_CLIENT_ID_SECRET_KEY,
            PLATFORM_CLIENT_SECRET_SECRET_KEY,
            PLATFORM_COOKIE_SECRET_KEY,
        ):
            assert secret_key in data_keys, (
                f"Secret '{platform_credentials_secret.name}' missing key '{secret_key}'. "
                f"Found keys: {sorted(data_keys)}"
            )

    @pytest.mark.pre_upgrade
    def test_kube_auth_proxy_deployment_available(
        self,
        kube_auth_proxy_deployment: Deployment,
    ) -> None:
        """Given the platform default ingress is authenticated,
        When the gateway operator deploys the auth proxy,
        Then the kube-auth-proxy Deployment should exist with its container and
        report an Available condition.
        """
        assert kube_auth_proxy_deployment.exists, (
            f"Deployment '{kube_auth_proxy_deployment.name}' does not exist in "
            f"namespace '{kube_auth_proxy_deployment.namespace}'"
        )

        containers = kube_auth_proxy_deployment.instance.spec.template.spec.containers
        container_names = [container.name for container in containers]
        assert KUBE_AUTH_PROXY_CONTAINER_NAME in container_names, (
            f"Deployment '{kube_auth_proxy_deployment.name}' missing container "
            f"'{KUBE_AUTH_PROXY_CONTAINER_NAME}'. Containers: {container_names}"
        )

        conditions = kube_auth_proxy_deployment.instance.status.conditions or []
        available_conditions = [condition for condition in conditions if condition.type == "Available"]
        assert available_conditions, f"Deployment '{kube_auth_proxy_deployment.name}' has no Available condition"
        assert available_conditions[0].status == "True", (
            f"Deployment '{kube_auth_proxy_deployment.name}' Available condition is "
            f"'{available_conditions[0].status}', expected 'True'"
        )


class TestPostUpgradePlatformAuth:
    """Verify the platform default-ingress auth stack survived the upgrade.

    The pre-upgrade baseline is compared via digests and normalized specs only;
    raw secret values are never read into assertions or logs.

    Steps:
        1. Verify every captured GatewayConfig setting is unchanged (new fields allowed).
        2. Verify the default Gateway kept its identity (UID).
        3. Verify the OAuthClient registration (grant method, redirect URIs, secret) is unchanged.
        4. Verify the credentials Secret kept its keys.
        5. Verify the credentials Secret values (incl. cookie secret) were not rotated.
        6. Verify the kube-auth-proxy Deployment kept its identity, image repository and args
           (transport TLS ciphers/curves may evolve; image digest advances with the upgrade).
        7. Verify the kube-auth-proxy Deployment still reports an Available condition.
    """

    @pytest.mark.post_upgrade
    def test_gateway_config_settings_preserved(
        self,
        default_gateway_config: GatewayConfig,
        platform_auth_baseline: dict[str, Any],
    ) -> None:
        """Given a GatewayConfig spec captured before upgrade,
        When the upgrade completes,
        Then every captured setting should be unchanged; fields newly
        introduced by the upgraded version are allowed.
        """
        assert default_gateway_config.exists, (
            f"GatewayConfig '{default_gateway_config.name}' no longer exists after upgrade"
        )

        # The baseline stores the spec as plain JSON (captured via to_dict).
        baseline_spec = platform_auth_baseline["gateway_config_spec"]
        current_spec = default_gateway_config.instance.spec.to_dict()

        # Baseline settings must be preserved; fields newly introduced by the
        # upgrade are allowed, including inside existing nested settings.
        _assert_settings_subset(
            baseline=baseline_spec,
            current=current_spec,
            resource=f"GatewayConfig '{default_gateway_config.name}'",
            setting_path="spec",
        )

    @pytest.mark.post_upgrade
    def test_default_gateway_identity_preserved(
        self,
        default_gateway: Gateway,
        platform_auth_baseline: dict[str, Any],
    ) -> None:
        """Given a default Gateway with a captured UID,
        When the upgrade completes,
        Then the Gateway should still exist with the same UID (not recreated).
        """
        assert default_gateway.exists, f"Gateway '{default_gateway.name}' no longer exists after upgrade"
        assert default_gateway.instance.metadata.uid == platform_auth_baseline["default_gateway_uid"], (
            f"Gateway '{default_gateway.name}' was recreated during upgrade: "
            f"UID '{default_gateway.instance.metadata.uid}' != "
            f"baseline '{platform_auth_baseline['default_gateway_uid']}'"
        )

    @pytest.mark.post_upgrade
    def test_oauth_client_registration_preserved(
        self,
        platform_oauth_client: OAuthClient,
        platform_auth_baseline: dict[str, Any],
    ) -> None:
        """Given a platform OAuthClient registration captured before upgrade,
        When the upgrade completes,
        Then grant method, redirect URIs and the secret digest should be unchanged
        (secret compared via sha256 digest, never in raw form).
        """
        assert platform_oauth_client.exists, (
            f"OAuthClient '{platform_oauth_client.name}' no longer exists after upgrade"
        )

        assert platform_oauth_client.instance.grantMethod == platform_auth_baseline["oauth_client_grant_method"], (
            f"OAuthClient '{platform_oauth_client.name}' grantMethod changed after upgrade: "
            f"'{platform_oauth_client.instance.grantMethod}'"
        )

        assert (
            normalized_json(platform_oauth_client.instance.redirectURIs)
            == platform_auth_baseline["oauth_client_redirect_uris"]
        ), (
            f"OAuthClient '{platform_oauth_client.name}' redirectURIs changed after upgrade: "
            f"{platform_oauth_client.instance.redirectURIs}"
        )

        assert (
            sha256_digest(value=platform_oauth_client.instance.secret)
            == platform_auth_baseline["oauth_client_secret_digest"]
        ), f"OAuthClient '{platform_oauth_client.name}' secret was rotated during upgrade (sha256 digest mismatch)"

    @pytest.mark.post_upgrade
    def test_credentials_secret_keys_preserved(
        self,
        platform_credentials_secret: Secret,
        platform_auth_baseline: dict[str, Any],
    ) -> None:
        """Given a credentials Secret with captured key names,
        When the upgrade completes,
        Then the Secret should still exist with the same set of keys.
        """
        assert platform_credentials_secret.exists, (
            f"Secret '{platform_credentials_secret.name}' no longer exists after upgrade"
        )
        assert (
            sorted(platform_credentials_secret.instance.data.keys())
            == platform_auth_baseline["credentials_secret_keys"]
        ), (
            f"Secret '{platform_credentials_secret.name}' keys changed after upgrade: "
            f"{sorted(platform_credentials_secret.instance.data.keys())}"
        )

    @pytest.mark.post_upgrade
    def test_cookie_secret_not_rotated(
        self,
        platform_credentials_secret: Secret,
        platform_auth_baseline: dict[str, Any],
    ) -> None:
        """Given credentials Secret values captured as digests before upgrade,
        When the upgrade completes,
        Then all value digests (incl. the cookie secret) should be unchanged,
        proving users are not forced to re-login.
        """
        current_digests = secret_data_digests(secret=platform_credentials_secret)
        baseline_digests = platform_auth_baseline["credentials_secret_digests"]
        rotated_keys = sorted(
            digest_key
            for digest_key in set(current_digests) | set(baseline_digests)
            if current_digests.get(digest_key) != baseline_digests.get(digest_key)
        )
        assert current_digests == baseline_digests, (
            f"Secret '{platform_credentials_secret.name}' values were rotated during upgrade "
            f"(sha256 digest mismatch; keys affected: {rotated_keys})"
        )

    @pytest.mark.post_upgrade
    def test_kube_auth_proxy_identity_preserved(
        self,
        kube_auth_proxy_deployment: Deployment,
        platform_auth_baseline: dict[str, Any],
    ) -> None:
        """Given a kube-auth-proxy Deployment with captured identity, image and args,
        When the upgrade completes,
        Then UID and auth-relevant args should be unchanged (transport-level
        TLS cipher suites and curves may evolve; list values may reorder),
        and the image repository should be unchanged (the digest is expected
        to advance with the upgrade).
        """
        assert kube_auth_proxy_deployment.exists, (
            f"Deployment '{kube_auth_proxy_deployment.name}' no longer exists after upgrade"
        )

        assert kube_auth_proxy_deployment.instance.metadata.uid == platform_auth_baseline["kube_auth_proxy_uid"], (
            f"Deployment '{kube_auth_proxy_deployment.name}' was recreated during upgrade: "
            f"UID '{kube_auth_proxy_deployment.instance.metadata.uid}' != "
            f"baseline '{platform_auth_baseline['kube_auth_proxy_uid']}'"
        )

        containers = kube_auth_proxy_deployment.instance.spec.template.spec.containers
        auth_proxy_container = next(
            (container for container in containers if container.name == KUBE_AUTH_PROXY_CONTAINER_NAME),
            None,
        )
        assert auth_proxy_container is not None, (
            f"Deployment '{kube_auth_proxy_deployment.name}' has no '{KUBE_AUTH_PROXY_CONTAINER_NAME}' container"
        )

        baseline_repository = platform_auth_baseline["kube_auth_proxy_image"].split("@")[0]
        current_repository = auth_proxy_container.image.split("@")[0]
        assert current_repository == baseline_repository, (
            f"Deployment '{kube_auth_proxy_deployment.name}' container image repository changed "
            f"after upgrade: '{current_repository}' != baseline '{baseline_repository}'"
        )

        baseline_args = json.loads(platform_auth_baseline["kube_auth_proxy_args"])
        current_args = list(auth_proxy_container.args)
        baseline_arg_set = _normalized_arg_set(args=baseline_args)
        current_arg_set = _normalized_arg_set(args=current_args)

        def _auth_relevant(arg: str) -> bool:
            return arg.split("=", 1)[0] not in TRANSPORT_TLS_ARGS

        removed_auth_args = sorted(arg for arg in baseline_arg_set - current_arg_set if _auth_relevant(arg))
        assert not removed_auth_args, (
            f"Deployment '{kube_auth_proxy_deployment.name}' container lost auth-relevant args "
            f"after upgrade: {removed_auth_args}"
        )

        added_auth_args = sorted(arg for arg in current_arg_set - baseline_arg_set if _auth_relevant(arg))
        assert not added_auth_args, (
            f"Deployment '{kube_auth_proxy_deployment.name}' container gained new auth-relevant args "
            f"after upgrade: {added_auth_args}"
        )

    @pytest.mark.post_upgrade
    def test_kube_auth_proxy_deployment_available_after_upgrade(
        self,
        kube_auth_proxy_deployment: Deployment,
    ) -> None:
        """Given a kube-auth-proxy Deployment that was Available before upgrade,
        When the upgrade completes and the deployment reconciles,
        Then the Deployment should still report an Available condition.
        """
        assert kube_auth_proxy_deployment.exists, (
            f"Deployment '{kube_auth_proxy_deployment.name}' no longer exists after upgrade"
        )

        conditions = kube_auth_proxy_deployment.instance.status.conditions or []
        available_conditions = [condition for condition in conditions if condition.type == "Available"]
        assert available_conditions, f"Deployment '{kube_auth_proxy_deployment.name}' has no Available condition"
        assert available_conditions[0].status == "True", (
            f"Deployment '{kube_auth_proxy_deployment.name}' Available condition is "
            f"'{available_conditions[0].status}' after upgrade; the upgraded auth proxy is not serving"
        )
