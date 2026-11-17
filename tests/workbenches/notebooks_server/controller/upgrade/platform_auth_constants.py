"""Constants for platform default-ingress auth upgrade tests (RHOAIENG-101897).

Resource names of the platform default-auth stack, discovered on a RHOAI 3.5.3
install. Kept in a shared module so both conftest.py and test files reference a
single source of truth.
"""

UPGRADE_PLATFORM_AUTH_BASELINE_CM_NAME: str = "upgrade-platform-auth-baseline"
UPGRADE_PLATFORM_AUTH_NAMESPACE: str = "upgrade-platform-auth"

# The platform default-auth resources live in openshift-ingress (namespaced) or
# cluster scope; the baseline ConfigMap lives in its own namespace instead, so
# the suite stays self-contained and no RHOAI operator reconciles it away.
PLATFORM_AUTH_NAMESPACE: str = "openshift-ingress"
DEFAULT_GATEWAY_CONFIG_NAME: str = "default-gateway"
DEFAULT_GATEWAY_NAME: str = "data-science-gateway"
PLATFORM_OAUTH_CLIENT_NAME: str = "data-science"
PLATFORM_CREDENTIALS_SECRET_NAME: str = "kube-auth-proxy-creds"
PLATFORM_OAUTH_CALLBACK_PATH: str = "/oauth2/callback"

PLATFORM_CLIENT_ID_SECRET_KEY: str = "OAUTH2_PROXY_CLIENT_ID"
PLATFORM_CLIENT_SECRET_SECRET_KEY: str = "OAUTH2_PROXY_CLIENT_SECRET"
PLATFORM_COOKIE_SECRET_KEY: str = "OAUTH2_PROXY_COOKIE_SECRET"

KUBE_AUTH_PROXY_DEPLOYMENT_NAME: str = "kube-auth-proxy"
KUBE_AUTH_PROXY_CONTAINER_NAME: str = "kube-auth-proxy"

# oauth2-proxy-style flags whose values are secret material. The baseline
# capture refuses to store any container arg starting with these, so a future
# move of credentials into args becomes a detected failure instead of a leak.
SECRET_LIKE_ARG_FLAGS: frozenset[str] = frozenset((
    "--cookie-secret",
    "--client-secret",
))
