import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService

from utilities.constants import RuntimeTemplates

CA_BUNDLE_VOLUME_NAME = "openshift-service-ca-bundle"
CA_BUNDLE_CONFIGMAP_NAME = "openshift-service-ca.crt"
CA_BUNDLE_MOUNT_PATH = "/etc/odh/openshift-service-ca-bundle"
CA_BUNDLE_CERT_FILE = "service-ca.crt"
KSERVE_CONTAINER_NAME = "kserve-container"


def _get_component_deployment(
    client: DynamicClient,
    isvc: InferenceService,
    component: str,
) -> Deployment:
    """Retrieve the deployment for a specific ISVC component (predictor or transformer)."""
    label_selector = f"serving.kserve.io/inferenceservice={isvc.name},component={component}"
    deployments = list(Deployment.get(client=client, namespace=isvc.namespace, label_selector=label_selector))
    assert deployments, f"No {component} deployment found for ISVC {isvc.name}"
    return deployments[0]


def _get_kserve_container(deployment: Deployment):
    """Find the kserve-container in a deployment's pod spec."""
    containers = deployment.instance.spec.template.spec.containers
    for container in containers:
        if container.name == KSERVE_CONTAINER_NAME:
            return container
    raise AssertionError(
        f"Container {KSERVE_CONTAINER_NAME!r} not found in deployment {deployment.name}. "
        f"Available: {[c.name for c in containers]}"
    )


@pytest.mark.tls
@pytest.mark.tier1
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "unprivileged_model_namespace, transformer_auth_inference_service",
    [
        pytest.param(
            {"name": "test-kserve-transformer-tls"},
            {
                "name": "onnx",
                "template-name": RuntimeTemplates.MLSERVER,
                "multi-model": False,
                "model-dir": "sentiment-analysis",
            },
        )
    ],
    indirect=True,
)
class TestTransformerTLSInfrastructure:
    """Verify TLS infrastructure injection into transformer deployment.

    When ``security.opendatahub.io/enable-auth`` is ``"true"`` on an
    InferenceService with a transformer, the controller should inject:
      - ``openshift-service-ca-bundle`` volume (from the service-ca ConfigMap)
      - Volume mount into ``kserve-container`` at ``/etc/odh/openshift-service-ca-bundle``
      - ``SSL_CERT_DIR`` and ``REQUESTS_CA_BUNDLE`` env vars for CA trust
      - ``PREDICTOR_HOST``, ``PREDICTOR_PORT``, ``PREDICTOR_PROTOCOL`` env vars
        for the transformer to discover the predictor's TLS endpoint

    The predictor deployment should NOT receive these env vars.
    """

    def test_transformer_has_ca_bundle_volume(self, unprivileged_client, transformer_auth_inference_service):
        """Transformer deployment has openshift-service-ca-bundle volume from ConfigMap."""
        deployment = _get_component_deployment(
            client=unprivileged_client,
            isvc=transformer_auth_inference_service,
            component="transformer",
        )
        volumes = deployment.instance.spec.template.spec.volumes
        volume_names = [v.name for v in volumes]
        assert CA_BUNDLE_VOLUME_NAME in volume_names, (
            f"Transformer deployment should have {CA_BUNDLE_VOLUME_NAME} volume, got: {volume_names}"
        )

        ca_volume = next(v for v in volumes if v.name == CA_BUNDLE_VOLUME_NAME)
        assert ca_volume.configMap is not None, "CA bundle volume should reference a ConfigMap"
        assert ca_volume.configMap.name == CA_BUNDLE_CONFIGMAP_NAME, (
            f"Expected ConfigMap name {CA_BUNDLE_CONFIGMAP_NAME}, got {ca_volume.configMap.name}"
        )

    def test_transformer_has_ca_bundle_volume_mount(self, unprivileged_client, transformer_auth_inference_service):
        """kserve-container has CA bundle mount at /etc/odh/openshift-service-ca-bundle."""
        deployment = _get_component_deployment(
            client=unprivileged_client,
            isvc=transformer_auth_inference_service,
            component="transformer",
        )
        container = _get_kserve_container(deployment=deployment)

        mount_names = [vm.name for vm in container.volumeMounts]
        assert CA_BUNDLE_VOLUME_NAME in mount_names, (
            f"kserve-container should have {CA_BUNDLE_VOLUME_NAME} volume mount, got: {mount_names}"
        )

        ca_mount = next(vm for vm in container.volumeMounts if vm.name == CA_BUNDLE_VOLUME_NAME)
        assert ca_mount.mountPath == CA_BUNDLE_MOUNT_PATH, (
            f"Expected mount path {CA_BUNDLE_MOUNT_PATH}, got {ca_mount.mountPath}"
        )
        assert ca_mount.readOnly is True, "CA bundle mount should be read-only"

    def test_transformer_has_tls_env_vars_and_ssl_arg(self, unprivileged_client, transformer_auth_inference_service):
        """kserve-container has TLS env vars and --predictor_use_ssl arg."""
        deployment = _get_component_deployment(
            client=unprivileged_client,
            isvc=transformer_auth_inference_service,
            component="transformer",
        )
        container = _get_kserve_container(deployment=deployment)
        env_map = {env.name: env.value for env in container.env}

        expected_predictor_host = (
            f"{transformer_auth_inference_service.name}-predictor.{transformer_auth_inference_service.namespace}.svc"
        )

        assert env_map.get("SSL_CERT_DIR") == CA_BUNDLE_MOUNT_PATH, (
            f"Expected SSL_CERT_DIR={CA_BUNDLE_MOUNT_PATH}, got: {env_map.get('SSL_CERT_DIR')}"
        )
        assert env_map.get("REQUESTS_CA_BUNDLE") == f"{CA_BUNDLE_MOUNT_PATH}/{CA_BUNDLE_CERT_FILE}", (
            f"Expected REQUESTS_CA_BUNDLE with cert file, got: {env_map.get('REQUESTS_CA_BUNDLE')}"
        )
        assert env_map.get("PREDICTOR_HOST") == expected_predictor_host, (
            f"Expected PREDICTOR_HOST={expected_predictor_host}, got: {env_map.get('PREDICTOR_HOST')}"
        )
        assert env_map.get("PREDICTOR_PORT") == "8443", (
            f"Expected PREDICTOR_PORT=8443, got: {env_map.get('PREDICTOR_PORT')}"
        )
        assert env_map.get("PREDICTOR_PROTOCOL") == "https", (
            f"Expected PREDICTOR_PROTOCOL=https, got: {env_map.get('PREDICTOR_PROTOCOL')}"
        )

        # Controller also appends --predictor_use_ssl true to container args
        args = container.args or []
        assert "--predictor_use_ssl" in args, f"Expected --predictor_use_ssl in transformer container args, got: {args}"
        ssl_idx = args.index("--predictor_use_ssl")
        assert ssl_idx + 1 < len(args) and args[ssl_idx + 1] == "true", (
            f"Expected --predictor_use_ssl true, got: {args[ssl_idx:]}"
        )

    def test_predictor_does_not_have_tls_env_vars(self, unprivileged_client, transformer_auth_inference_service):
        """Predictor deployment must NOT have PREDICTOR_HOST/PORT/PROTOCOL env vars."""
        deployment = _get_component_deployment(
            client=unprivileged_client,
            isvc=transformer_auth_inference_service,
            component="predictor",
        )
        container = _get_kserve_container(deployment=deployment)
        env_names = [env.name for env in container.env] if container.env else []

        tls_only_env_vars = ["PREDICTOR_HOST", "PREDICTOR_PORT", "PREDICTOR_PROTOCOL"]
        for var in tls_only_env_vars:
            assert var not in env_names, f"Predictor should NOT have {var} env var"
