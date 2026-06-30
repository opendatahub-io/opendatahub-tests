"""TrustyAI Service CRD conversion webhook tests.

Tests for the v1alpha1 <-> v1 conversion webhook on the TrustyAI Service CRD
(trustyaiservices.trustyai.opendatahub.io). Both API versions are served; v1
is the storage version. The operator implements bidirectional conversion via a
webhook so resources created in either version are readable via the other.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.namespace import Namespace
from ocp_resources.trustyai_service import TrustyAIService

TRUSTYAI_API_GROUP: str = "trustyai.opendatahub.io"
TRUSTYAI_PLURAL: str = "trustyaiservices"


class TrustyAIServiceV1(TrustyAIService):
    api_version = f"{TRUSTYAI_API_GROUP}/v1"


class TrustyAIServiceV1Alpha1(TrustyAIService):
    api_version = f"{TRUSTYAI_API_GROUP}/v1alpha1"


_STORAGE_PVC = {"format": "PVC", "folder": "/inputs", "size": "1Gi"}
_DATA = {"filename": "data.csv", "format": "CSV"}
_METRICS = {"schedule": "5s"}


@pytest.mark.smoke
@pytest.mark.ai_safety
def test_trustyai_service_crd_serves_both_versions(
    admin_client: DynamicClient,
) -> None:
    """Verify the TrustyAI Service CRD advertises both v1alpha1 and v1 versions."""
    crd_name = f"{TRUSTYAI_PLURAL}.{TRUSTYAI_API_GROUP}"
    crd = CustomResourceDefinition(
        client=admin_client,
        name=crd_name,
        ensure_exists=True,
    )
    assert crd.exists, f"CRD {crd_name} not found"

    served_versions = {version["name"] for version in crd.instance.spec.versions if version.get("served", False)}
    assert "v1alpha1" in served_versions, f"v1alpha1 not served; served versions: {served_versions}"
    assert "v1" in served_versions, f"v1 not served; served versions: {served_versions}"


@pytest.mark.smoke
@pytest.mark.ai_safety
def test_trustyai_service_crd_conversion_strategy_is_webhook(
    admin_client: DynamicClient,
) -> None:
    """Verify the TrustyAI Service CRD uses the Webhook conversion strategy."""
    crd_name = f"{TRUSTYAI_PLURAL}.{TRUSTYAI_API_GROUP}"
    crd = CustomResourceDefinition(
        client=admin_client,
        name=crd_name,
        ensure_exists=True,
    )
    assert crd.exists, f"CRD {crd_name} not found"

    conversion = crd.instance.spec.get("conversion", {})
    assert conversion.get("strategy") == "Webhook", (
        f"Expected conversion strategy 'Webhook', got '{conversion.get('strategy')}'"
    )


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-trustyai-conversion"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.ai_safety
class TestTrustyAIServiceCRDConversion:
    """Tests for the TrustyAI Service v1alpha1 <-> v1 CRD conversion webhook."""

    def test_create_v1alpha1_read_as_v1(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
    ) -> None:
        """Create a TrustyAIService via v1alpha1 and read it back as v1."""
        with TrustyAIServiceV1Alpha1(
            client=admin_client,
            name="conv-v1alpha1-to-v1",
            namespace=model_namespace.name,
            storage=_STORAGE_PVC,
            data=_DATA,
            metrics=_METRICS,
        ) as svc:
            result = TrustyAIServiceV1(
                client=admin_client,
                name=svc.name,
                namespace=model_namespace.name,
                ensure_exists=True,
            )
            assert result.instance.spec.storage.format == "PVC"
            assert result.instance.spec.data.filename == "data.csv"
            assert result.instance.spec.metrics.schedule == "5s"

    def test_create_v1_read_as_v1alpha1(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
    ) -> None:
        """Create a TrustyAIService via v1 and read it back as v1alpha1."""
        with TrustyAIServiceV1(
            client=admin_client,
            name="conv-v1-to-v1alpha1",
            namespace=model_namespace.name,
            storage=_STORAGE_PVC,
            data=_DATA,
            metrics=_METRICS,
        ) as svc:
            result = TrustyAIServiceV1Alpha1(
                client=admin_client,
                name=svc.name,
                namespace=model_namespace.name,
                ensure_exists=True,
            )
            assert result.instance.spec.storage.format == "PVC"
            assert result.instance.spec.data.filename == "data.csv"
            assert result.instance.spec.metrics.schedule == "5s"

    def test_conversion_preserves_storage_config(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
    ) -> None:
        """Create a TrustyAIService with full storage config in v1alpha1, read as v1, verify all fields."""
        with TrustyAIServiceV1Alpha1(
            client=admin_client,
            name="conv-storage-preserve",
            namespace=model_namespace.name,
            storage={"format": "PVC", "folder": "/inputs", "size": "2Gi"},
            data={"filename": "data.csv", "format": "CSV"},
            metrics={"schedule": "10s"},
            replicas=1,
        ) as svc:
            result = TrustyAIServiceV1(
                client=admin_client,
                name=svc.name,
                namespace=model_namespace.name,
                ensure_exists=True,
            )
            assert result.instance.spec.storage.format == "PVC"
            assert result.instance.spec.storage.folder == "/inputs"
            assert result.instance.spec.storage.size == "2Gi"
            assert result.instance.spec.data.filename == "data.csv"
            assert result.instance.spec.data.format == "CSV"
            assert result.instance.spec.metrics.schedule == "10s"

    def test_roundtrip_v1alpha1_to_v1_to_v1alpha1(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
    ) -> None:
        """Verify roundtrip: create as v1alpha1, read as v1, read back as v1alpha1."""
        with TrustyAIServiceV1Alpha1(
            client=admin_client,
            name="conv-roundtrip",
            namespace=model_namespace.name,
            storage=_STORAGE_PVC,
            data=_DATA,
            metrics=_METRICS,
        ) as svc:
            v1_result = TrustyAIServiceV1(
                client=admin_client,
                name=svc.name,
                namespace=model_namespace.name,
                ensure_exists=True,
            )
            assert v1_result.instance.spec.storage.format == "PVC"
            assert v1_result.instance.spec.data.filename == "data.csv"

            roundtrip = TrustyAIServiceV1Alpha1(
                client=admin_client,
                name=svc.name,
                namespace=model_namespace.name,
                ensure_exists=True,
            )
            assert roundtrip.instance.spec.storage.format == "PVC"
            assert roundtrip.instance.spec.storage.folder == "/inputs"
            assert roundtrip.instance.spec.data.filename == "data.csv"
            assert roundtrip.instance.spec.metrics.schedule == "5s"
