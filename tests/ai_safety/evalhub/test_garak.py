import pytest
import structlog
from ocp_resources.route import Route
from ocp_resources.secret import Secret

from tests.ai_safety.evalhub.constants import (
    GARAK_PROVIDER_ID,
)
from tests.ai_safety.evalhub.utils import (
    validate_evalhub_health,
    validate_evalhub_providers,
    wait_for_job_completion,
)

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-garak"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.usefixtures("patched_dsc_garak_kfp")
class TestGarakBenchmark:
    """Tests for running a garak security evaluation via EvalHub with KFP provider.

    Test order:
    1. Health check
    2. Provider availability
    3. Quick benchmark completion
    4. Intents benchmark completion + S3 outputs
    """

    @pytest.mark.dependency(name="garak_health")
    def test_evalhub_health(
        self,
        tenant_user_token: str,
        evalhub_ca_bundle_file: str,
        garak_evalhub_route: Route,
    ) -> None:
        """Verify the EvalHub service is healthy before running garak benchmark."""
        validate_evalhub_health(
            host=garak_evalhub_route.host,
            token=tenant_user_token,
            ca_bundle_file=evalhub_ca_bundle_file,
        )

    @pytest.mark.dependency(name="garak_providers", depends=["garak_health"])
    def test_evalhub_providers(
        self,
        tenant_user_token: str,
        evalhub_ca_bundle_file: str,
        garak_evalhub_route: Route,
        tenant_namespace,
    ) -> None:
        """Verify the garak-kfp provider is available."""
        validate_evalhub_providers(
            host=garak_evalhub_route.host,
            token=tenant_user_token,
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant_namespace=tenant_namespace.name,
            expected_providers=[GARAK_PROVIDER_ID],
        )

    @pytest.mark.dependency(name="garak_quick_completes", depends=["garak_providers"])
    def test_quick_kfp_garak_job_completes(
        self,
        tenant_user_token: str,
        evalhub_ca_bundle_file: str,
        garak_evalhub_route: Route,
        tenant_namespace,
        quick_kfp_garak_job_id: str,
    ) -> None:
        """Poll and verify that the quick KFP garak job completes successfully."""
        result = wait_for_job_completion(
            host=garak_evalhub_route.host,
            token=tenant_user_token,
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant_namespace=tenant_namespace.name,
            job_id=quick_kfp_garak_job_id,
            timeout=600,
        )
        assert result, "Quick KFP job completion returned empty result"

    @pytest.mark.dependency(name="garak_job_completes", depends=["garak_quick_completes"])
    def test_garak_job_completes(
        self,
        tenant_user_token: str,
        evalhub_ca_bundle_file: str,
        garak_evalhub_route: Route,
        tenant_namespace,
        intents_kfp_garak_job_id: str,
    ) -> None:
        """Poll and verify that the garak evaluation job completes successfully."""
        result = wait_for_job_completion(
            host=garak_evalhub_route.host,
            token=tenant_user_token,
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant_namespace=tenant_namespace.name,
            job_id=intents_kfp_garak_job_id,
        )
        assert result, "Job completion returned empty result"
        self.__class__.garak_job_id = intents_kfp_garak_job_id

    @pytest.mark.dependency(depends=["garak_job_completes"])
    def test_garak_s3_outputs(
        self,
        admin_client,
        tenant_namespace,
        dspa_secret_patch: Secret,
        garak_s3_listing: str,
    ) -> None:
        """Verify the garak job produces expected S3 output files."""
        job_id = self.__class__.garak_job_id
        expected_prefix = f"evalhub-garak-kfp/{job_id}/"

        lines = garak_s3_listing.strip().split("\n") if garak_s3_listing.strip() else []
        job_files = [line for line in lines if expected_prefix in line]

        expected_files = {
            "scan.intents.html": ("HTML report of intents scan results", True),
            "scan.report.jsonl": ("JSONL report with detailed findings", True),
            "hitlog.jsonl": ("Conversation logs from garak interactions", True),
            "scan.log": ("Garak execution logs and debug output", False),
        }

        found_files = {}
        missing_required = []

        for filename, (description, required) in expected_files.items():
            is_found = any(filename in f for f in job_files)
            found_files[filename] = is_found
            if required and not is_found:
                missing_required.append(f"{filename} ({description})")

        if missing_required:
            LOGGER.error(
                f"Missing required S3 files for job {job_id}: {missing_required}. "
                f"Found {len(job_files)} files: {job_files}"
            )

        assert found_files["scan.intents.html"], f"Missing scan.intents.html in S3 outputs. Files found: {job_files}"
        assert found_files["scan.report.jsonl"], f"Missing scan.report.jsonl in S3 outputs. Files found: {job_files}"
        assert found_files["hitlog.jsonl"], f"Missing hitlog.jsonl in S3 outputs. Files found: {job_files}"
