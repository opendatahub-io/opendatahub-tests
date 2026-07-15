import pytest
from ocp_resources.data_science_pipelines_application import DataSciencePipelinesApplication
from ocp_resources.route import Route
from ocp_resources.secret import Secret

from tests.ai_safety.evalhub.constants import (
    GARAK_BENCHMARK_ID,
    GARAK_PROVIDER_ID,
    GARAK_QUICK_BENCHMARK_ID,
)
from tests.ai_safety.evalhub.utils import (
    submit_garak_job,
    validate_evalhub_health,
    validate_evalhub_providers,
    wait_for_job_completion,
)
from utilities.constants import LLMdInferenceSimConfig


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
@pytest.mark.ai_safety
@pytest.mark.usefixtures("patched_dsc_garak_kfp")
class TestGarakBenchmark:
    """Tests for running a garak security evaluation via EvalHub with KFP provider.

    Test order:
    1. Health check
    2. Provider availability
    3. Quick benchmark (smoke test via KFP - 1 probe)
    4. Quick benchmark completion
    5. Intents benchmark (full intents scan via KFP)
    6. Intents completion + S3 outputs
    """

    quick_job_id = None
    garak_job_id = None

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

    # ------------------------------------------------------------------
    # Quick benchmark (KFP smoke test)
    # ------------------------------------------------------------------

    @pytest.mark.dependency(name="garak_quick_submit", depends=["garak_providers"])
    def test_submit_quick_kfp_garak_job(
        self,
        tenant_user_token: str,
        evalhub_ca_bundle_file: str,
        garak_evalhub_route: Route,
        tenant_namespace,
        tenant_dspa: DataSciencePipelinesApplication,
        dspa_secret_patch: Secret,
        model_auth_secret_sidecar: Secret,
        dsp_access_for_job_sa,
        garak_tenant_rbac_ready: None,
        evalhub_service_secret_reader,
        garak_sim_isvc_url: str,
    ) -> None:
        """Submit a quick garak benchmark via KFP to validate the pipeline end-to-end."""
        payload = {
            "name": "garak-kfp-quick-test",
            "model": {
                "url": garak_sim_isvc_url,
                "name": LLMdInferenceSimConfig.model_name,
                # model auth secret contains K8s/KFP proxy credentials for sidecar (evalhub >= 0.4.4)
                "auth": {"secret_ref": model_auth_secret_sidecar.name},
            },
            "benchmarks": [
                {
                    "id": GARAK_QUICK_BENCHMARK_ID,
                    "provider_id": GARAK_PROVIDER_ID,
                    "parameters": {
                        "kfp_config": {
                            "namespace": tenant_namespace.name,
                            "s3_secret_name": dspa_secret_patch.name,
                            "verify_ssl": False,
                            # Real model URL for KFP pods (sidecar rewrites model.url to localhost)
                            "model_url": garak_sim_isvc_url,
                        },
                    },
                }
            ],
            "experiment": {
                "name": "garak-kfp-quick-test",
            },
        }

        job_id = submit_garak_job(
            host=garak_evalhub_route.host,
            token=tenant_user_token,
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant_namespace=tenant_namespace.name,
            payload=payload,
        )
        self.__class__.quick_job_id = job_id

    @pytest.mark.dependency(name="garak_quick_completes", depends=["garak_quick_submit"])
    def test_quick_kfp_garak_job_completes(
        self,
        tenant_user_token: str,
        evalhub_ca_bundle_file: str,
        garak_evalhub_route: Route,
        tenant_namespace,
    ) -> None:
        """Poll and verify that the quick KFP garak job completes successfully."""
        result = wait_for_job_completion(
            host=garak_evalhub_route.host,
            token=tenant_user_token,
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant_namespace=tenant_namespace.name,
            job_id=self.__class__.quick_job_id,
            timeout=600,
        )
        assert result, "Quick KFP job completion returned empty result"

    # ------------------------------------------------------------------
    # Intents benchmark (full KFP scan)
    # ------------------------------------------------------------------

    @pytest.mark.dependency(name="garak_submit", depends=["garak_quick_completes"])
    def test_submit_garak_job(
        self,
        tenant_user_token: str,
        evalhub_ca_bundle_file: str,
        garak_evalhub_route: Route,
        tenant_namespace,
        tenant_dspa: DataSciencePipelinesApplication,
        dspa_secret_patch: Secret,
        model_auth_secret_sidecar: Secret,
        dsp_access_for_job_sa,
        garak_tenant_rbac_ready: None,
        garak_sim_isvc_url: str,
        garak_intents_csv: str,
    ) -> None:
        """Submit a garak intents benchmark evaluation job using LLM-d inference simulator.

        This test is compatible with both traditional evalhub and evalhub >= 0.4.4 sidecar proxy:
        - Uses dspa_secret_patch for traditional S3 access
        - Uses model_auth_secret_sidecar for sidecar proxy cascading credential resolution
        - The garak adapter will automatically use the appropriate credential source
        """
        payload = {
            "name": "garak-intents-test",
            "model": {
                "url": garak_sim_isvc_url,
                "name": LLMdInferenceSimConfig.model_name,
                # model auth secret contains K8s/KFP proxy credentials for sidecar (evalhub >= 0.4.4)
                "auth": {"secret_ref": model_auth_secret_sidecar.name},
            },
            "benchmarks": [
                {
                    "id": GARAK_BENCHMARK_ID,
                    "provider_id": GARAK_PROVIDER_ID,
                    "parameters": {
                        "kfp_config": {
                            # endpoint omitted — sidecar injects kfp_url from model auth secret
                            "namespace": tenant_namespace.name,
                            "s3_secret_name": dspa_secret_patch.name,
                            "verify_ssl": False,
                            # Real model URL for KFP pods (sidecar rewrites model.url to localhost)
                            "model_url": garak_sim_isvc_url,
                        },
                        # Skip the SDGHub step, it'll fail to produce a dataset with our dummy model
                        "intents_s3_key": garak_intents_csv,
                        "intents_models": {  # This is a required parameter even if not used in practice
                            "judge": {"url": garak_sim_isvc_url, "name": LLMdInferenceSimConfig.model_name}
                        },
                        "garak_config": {
                            "plugins": {
                                # We only run one single probe to speed up computation
                                "probe_spec": "spo.SPOIntent",
                                # Instead of using the default model as a judge, we use a test detector
                                "detector_spec": "always.Fail",
                            },
                            "run": {"generations": 1},
                        },
                    },
                }
            ],
            "experiment": {
                "name": "garak-intents-test",
            },
        }

        job_id = submit_garak_job(
            host=garak_evalhub_route.host,
            token=tenant_user_token,
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant_namespace=tenant_namespace.name,
            payload=payload,
        )
        self.__class__.garak_job_id = job_id

    @pytest.mark.dependency(name="garak_job_completes", depends=["garak_submit"])
    def test_garak_job_completes(
        self,
        tenant_user_token: str,
        evalhub_ca_bundle_file: str,
        garak_evalhub_route: Route,
        tenant_namespace,
    ) -> None:
        """Poll and verify that the garak evaluation job completes successfully."""
        result = wait_for_job_completion(
            host=garak_evalhub_route.host,
            token=tenant_user_token,
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant_namespace=tenant_namespace.name,
            job_id=self.__class__.garak_job_id,
        )
        assert result, "Job completion returned empty result"

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

        # Parse the listing output
        lines = garak_s3_listing.strip().split("\n") if garak_s3_listing.strip() else []

        # Filter to files under the expected job path
        job_files = [line for line in lines if expected_prefix in line]

        # Expected output files
        expected_files = {
            "scan.intents.html": ("HTML report of intents scan results", True),
            "scan.report.jsonl": ("JSONL report with detailed findings", True),
            "hitlog.jsonl": ("Conversation logs from garak interactions", False), # Optional
            "scan.log": ("Garak execution logs and debug output", True),
        }

        # Check for each expected file
        found_files = {}
        missing_required = []

        for filename, (description, required) in expected_files.items():
            is_found = any(filename in f for f in job_files)
            found_files[filename] = is_found

            if required and not is_found:
                missing_required.append(f"✗ {filename} ({description})")

        # Enhanced debugging output for missing files
        if missing_required:
            print(f"\n=== S3 Output Validation Results ===")
            print(f"Expected prefix: {expected_prefix}")
            print(f"\nMISSING REQUIRED FILES:")
            for missing in missing_required:
                print(f"  {missing}")
            print(f"\nAll files matching job prefix ({len(job_files)} total):")
            for i, job_file in enumerate(job_files, 1):
                print(f"  {i}. {job_file}")
            if not job_files:
                print(f"\nFull bucket listing (no job files found):")
                print(garak_s3_listing)
            print("=== End validation ===\n")

        # Core assertions
        assert found_files["scan.intents.html"], f"Missing scan.intents.html in S3 outputs. Files found: {job_files}"
        assert found_files["scan.report.jsonl"], f"Missing scan.report.jsonl in S3 outputs. Files found: {job_files}"

        # Validate hitlog.jsonl
        if found_files["hitlog.jsonl"]:
            print(f"✓ hitlog.jsonl found in S3 outputs. Files found: {job_files}")
        else:
            print(f"✗ hitlog.jsonl not found in S3 outputs. Files found: {job_files}")

        print(f"✓ All required S3 outputs validated for job {job_id}")
