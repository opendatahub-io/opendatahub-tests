import pytest
from ocp_resources.data_science_pipelines_application import DataSciencePipelinesApplication
from ocp_resources.route import Route
from ocp_resources.secret import Secret

from tests.model_explainability.evalhub.constants import (
    GARAK_BENCHMARK_ID,
    GARAK_PROVIDER_ID,
)
from tests.model_explainability.evalhub.utils import (
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
@pytest.mark.model_explainability
@pytest.mark.usefixtures("patched_dsc_garak_kfp")
class TestGarakBenchmark:
    """Tests for running a garak security evaluation via EvalHub with KFP provider."""

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

    @pytest.mark.dependency(name="garak_submit", depends=["garak_providers"])
    def test_submit_garak_job(
        self,
        tenant_user_token: str,
        evalhub_ca_bundle_file: str,
        garak_evalhub_route: Route,
        tenant_namespace,
        tenant_dspa: DataSciencePipelinesApplication,
        dspa_secret_patch: Secret,
        dsp_access_for_job_sa,
        garak_tenant_rbac_ready: None,
        garak_sim_isvc_url: str,
        garak_intents_csv: str,
    ) -> None:
        """Submit a garak quick benchmark evaluation job using LLM-d inference simulator."""
        kfp_endpoint = f"https://ds-pipeline-dspa.{tenant_namespace.name}.svc.cluster.local:8443"

        payload = {
            "name": "garak-quick-test",
            "model": {
                "url": garak_sim_isvc_url,
                "name": LLMdInferenceSimConfig.model_name,
            },
            "benchmarks": [
                {
                    "id": GARAK_BENCHMARK_ID,
                    "provider_id": GARAK_PROVIDER_ID,
                    "parameters": {
                        "kfp_config": {
                            "endpoint": kfp_endpoint,
                            "namespace": tenant_namespace.name,
                            "s3_secret_name": dspa_secret_patch.name,
                            "s3_endpoint": f"http://minio-dspa.{tenant_namespace.name}.svc.cluster.local:9000",
                            "s3_bucket": "mlpipeline",
                            "verify_ssl": False,
                        },
                        # Skip the SDGHub step, it'll fail to produce a dataset with our dummy model
                        "intents_s3_key": garak_intents_csv,
                        "intents_models": { # This is a required parameter even if not used in practice
                            "judge": {
                                "url": garak_sim_isvc_url,
                                "name": LLMdInferenceSimConfig.model_name
                            }
                        },
                        "garak_config": {
                            "plugins": {
                                # We only run one single probe to speed up computation
                                "probe_spec": "spo.SPOIntent",
                                # Instead of using the default model as a judge, we use a test detector
                                "detector_spec": "always.Fail",
                            },
                            "run": {
                                "generations": 1
                            }
                        }
                    },
                }
            ],
            "experiment": {
                "name": "garak-quick-test",
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

    @pytest.mark.dependency(depends=["garak_submit"])
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
