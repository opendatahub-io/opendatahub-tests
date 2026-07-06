import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.prometheus import Prometheus

from tests.model_serving.model_server.llmd.llmd_configs import (
    SingleNodePDFast1Config,
    SingleNodePDFast2Config,
    SingleNodePrefillDecodeConfig,
)
from tests.model_serving.model_server.llmd.utils import (
    assert_kv_transfer,
    get_llmd_pod_by_role,
    get_llmd_workload_pods,
    ns_from_file,
    parse_prompt_tokens,
    scheduler_has_plugin,
    send_chat_completions,
    send_completions,
)

LOGGER = structlog.get_logger(name=__name__)

pytestmark = [pytest.mark.tier2, pytest.mark.llmd_gpu]

NAMESPACE = ns_from_file(file=__file__)

PROMPTS = [
    f"Write a detailed paragraph about topic number {i}. " + text
    for i, text in enumerate([
        "the capital of Italy, including its historical significance, population, major landmarks, the river that flows through the city, famous fountains, ancient structures, the climate and weather patterns throughout different seasons",  # noqa: E501
        "the complete process of photosynthesis, including light-dependent reactions, the Calvin cycle, chlorophyll's role, how plants convert carbon dioxide and water into glucose and oxygen, and how different wavelengths affect photosynthesis rates",  # noqa: E501
        "at least five planets in our solar system, their distance from the Sun, atmospheric composition, number of moons, notable features like rings and storms, how each was discovered, and what space missions have studied them",  # noqa: E501
        "the Python programming language, its naming origin, creation by Guido van Rossum, design philosophy emphasizing readability, popularity in data science, and key differences between Python 2 and Python 3",  # noqa: E501
        "the color blue described for someone who has never seen it, using analogies to sound and touch, how blue is perceived across cultures, its psychological effects, why the sky appears blue, and ocean light absorption",  # noqa: E501
        "the Pacific Ocean as Earth's largest ocean, covering one third of the surface, extending from Arctic to Antarctic, bordered by Asia and the Americas, containing the Mariana Trench as the deepest known point",  # noqa: E501
        "binary search algorithms that divide sorted arrays in half, comparing targets with middle elements, eliminating half the search space per comparison, achieving logarithmic complexity, faster than linear search for large datasets",  # noqa: E501
        "water's chemical formula H2O with two hydrogen atoms bonded to oxygen, its essential role for life, unique properties like high specific heat capacity, surface tension, and anomalous expansion when freezing",  # noqa: E501
        "Mount Everest in the Mahalangur Himal sub-range of the Himalayas, on the Nepal-Tibet border, standing at 8849 meters, the highest point on Earth, first ascended by Hillary and Norgay in nineteen fifty three",  # noqa: E501
        "the speed of light at approximately 299792458 meters per second in vacuum, often rounded to 300000 kilometers per second, fundamental to Einstein's special relativity and E equals mc squared",  # noqa: E501
        "the history of the Roman Empire from its founding through the republic period to the imperial era, including key emperors, military conquests, architectural achievements, and eventual decline and fall",  # noqa: E501
        "how neural networks function in machine learning, including layers of interconnected nodes, activation functions, backpropagation, gradient descent optimization, and applications in image recognition and language processing",  # noqa: E501
        "the Amazon rainforest as the largest tropical forest, covering nine countries in South America, containing millions of species, its role in global climate regulation, and threats from deforestation and development",  # noqa: E501
        "the periodic table of elements organized by atomic number, electron configuration, and chemical properties, including the contributions of Mendeleev, modern additions, and how elements are grouped into families",  # noqa: E501
        "Shakespeare's influence on English literature and language, his major plays and sonnets, the Globe Theatre, Elizabethan era context, and how his works continue to be performed and adapted worldwide today",  # noqa: E501
        "the human cardiovascular system including the heart's four chambers, blood circulation through arteries and veins, the role of red and white blood cells, blood pressure regulation, and common cardiovascular diseases",  # noqa: E501
        "the theory of plate tectonics explaining Earth's lithosphere divided into moving plates, continental drift evidence, seafloor spreading, subduction zones, and how plate movements cause earthquakes and volcanic eruptions",  # noqa: E501
        "the invention and evolution of the internet from ARPANET to the modern world wide web, including TCP/IP protocols, web browsers, search engines, social media, cloud computing, and mobile connectivity",  # noqa: E501
        "quantum mechanics fundamentals including wave-particle duality, the uncertainty principle, quantum superposition and entanglement, Schrodinger's equation, and applications in computing and cryptography",  # noqa: E501
        "the French Revolution from its causes including social inequality and financial crisis, through the storming of the Bastille, the Reign of Terror, Napoleon's rise, and its lasting impact on democracy",  # noqa: E501
    ])
]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [
        pytest.param({"name": NAMESPACE}, SingleNodePrefillDecodeConfig, id="standard"),
        pytest.param(
            {"name": NAMESPACE},
            SingleNodePDFast1Config,
            id="fast-1",
            marks=pytest.mark.fast_vllm,
        ),
        pytest.param(
            {"name": NAMESPACE},
            SingleNodePDFast2Config,
            id="fast-2",
            marks=pytest.mark.fast_vllm,
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("skip_if_disconnected")
class TestSingleNodePrefillDecode:
    """Single-node P/D LLMISVC with controller-generated scheduler config.

    Tests the disaggregated Prefill/Decode topology:
    - Controller creates {name}-kserve (decode, llm-d.ai/role=decode) and
      {name}-kserve-prefill (prefill, llm-d.ai/role=prefill) Deployments
    - Decode pod has llm-d-routing-sidecar init container; prefill pod does not
    - No custom scheduler config: controller generates the full P/D
      EndpointPickerConfig with disaggregation plugins when spec.prefill != nil
    - KV transfer uses NixlConnector over UCX
    """

    def test_prefill_decode_topology(
        self,
        unprivileged_client: DynamicClient,
        llmisvc: LLMInferenceService,
    ):
        """Assert the P/D deployment topology created by the controller."""

        workload_pods = get_llmd_workload_pods(client=unprivileged_client, llmisvc=llmisvc)
        roles = {}
        for pod in workload_pods:
            role = pod.instance.metadata.labels.get("llm-d.ai/role", "unknown")
            roles.setdefault(role, []).append(pod.name)

        # Verify exactly 1 decode pod and 1 prefill pod exist with correct role labels
        assert sorted(roles.keys()) == ["decode", "prefill"], (
            f"Expected exactly 'decode' and 'prefill' roles, got: {dict(roles)}"
        )
        assert len(roles["decode"]) == 1, f"Expected 1 decode pod, got {len(roles['decode'])}: {roles['decode']}"
        assert len(roles["prefill"]) == 1, f"Expected 1 prefill pod, got {len(roles['prefill'])}: {roles['prefill']}"

        # Verify decode pod has llm-d-routing-sidecar init container (Kubernetes native sidecar)
        decode_pod = get_llmd_pod_by_role(client=unprivileged_client, llmisvc=llmisvc, role="decode")
        init_containers = [container.name for container in (decode_pod.instance.spec.get("initContainers") or [])]
        assert "llm-d-routing-sidecar" in init_containers, (
            f"Decode pod missing llm-d-routing-sidecar init container, found: {init_containers}"
        )

        # Verify prefill pod has no sidecar
        prefill_pod = get_llmd_pod_by_role(client=unprivileged_client, llmisvc=llmisvc, role="prefill")
        prefill_init_containers = [c.name for c in (prefill_pod.instance.spec.get("initContainers") or [])]
        assert "llm-d-routing-sidecar" not in prefill_init_containers, (
            f"Prefill pod should NOT have llm-d-routing-sidecar, found: {prefill_init_containers}"
        )

        # Verify scheduler config contains all expected P/D plugins
        for expected_plugin in [
            "disagg-headers-handler",
            "prefill-filter",
            "decode-filter",
            "always-disagg-pd-decider",
            "disagg-profile-handler",
        ]:
            assert scheduler_has_plugin(client=unprivileged_client, llmisvc=llmisvc, plugin_name=expected_plugin), (
                f"Scheduler config missing expected P/D plugin: {expected_plugin}"
            )

    def test_kv_transfer(
        self,
        unprivileged_client: DynamicClient,
        llmisvc: LLMInferenceService,
        prometheus: Prometheus,
    ):
        """Test inference requests go through P/D with KV transfer.

        Sends inference requests using both /v1/completions and /v1/chat/completions,
        then asserts KV transfer metrics to verify disaggregation is working.
        """

        total_prompt_tokens = 0
        for i, prompt in enumerate(PROMPTS):
            if i < len(PROMPTS) // 2:
                status, body = send_completions(llmisvc=llmisvc, prompt=prompt)
                endpoint = "completions"
            else:
                status, body = send_chat_completions(llmisvc=llmisvc, prompt=prompt)
                endpoint = "chat/completions"
            assert status == 200, f"Request {i + 1}/{len(PROMPTS)}: expected 200, got {status}: {body}"
            tokens = parse_prompt_tokens(response_body=body)
            LOGGER.info(f"Request {i + 1}/{len(PROMPTS)} [{endpoint}] prompt_tokens={tokens}")
            total_prompt_tokens += tokens

        LOGGER.info(f"Total prompt tokens from {len(PROMPTS)} requests: {total_prompt_tokens}")

        # Verify KV transfer metrics (retries up to 120s for Prometheus scrape delay)
        assert_kv_transfer(
            prometheus=prometheus,
            unprivileged_client=unprivileged_client,
            llmisvc=llmisvc,
            expected_transferred_tokens=total_prompt_tokens,
            num_requests=len(PROMPTS),
        )
