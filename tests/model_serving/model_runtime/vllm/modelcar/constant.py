from typing import Any

COMPLETION_QUERY: list[dict[str, Any]] = [
    {
        "text": "What are the key benefits of renewable energy sources compared to fossil fuels?",
        "keywords": [
            "renewable",
            "energy",
            "solar",
            "wind",
            "fossil",
            "carbon",
            "emission",
            "sustainable",
            "clean",
            "pollution",
        ],
    },
    {
        "text": "Translate the following English sentence into Spanish, German, and Mandarin: 'Knowledge is power.'",
        "keywords": [
            "spanish",
            "german",
            "mandarin",
            "translation",
            "conocimiento",
            "poder",
            "wissen",
            "macht",
            "knowledge",
            "power",
        ],
    },
    {
        "text": "Write a poem about the beauty of the night sky and the mysteries it holds.",
        "keywords": [
            "night",
            "sky",
            "stars",
            "moon",
            "dark",
            "mystery",
            "universe",
            "cosmic",
            "celestial",
            "heaven",
            "galaxy",
        ],
    },
    {
        "text": "Explain the significance of the Great Wall of China in history and its impact on modern tourism.",
        "keywords": [
            "great wall",
            "china",
            "chinese",
            "history",
            "tourism",
            "dynasty",
            "wall",
            "monument",
            "ancient",
            "visitor",
        ],
    },
    {
        "text": "Discuss the ethical implications of using artificial intelligence in healthcare decision-making.",
        "keywords": [
            "ethical",
            "ethics",
            "ai",
            "artificial intelligence",
            "healthcare",
            "medical",
            "decision",
            "patient",
            "privacy",
            "bias",
        ],
    },
    {
        "text": "Summarize the main events of the Apollo 11 moon landing and its importance in space exploration history.",  # noqa: E501
        "keywords": [
            "apollo",
            "moon",
            "landing",
            "nasa",
            "armstrong",
            "space",
            "astronaut",
            "lunar",
            "exploration",
            "1969",
        ],
    },
]

EMBEDDING_QUERY: list[dict[str, str]] = [
    {
        "text": "What are the key benefits of renewable energy sources compared to fossil fuels?",
    },
    {"text": "Translate the following English sentence into Spanish, German, and Mandarin: 'Knowledge is power.'"},
    {"text": "Write a poem about the beauty of the night sky and the mysteries it holds."},
    {"text": "Explain the significance of the Great Wall of China in history and its impact on modern tourism."},
    {"text": "Discuss the ethical implications of using artificial intelligence in healthcare decision-making."},
    {
        "text": "Summarize the main events of the Apollo 11 moon landing and its importance in space exploration history."  # noqa: E501
    },
]

PULL_SECRET_ACCESS_TYPE: str = '["Pull"]'
PULL_SECRET_NAME: str = "oci-registry-pull-secret"  # pragma: allowlist secret

SUPPORTED_MODELCAR_REGISTRY_HOSTS: frozenset[str] = frozenset({
    "registry.redhat.io",
    "registry.stage.redhat.io",
    "quay.io",
})
SPYRE_INFERENCE_SERVICE_PORT: int = 8000
TIMEOUT_20MIN: int = 30 * 60
OPENAI_ENDPOINT_NAME: str = "openai"
AUDIO_FILE_URL: str = (
    "https://raw.githubusercontent.com/realpython/python-speech-recognition/master/audio_files/harvard.wav"
)
AUDIO_FILE_LOCAL_PATH: str = "/tmp/harvard.wav"
# Known phrases from harvard.wav used for generic audio transcription smoke validation.
AUDIO_TRANSCRIPTION_KEYWORDS: list[str] = [
    "beer",
    "pickle",
    "tacos",
    "stale",
    "ham",
    "zest",
    "odor",
    "health",
]
