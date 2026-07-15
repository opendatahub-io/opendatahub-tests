from typing import Any

COMPLETION_QUERY: list[dict[str, Any]] = [
    {
        "text": "What are the key benefits of renewable energy sources compared to fossil fuels?",
    },
    {"text": "Translate the following English sentence into Spanish, German, and Mandarin: 'Knowledge is power.'"},
    {"text": "Write a poem about the beauty of the night sky and the mysteries it holds."},
    {"text": "Explain the significance of the Great Wall of China in history and its impact on modern tourism."},
    {"text": "Discuss the ethical implications of using artificial intelligence in healthcare decision-making."},
    {
        "text": "Summarize the main events of the Apollo 11 moon landing and its importance in space exploration history."  # noqa: E122, E501
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
        "text": "Summarize the main events of the Apollo 11 moon landing and its importance in space exploration history."  # noqa: E122, E501
    },
]

PULL_SECRET_ACCESS_TYPE: str = '["Pull"]'
PULL_SECRET_NAME: str = "oci-registry-pull-secret"  # pragma: allowlist secret

PULL_SECRET_ACCESS_TYPE: str = '["Pull"]'
PULL_SECRET_NAME: str = "oci-registry-pull-secret"  # pragma: allowlist secret

SUPPORTED_MODELCAR_REGISTRY_HOSTS: frozenset[str] = frozenset({
    "registry.redhat.io",
    "registry.stage.redhat.io",
    "quay.io",
})
TIMEOUT_20MIN: int = 30 * 60
OPENAI_ENDPOINT_NAME: str = "openai"
AUDIO_FILE_URL: str = (
    "https://raw.githubusercontent.com/realpython/python-speech-recognition/master/audio_files/harvard.wav"
)
AUDIO_FILE_LOCAL_PATH: str = "/tmp/harvard.wav"
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
