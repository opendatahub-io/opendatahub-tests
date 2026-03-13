class OpenAIEnpoints:
    CHAT_COMPLETIONS: str = "/v1/chat/completions"
    COMPLETIONS: str = "/v1/completions"
    EMBEDDINGS: str = "/v1/embeddings"
    MODELS_INFO: str = "/v1/models"
    METRICS: str = "/metrics/"
    AUDIO_TRANSCRIPTION: str = "/v1/audio/transcriptions"


class RestHeader:
    HEADERS: dict[str, str] = {"Content-Type": "application/json"}  # noqa: RUF012
