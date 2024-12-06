from typing import Any, Dict

from utilities.manifests.models_inqueries import INQUIRIES

RUNTIMES_QUERY_CONFIG: Dict[str, Any] = {
    "caikit-tgis-runtime": {
        "default_query_model": {
            "text": INQUIRIES["water_boil"]["query_text"],
            "model": INQUIRIES["water_boil"]["models"]["flan-t5-small-caikit"],
        },
        "all-tokens": {
            "grpc": {
                "endpoint": "caikit.runtime.Nlp.NlpService/TextGenerationTaskPredict",
                "header": "mm-model-id: $model_name",
                "body": '{"text": "$query_text"}',
                "response_fields_map": {
                    "response_text": "generated_text",
                },
            },
            "http": {
                "endpoint": "api/v1/task/text-generation",
                "header": "Content-type:application/json",
                "body": '{"model_id": "$model_name","inputs": "$query_text"}',
                "response_fields_map": {
                    "response_text": "generated_text",
                },
            },
        },
        "streaming": {
            "grpc": {
                "endpoint": "caikit.runtime.Nlp.NlpService/ServerStreamingTextGenerationTaskPredict",
                "header": "mm-model-id: $model_name",
                "body": '{"text": "$query_text"}',
                "response_fields_map": {"response_text": "generated_text"},
            },
            "http": {
                "endpoint": "api/v1/task/server-streaming-text-generation",
                "header": "Content-type:application/json",
                "body": '{"model_id": "$model_name","inputs": "$query_text"}',
                "response_fields_map": {"response_text": "generated_text"},
            },
        },
    }
}
