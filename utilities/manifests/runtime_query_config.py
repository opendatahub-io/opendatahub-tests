from typing import Any, Dict

from utilities.constants import ModelAndFormat, ModelName, RuntimeQueryKeys
from utilities.manifests.models_inqueries import INQUIRIES

GENERATION_PROTO_FILEPATH: str = "utilities/manifests/text-generation-inference/generation.proto"

RUNTIMES_QUERY_CONFIG: Dict[str, Any] = {
    RuntimeQueryKeys.CAIKIT_TGIS_RUNTIME: {
        "default_query_model": {
            "input": INQUIRIES["water_boil"]["query_input"],
            "model": INQUIRIES["water_boil"]["models"][ModelAndFormat.FLAN_T5_SMALL_CAIKIT],
        },
        "all-tokens": {
            "grpc": {
                "endpoint": "caikit.runtime.Nlp.NlpService/TextGenerationTaskPredict",
                "header": "mm-model-id: $model_name",
                "body": '{"text": "$query_input"}',
                "response_fields_map": {
                    "response_output": "generated_text",
                },
            },
            "http": {
                "endpoint": "api/v1/task/text-generation",
                "header": "Content-type:application/json",
                "body": '{"model_id": "$model_name","inputs": "$query_input"}',
                "response_fields_map": {
                    "response_output": "generated_text",
                },
            },
        },
        "streaming": {
            "grpc": {
                "endpoint": "caikit.runtime.Nlp.NlpService/ServerStreamingTextGenerationTaskPredict",
                "header": "mm-model-id: $model_name",
                "body": '{"text": "$query_input"}',
                "response_fields_map": {"response_output": "generated_text"},
            },
            "http": {
                "endpoint": "api/v1/task/server-streaming-text-generation",
                "header": "Content-type:application/json",
                "body": '{"model_id": "$model_name","inputs": "$query_input"}',
                "response_fields_map": {"response_output": "generated_text"},
            },
        },
    },
    RuntimeQueryKeys.OPENVINO_RUNTIME: {
        "default_query_model": {
            "input": INQUIRIES["infer"]["query_input"],
            "model": INQUIRIES["infer"]["models"][ModelAndFormat.OPENVINO_IR],
        },
        "infer": {
            "http": {
                "endpoint": "v2/models/$model_name/infer",
                "header": "Content-type:application/json",
                "body": '{"inputs": $query_input}',
                "response_fields_map": {
                    "response_output": "output",
                },
            },
        },
    },
    RuntimeQueryKeys.OPENVINO_KSERVE_RUNTIME: {
        "default_query_model": {
            "input": INQUIRIES["infer"]["query_input"],
            "model": INQUIRIES["infer"]["models"][ModelAndFormat.KSERVE_OPENVINO_IR],
        },
        "infer": {
            "http": {
                "endpoint": "v2/models/$model_name/infer",
                "header": "Content-type:application/json",
                "body": '{"inputs": $query_input}',
                "response_fields_map": {
                    "response_output": "output",
                },
            },
        },
    },
    RuntimeQueryKeys.TGIS_RUNTIME: {
        "default_query_model": {
            "input": INQUIRIES["water_boil"]["query_input"],
            "model": INQUIRIES["water_boil"]["models"][ModelName.FLAN_T5_SMALL_HF],
        },
        "all-tokens": {
            "grpc": {
                "endpoint": "fmaas.GenerationService/Generate",
                "header": "mm-model-id: $model_name",
                "body": '{"requests": [{"text":"$query_input"}]}',
                "args": f"-proto {GENERATION_PROTO_FILEPATH}",
                "response_fields_map": {"response": "responses", "response_output": "text"},
            }
        },
        "streaming": {
            "grpc": {
                "endpoint": "fmaas.GenerationService/GenerateStream",
                "header": "mm-model-id: $model_name",
                "body": '{"requests": [{"text":"$query_input"}]}',
                "args": f"-proto {GENERATION_PROTO_FILEPATH}",
                "response_fields_map": {"response": "responses", "response_output": "text"},
            }
        },
    },
}
