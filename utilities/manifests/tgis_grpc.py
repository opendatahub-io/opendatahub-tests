GENERATION_PROTO_FILEPATH: str = "utilities/manifests/text-generation-inference/generation.proto"

TGIS_INFERENCE_CONFIG = {
    "default_query_model": {
        "query_input": "At what temperature does water boil?",
        "query_output": "74 degrees F",
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
}
