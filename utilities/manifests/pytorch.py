GENERATION_PROTO_FILEPATH: str = "utilities/manifests/text-generation-inference/generation.proto"

PYTORCH_TGIS_INFERENCE_CONFIG = {
    "default_query_model": {
        "query_input": "At what temperature does water boil?",
        "query_output": r'\[{"generatedTokenCount":\d+,"text":".*","inputTokenCount":\d+,"stopReason":"MAX_TOKENS"}\]',
        "use_regex": True
    },
    "all-tokens": {
        "grpc": {
            "endpoint": "fmaas.GenerationService/Generate",
            "header": "mm-model-id: $model_name",
            "body": '{"requests": [{"text":"$query_input"}]}',
            "args": f"-proto {GENERATION_PROTO_FILEPATH}",
            "response_fields_map": {"response_output": "responses"},
        }
    }
}
