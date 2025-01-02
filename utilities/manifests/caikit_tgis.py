CAIKIT_TGIS_INFERENCE_CONFIG = {
        "default_query_model": {
            "query_input": "At what temperature does water boil?",
            "query_output": "74 degrees F",
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
    }
