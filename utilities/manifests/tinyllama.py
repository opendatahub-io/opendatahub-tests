TINYLLAMA_INFERENCE_CONFIG = {
    "default_query_model": {
        "query_input": "What is the capital of France?",
        "query_output": r'.*[Pp][Aa][Rr][Ii][Ss].*',
        "use_regex": True,
    },
    "chat_completions": {
        "http": {
            "endpoint": "v1/chat/completions",
            "header": "Content-Type:application/json",
            "body": '{"model": "$model_name", "messages": [{"role": "user", "content": "$query_input"}], "max_tokens": 50, "temperature": 0.0, "stream": false}',
            "response_fields_map": {
                "response_output": "output",
            },
        },
    },
    "completions": {
        "http": {
            "endpoint": "v1/completions",
            "header": "Content-Type:application/json",
            "body": '{"model": "$model_name", "prompt": "$query_input", "max_tokens": 50, "temperature": 0.0}',
            "response_fields_map": {
                "response_output": "output",
            },
        },
    },
}
