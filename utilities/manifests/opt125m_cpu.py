OPT125M_CPU_INFERENCE_CONFIG = {
    "default_query_model": {
        "query_input": "What is the boiling point of water?",
        "query_output": r'.*',
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
