VLLM_INFERENCE_CONFIG = {
        "default_query_model": {
            "query_input": [
            {
                "model": "$model_name",
                "$model_name": "At what temperature does Nitrogen boil?",
                "max_tokens": 100,
                "temperature": 0,
            }
        ],
            "query_output": {
                "response_output": '{"model_name":"$model_name","model_version":"1",'
                '"outputs":[{"name":"Plus214_Output_0","shape":[1,10],'
                '"datatype":"FP32","data":[-8.233055114746094,-7.749701976776123,-3.42368221282959,12.363025665283203,-12.079105377197266,17.266592025756836,-10.570975303649902,0.7130775451660156,3.3217146396636963,1.3621217012405396]}]}',
            },
        },
        "completions": {
            "http": {
                "endpoint": "v1/completions",
                "header": "Content-type:application/json",
                "body": '{$query_input}',
                "response_fields_map": {
                    "response_output": "output",
                },
            },
        },
    }
