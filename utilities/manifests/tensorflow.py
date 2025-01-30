TENSORFLOW_INFERENCE_CONFIG = {
        "default_query_model": {
            "query_input": "@utilities/manifests/tensorflow/vehicle-detection-inputs.txt",
            "query_output": r'{"model_name":"${model_name}__isvc-[0-9a-z]+","model_version":"1","outputs":\[{"name":"InceptionResnetV2/AuxLogits/Logits/BiasAdd:0","datatype":"FP32","shape":\[1,1001\],"data":\[.*\]}\]}',
            "use_regex": True
        },
        "infer": {
            "http": {
                "endpoint": "v2/models/$model_name/infer",
                "header": "application/x-www-form-urlencoded",
                "body": '$query_input',
                "response_fields_map": {
                    "response_output": "output",
                },
            },
        },
    }
