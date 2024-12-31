from typing import Any, Dict

from utilities.constants import ModelAndFormat, ModelName

INQUIRIES: Dict[str, Any] = {
    "water_boil": {
        "query_input": "At what temperature does water boil?",
        "models": {
            ModelAndFormat.FLAN_T5_SMALL_CAIKIT: {
                "response_tokens": 5,
                "response_output": "74 degrees F",
            },
            ModelName.FLAN_T5_SMALL_HF: {
                "response_tokens": 5,
                "response_output": "74 degrees F",
            },
        },
    },
    "infer": {
        "query_input": [
            {
                "name": "Func/StatefulPartitionedCall/input/_0:0",
                "shape": [1, 30],
                "datatype": "FP32",
                "data": [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                ],
            }
        ],
        "models": {
            ModelAndFormat.OPENVINO_IR: {
                "response_output": '{"model_name":"http-openvino__isvc-ac836837df","model_version":"1",'
                '"outputs":[{"name":"Func/StatefulPartitionedCall/output/_13:0","datatype":"FP32",'
                '"shape":[1,1],"data":[0]}]}',
            },
            ModelAndFormat.KSERVE_OPENVINO_IR: {
                "response_output": '{"model_name":"http-openvino","model_version":"1",'
                '"outputs":[{"name":"Func/StatefulPartitionedCall/output/_13:0","shape":[1,1],'
                '"datatype":"FP32","data":[0.0]}]}',
            },
        },
    },
}
