CUSTOM_UNITXT_TASK_DATA = {
    "task_list": {
        "custom": {
            "systemPrompts": [
                {"name": "sp_0", "value": "Be concise. At every point give the shortest acceptable answer."}
            ],
            "templates": [
                {
                    "name": "tp_0",
                    "value": '{ "__type__": "input_output_template", '
                    '"input_format": "{text_a_type}: {text_a}\\n'
                    '{text_b_type}: {text_b}", '
                    '"output_format": "{label}", '
                    '"target_prefix": '
                    '"The {type_of_relation} class is ", '
                    '"instruction": "Given a {text_a_type} and {text_b_type} '
                    'classify the {type_of_relation} of the {text_b_type} to one of {classes}.",'
                    ' "postprocessors": [ "processors.take_first_non_empty_line",'
                    ' "processors.lower_case_till_punc" ] }',
                }
            ],
        },
        "taskRecipes": [{"card": {"name": "cards.wnli"}, "systemPrompt": {"ref": "sp_0"}, "template": {"ref": "tp_0"}}],
    }
}

LLMAAJ_TASK_DATA = {
    "task_list": {
        "custom": {
            "templates": [
                {
                    "name": "response_assessment.rating.mt_bench_single_turn",
                    "value": '{\n    "__type__": "input_output_template",\n    "instruction":'
                    ' "Please act as an impartial judge and evaluate the quality of the '
                    "response provided by an AI assistant to the user question displayed below."
                    " Your evaluation should consider factors such as the helpfulness, relevance,"
                    " accuracy, depth, creativity, and level of detail of the response. Begin your"
                    " evaluation by providing a short explanation. Be as objective as possible. "
                    "After providing your explanation, you must rate the response on a scale of 1 to 10"
                    ' by strictly following this format: \\"[[rating]]\\", for example: \\"Rating: '
                    '[[5]]\\".\\n\\n",\n    "input_format": "[Question]\\n{question}\\n\\n[The Start '
                    "of Assistant's Answer]\\n{answer}\\n[The End of Assistant's Answer]\",\n    "
                    '"output_format": "[[{rating}]]",\n    "postprocessors": [\n        '
                    '"processors.extract_mt_bench_rating_judgment"\n    ]\n}\n',
                }
            ],
            "tasks": [
                {
                    "name": "response_assessment.rating.single_turn",
                    "value": '{\n    "__type__": "task",\n    "input_fields": {\n        '
                    '"question": "str",\n        "answer": "str"\n    },\n    '
                    '"outputs": {\n        "rating": "float"\n    },\n    '
                    '"metrics": [\n        "metrics.spearman"\n    ]\n}\n',
                }
            ],
            "metrics": [
                {
                    "name": "llmaaj_metric",
                    "value": '{\n    "__type__": "llm_as_judge",\n    "inference_model": {\n        '
                    '"__type__": "hf_pipeline_based_inference_engine",\n        '
                    '"model_name": "rgeada/tiny-untrained-granite",\n        '
                    '"max_new_tokens": 256,\n "use_fp16": true},\n    '
                    '"template": "templates.response_assessment.rating.mt_bench_single_turn",\n    '
                    '"task": "response_assessment.rating.single_turn",\n    '
                    '"main_score": "mistral_7b_instruct_v0_2_huggingface_template_mt_bench_single_turn"\n}',
                }
            ],
        },
        "taskRecipes": [
            {
                "card": {
                    "custom": '{\n    "__type__": "task_card",\n    "loader": '
                    '{\n        "__type__": "load_hf",\n        '
                    '"path": "OfirArviv/mt_bench_single_score_gpt4_judgement",\n        '
                    '"split": "train"\n    },\n    "preprocess_steps": [\n        '
                    '{\n            "__type__": "rename_splits",\n            '
                    '"mapper": {\n                "train": "test"\n            }\n        },\n        '
                    '{\n            "__type__": "filter_by_condition",\n            '
                    '"values": {\n                "turn": 1\n            },\n            '
                    '"condition": "eq"\n        },\n        {\n            '
                    '"__type__": "filter_by_condition",\n            '
                    '"values": {\n                "reference": "[]"\n            },\n            '
                    '"condition": "eq"\n        },\n        {\n            '
                    '"__type__": "rename",\n            "field_to_field": {\n                '
                    '"model_input": "question",\n                '
                    '"score": "rating",\n                '
                    '"category": "group",\n                '
                    '"model_output": "answer"\n            }\n        },\n        '
                    '{\n            "__type__": "literal_eval",\n            '
                    '"field": "question"\n        },\n        '
                    '{\n            "__type__": "copy",\n            '
                    '"field": "question/0",\n            '
                    '"to_field": "question"\n        },\n        '
                    '{\n            "__type__": "literal_eval",\n            '
                    '"field": "answer"\n        },\n        {\n            '
                    '"__type__": "copy",\n            '
                    '"field": "answer/0",\n            '
                    '"to_field": "answer"\n        }\n    ],\n    '
                    '"task": "tasks.response_assessment.rating.single_turn",\n    '
                    '"templates": [\n        '
                    '"templates.response_assessment.rating.mt_bench_single_turn"\n    ]\n}\n',
                    "template": {"ref": "response_assessment.rating.mt_bench_single_turn"},
                    "metrics": [{"ref": "llmaaj_metric"}],
                }
            }
        ],
    }
}

FLAN_T5_IMAGE: str = (
    "quay.io/trustyai_testing/lmeval-assets-flan-t5-base"
    "@sha256:f7326d5b4069e9aa0b12ab77b1e8aa8dd25dd0bffd77b08fcc84988ea8869f7f"
)

ARC_EASY_DATASET_IMAGE: str = (
    "quay.io/trustyai_testing/lmeval-assets-arc-easy"
    "@sha256:1558997a838f2ac8ecd887b4f77485d810e5120b9f2700ecb71627e37c6d3a1b"
)

LMEVAL_OCI_REPO = "lmeval/offline-oci"
LMEVAL_OCI_TAG = "v1"
