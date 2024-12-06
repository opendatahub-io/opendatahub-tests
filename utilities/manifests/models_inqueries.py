from typing import Any, Dict

INQUIRIES: Dict[str, Any] = {
    "water_boil": {
        "query_text": "At what temperature does water boil?",
        "models": {
            "flan-t5-small-caikit": {
                "response_tokens": 5,
                "response_text": "74 degrees F",
            },
        },
    },
}
