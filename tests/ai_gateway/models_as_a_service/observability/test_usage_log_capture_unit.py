"""Focused checks for Loki's two supported usage-log field locations."""

import pytest

from tests.ai_gateway.models_as_a_service.observability.usage_log_capture import extract_log_fields


@pytest.mark.tier1
def test_extract_log_fields_reads_loki_structured_metadata() -> None:
    """Given a Loki stream with structured metadata, when parsed, then identity fields remain visible."""
    payload = {
        "data": {
            "result": [
                {
                    "stream": {"subscription": "sub-a"},
                    "values": [
                        [
                            "123",
                            "request complete",
                            {"key_name": "test-key", "user_id": "alice", "organization_id": "org-a"},
                        ]
                    ],
                }
            ]
        }
    }

    assert extract_log_fields(payload=payload) == [
        {
            "subscription": "sub-a",
            "key_name": "test-key",
            "user_id": "alice",
            "organization_id": "org-a",
        }
    ]


@pytest.mark.tier1
def test_extract_log_fields_keeps_disabled_fields_absent() -> None:
    """Given a JSON log without identity fields, when parsed, then the fields stay absent."""
    payload = {
        "data": {
            "result": [
                {
                    "stream": {"subscription": "sub-a"},
                    "values": [["123", '{"key_name":"test-key","response_type":"success"}']],
                }
            ]
        }
    }

    assert extract_log_fields(payload=payload) == [
        {"subscription": "sub-a", "key_name": "test-key", "response_type": "success"}
    ]


@pytest.mark.tier1
def test_extract_log_fields_does_not_turn_null_identity_into_a_string() -> None:
    """A null identity must not look like a captured username to the E2E assertion."""
    payload = {
        "data": {
            "result": [
                {
                    "stream": {"subscription": "sub-a"},
                    "values": [["123", '{"user_id":null}', {"organization_id": None}]],
                }
            ]
        }
    }

    assert extract_log_fields(payload=payload) == [{"subscription": "sub-a", "user_id": None, "organization_id": None}]
