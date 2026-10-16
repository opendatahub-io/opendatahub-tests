"""Shared helpers for workbenches tests."""

# TestOps skip-allowlist prefixes. Must be the first token of pytest.skip() reasons
# so JUnit ``<skipped message>`` can be classified. See rhods-qe-tools
# jira/skip_allowlist.yaml (prefix has no trailing colon; a colon may follow).
SKIP_EXPECTED_CONNECTED_ONLY = "SKIP_EXPECTED_CONNECTED_ONLY"
SKIP_EXPECTED_DOWNSTREAM_ONLY = "SKIP_EXPECTED_DOWNSTREAM_ONLY"
SKIP_EXPECTED_EUS_ONLY = "SKIP_EXPECTED_EUS_ONLY"
SKIP_EXPECTED_IMAGE_BUMP_ONLY = "SKIP_EXPECTED_IMAGE_BUMP_ONLY"
SKIP_EXPECTED_TEST_DEPENDENCY_FAILURE = "SKIP_EXPECTED_TEST_DEPENDENCY_FAILURE"


def expected_skip(prefix: str, detail: str) -> str:
    """Build a skip reason that starts with a TestOps allow-list prefix.

    Args:
        prefix: Allow-list token such as ``SKIP_EXPECTED_DOWNSTREAM_ONLY``.
        detail: Human-readable explanation that follows the prefix.

    Returns:
        A skip message of the form ``{prefix}: {detail}``.
    """
    return f"{prefix}: {detail}"
