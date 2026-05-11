"""Standalone pytest entry point for EvalHub + Kueue environment checks only."""

import pytest


@pytest.mark.smoke
def test_evalhub_preflight(evalhub_preflight_verified: None) -> None:
    """Given kube context and EvalHub env vars, when preflight runs, then API + Kueue checks pass.

    Run without the full suite::

        uv run pytest tests/eval_hub/test_evalhub_preflight.py -q

    Replaces the removed ``scripts/verify_evalhub_setup.py`` helper.
    """
