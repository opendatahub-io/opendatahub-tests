import pytest

from tests.workbenches.notebooks_server.controller.utils import MutatingWebhookConfiguration


@pytest.mark.usefixtures("capture_notebook_baseline")
class TestPreUpgradeWebhook:
    """Verify notebook controller webhook is available before upgrade.

    Steps:
        1. Verify the MutatingWebhookConfiguration exists.
        2. Verify it has at least one webhook rule configured.
    """

    @pytest.mark.pre_upgrade
    def test_webhook_exists_before_upgrade(
        self,
        notebook_mutating_webhook: MutatingWebhookConfiguration,
    ) -> None:
        """Given the notebook controller is deployed,
        When we check for the MutatingWebhookConfiguration,
        Then it should exist on the cluster.
        """
        assert notebook_mutating_webhook.exists, (
            f"MutatingWebhookConfiguration '{notebook_mutating_webhook.name}' not found. "
            f"The notebook controller webhook is required for notebook injection."
        )

    @pytest.mark.pre_upgrade
    def test_webhook_has_rules_before_upgrade(
        self,
        notebook_mutating_webhook: MutatingWebhookConfiguration,
    ) -> None:
        """Given the MutatingWebhookConfiguration exists,
        When we inspect its webhooks list,
        Then at least one webhook rule should be configured.
        """
        assert notebook_mutating_webhook.exists, (
            f"MutatingWebhookConfiguration '{notebook_mutating_webhook.name}' not found. "
            f"The notebook controller webhook is required for notebook injection."
        )
        webhooks = notebook_mutating_webhook.instance.get("webhooks", [])
        assert webhooks, (
            f"MutatingWebhookConfiguration '{notebook_mutating_webhook.name}' has no webhook rules configured."
        )


class TestPostUpgradeWebhook:
    """Verify notebook controller webhook survived the platform upgrade.

    Steps:
        1. Verify the MutatingWebhookConfiguration still exists after upgrade.
        2. Verify it still has at least one webhook rule configured.
    """

    @pytest.mark.post_upgrade
    def test_webhook_exists_after_upgrade(
        self,
        notebook_mutating_webhook: MutatingWebhookConfiguration,
    ) -> None:
        """Given the MutatingWebhookConfiguration existed before upgrade,
        When the upgrade completes,
        Then it should still exist on the cluster.
        """
        assert notebook_mutating_webhook.exists, (
            f"MutatingWebhookConfiguration '{notebook_mutating_webhook.name}' "
            f"no longer exists after upgrade. The notebook controller webhook is critical "
            f"for notebook injection."
        )

    @pytest.mark.post_upgrade
    def test_webhook_has_rules_after_upgrade(
        self,
        notebook_mutating_webhook: MutatingWebhookConfiguration,
    ) -> None:
        """Given the MutatingWebhookConfiguration had webhook rules before upgrade,
        When the upgrade completes,
        Then at least one webhook rule should still be configured.
        """
        assert notebook_mutating_webhook.exists, (
            f"MutatingWebhookConfiguration '{notebook_mutating_webhook.name}' "
            f"no longer exists after upgrade. The notebook controller webhook is critical "
            f"for notebook injection."
        )
        webhooks = notebook_mutating_webhook.instance.get("webhooks", [])
        assert webhooks, (
            f"MutatingWebhookConfiguration '{notebook_mutating_webhook.name}' lost all webhook rules after upgrade."
        )
