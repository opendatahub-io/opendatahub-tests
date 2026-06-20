from ocp_resources.inference_service import InferenceService


def get_isvc_condition_messages(isvc: InferenceService) -> list[str]:
    """Extract message strings from all ISVC status conditions."""
    isvc.reload()
    conditions = getattr(isvc.instance.status, "conditions", None) or []
    messages: list[str] = []
    for condition in conditions:
        msg = getattr(condition, "message", None) or ""
        reason = getattr(condition, "reason", None) or ""
        status = getattr(condition, "status", None) or ""
        combined = f"{reason}: {msg} (status={status})"
        messages.append(combined)
    return messages
