from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod
from ocp_resources.self_subject_review import SelfSubjectReview
from ocp_resources.user import User
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)


def get_username(dyn_client: DynamicClient) -> str | None:
    """Gets the username for the client (see kubectl -v8 auth whoami)"""
    username: str | None
    try:
        self_subject_review = SelfSubjectReview(client=dyn_client, name="selfSubjectReview").create()
        assert self_subject_review
        username = self_subject_review.status.userInfo.username
    except NotImplementedError:
        LOGGER.info(
            "SelfSubjectReview not found. Falling back to user.openshift.io/v1/users/~ for OpenShift versions <=4.14"
        )
        user = User(client=dyn_client, name="~").instance
        username = user.get("metadata", {}).get("name", None)

    return username


def get_pod_failure_details(pod: Pod) -> str:
    """
    Collect diagnostic information when pod fails to reach ready state.

    Args:
        pod: The pod instance to diagnose

    Returns:
        Formatted diagnostic information string
    """
    details = []

    pod_status = pod.instance.status
    if not pod_status:
        return "Pod status unavailable"

    # Get pod phase
    details.append(f"Phase: {pod_status.phase}")

    # Get container statuses
    if pod_status.containerStatuses:
        details.append("\nContainer Statuses:")
        for container_status in pod_status.containerStatuses:
            container_name = container_status.name
            ready = container_status.ready

            details.append(f"  - {container_name}: ready={ready}")

            # Check waiting state
            if hasattr(container_status.state, "waiting") and container_status.state.waiting:
                waiting = container_status.state.waiting
                reason = waiting.reason
                message = waiting.message if hasattr(waiting, "message") else ""

                # Categorize common errors
                if reason == "ImagePullBackOff":
                    details.append(
                        f"    ⚠️  ImagePullBackOff: Failed to pull custom image\n"
                        f"    Verify registry access and image URL\n"
                        f"    Message: {message}"
                    )
                elif reason == "CrashLoopBackOff":
                    details.append(
                        f"    ⚠️  CrashLoopBackOff: Container is crashing\n"
                        f"    Check container logs for startup errors\n"
                        f"    Message: {message}"
                    )
                elif reason == "ErrImagePull":
                    details.append(
                        f"    ⚠️  ErrImagePull: Cannot pull image\n"
                        f"    Verify image exists and cluster has pull access\n"
                        f"    Message: {message}"
                    )
                else:
                    details.append(f"    Waiting Reason: {reason}\n    Message: {message}")

            # Check terminated state
            if hasattr(container_status.state, "terminated") and container_status.state.terminated:
                terminated = container_status.state.terminated
                details.append(
                    f"    ⚠️  Container terminated\n"
                    f"    Exit Code: {terminated.exitCode}\n"
                    f"    Reason: {terminated.reason}"
                )

    # Try to get pod logs for main container
    try:
        logs = pod.log(container=pod.instance.spec.containers[0].name, tail_lines=50)
        if logs:
            details.append(f"\nRecent Logs (last 50 lines):\n{logs}")
    except (RuntimeError, AttributeError, KeyError):
        details.append("\n(Could not retrieve pod logs)")

    return "\n".join(details)
