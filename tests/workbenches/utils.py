from kubernetes.dynamic import DynamicClient
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
