from __future__ import annotations

from kubernetes.dynamic import DynamicClient, Resource, ResourceInstance


def get_username(dyn_client: DynamicClient) -> str:
    """Gets the username for the client (see kubectl -v8 auth whoami)"""
    self_subject_review_resource: Resource = dyn_client.resources.get(
        api_version="authentication.k8s.io/v1", kind="SelfSubjectReview"
    )
    self_subject_review: ResourceInstance = dyn_client.create(self_subject_review_resource)
    username: str = self_subject_review.status.userInfo.username
    return username
