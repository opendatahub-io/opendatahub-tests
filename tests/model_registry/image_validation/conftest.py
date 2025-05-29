import pytest
from typing import Set
from kubernetes.dynamic import DynamicClient
from utilities.operator_utils import get_csv_related_images


@pytest.fixture()
def related_images_refs(admin_client: DynamicClient) -> Set[str]:
    related_images = get_csv_related_images(admin_client=admin_client)
    related_images_refs = {img["image"] for img in related_images}
    return related_images_refs
