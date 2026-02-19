"""ImageStream health checks for workbench-related images."""

from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.resource import NamespacedResource
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger


class ImageStream(NamespacedResource):
    """OpenShift ImageStream resource."""

    api_group: str = "image.openshift.io"


pytestmark = [pytest.mark.smoke]
LOGGER = get_logger(name=__name__)


def _validate_imagestream_tag_health(
    imagestream_name: str,
    tag_name: str,
    tag_data: dict[str, Any],
) -> list[str]:
    """Validate image resolution and import state for one ImageStream tag."""
    errors: list[str] = []

    raw_tag_items = tag_data.get("items")
    tag_items = raw_tag_items if isinstance(raw_tag_items, list) else []
    import_conditions = [
        condition for condition in (tag_data.get("conditions") or []) if condition.get("type") == "ImportSuccess"
    ]
    latest_import_condition = (
        max(import_conditions, key=lambda condition: condition.get("generation", -1)) if import_conditions else None
    )
    import_status = latest_import_condition.get("status") if latest_import_condition else "N/A"
    LOGGER.info(
        f"Checked ImageStream tag {imagestream_name}:{tag_name} "
        f"(items_count={len(tag_items)}, import_success={import_status})"
    )

    # A tag is considered unresolved if no image item exists.
    # In that case we expect an ImportSuccess=False condition to explain the failure reason.
    if not tag_items:
        failure_details = (
            "no ImportSuccess condition was reported"
            if not latest_import_condition
            else (
                f"status={latest_import_condition.get('status')}, "
                f"reason={latest_import_condition.get('reason')}, "
                f"message={latest_import_condition.get('message')}"
            )
        )
        errors.append(
            f"ImageStream {imagestream_name} tag {tag_name} has unresolved status.tags.items; "
            f"ImportSuccess details: {failure_details}"
        )
        return errors

    for item_index, item in enumerate(tag_items):
        docker_image_reference = str(item.get("dockerImageReference", ""))
        if "@sha256:" not in docker_image_reference:
            errors.append(
                f"ImageStream {imagestream_name} tag {tag_name} item #{item_index} "
                "has unresolved dockerImageReference: "
                f"{docker_image_reference}"
            )

        image_reference = str(item.get("image", ""))
        if not image_reference.startswith("sha256:"):
            errors.append(
                f"ImageStream {imagestream_name} tag {tag_name} item #{item_index} has unresolved image reference: "
                f"{image_reference}"
            )

    # If the tag resolved to items but ImportSuccess exists and reports failure, this is still an error.
    if latest_import_condition and latest_import_condition.get("status") != "True":
        errors.append(
            f"ImageStream {imagestream_name} tag {tag_name} has resolved items but ImportSuccess is not True: "
            f"status={latest_import_condition.get('status')}, "
            f"reason={latest_import_condition.get('reason')}, "
            f"message={latest_import_condition.get('message')}"
        )

    return errors


def _validate_imagestreams_with_label(
    imagestreams: list[ImageStream],
    label_selector: str,
    expected_count: int,
) -> None:
    """Validate ImageStream count and status.tag health for a label selector."""
    errors: list[str] = []
    actual_count = len(imagestreams)
    LOGGER.info(
        f"Checking ImageStreams for label selector '{label_selector}': "
        f"expected_count={expected_count}, actual_count={actual_count}"
    )
    if imagestreams:
        LOGGER.info(
            f"ImageStreams matched for '{label_selector}': {', '.join(sorted(is_obj.name for is_obj in imagestreams))}"
        )
    if actual_count != expected_count:
        imagestream_names = ", ".join(sorted(imagestream.name for imagestream in imagestreams))
        errors.append(
            f"Expected {expected_count} ImageStreams with label '{label_selector}', found {actual_count}. "
            f"Found: [{imagestream_names}]"
        )

    for imagestream in imagestreams:
        imagestream_data: dict[str, Any] = imagestream.instance.to_dict()
        imagestream_name = imagestream_data.get("metadata", {}).get("name", imagestream.name)
        LOGGER.info(f"Validating ImageStream {imagestream_name} (label selector: {label_selector})")

        spec_tag_names = {
            str(spec_tag.get("name"))
            for spec_tag in imagestream_data.get("spec", {}).get("tags", [])
            if spec_tag.get("name")
        }
        status_tags = imagestream_data.get("status", {}).get("tags", [])
        status_tag_names = {str(status_tag.get("tag")) for status_tag in status_tags if status_tag.get("tag")}

        missing_status_tags = sorted(spec_tag_names - status_tag_names)
        LOGGER.info(
            f"ImageStream {imagestream_name} tag coverage: "
            f"spec_tags={sorted(spec_tag_names)}, status_tags={sorted(status_tag_names)}"
        )
        for missing_tag in missing_status_tags:
            errors.append(
                f"ImageStream {imagestream_name} spec tag {missing_tag} is missing from status.tags "
                f"(label selector: {label_selector})"
            )

        for status_tag in status_tags:
            tag_name = str(status_tag.get("tag", "<missing-tag-name>"))
            errors.extend(
                _validate_imagestream_tag_health(
                    imagestream_name=imagestream_name,
                    tag_name=tag_name,
                    tag_data=status_tag,
                )
            )

    if errors:
        raise AssertionError("\n".join(errors))


@pytest.mark.parametrize(
    "label_selector, expected_imagestream_count",
    [
        pytest.param("opendatahub.io/notebook-image=true", 11, id="notebook_imagestreams"),
        pytest.param("opendatahub.io/runtime-image=true", 7, id="runtime_imagestreams"),
    ],
)
def test_workbench_imagestreams_health(
    admin_client: DynamicClient,
    label_selector: str,
    expected_imagestream_count: int,
) -> None:
    """
    Given workbench-related ImageStreams in the applications namespace.
    When ImageStreams are listed by the expected workbench labels.
    Then all expected ImageStreams exist and each tag is imported and resolved successfully.
    """
    imagestreams = list(
        ImageStream.get(
            client=admin_client,
            namespace=py_config["applications_namespace"],
            label_selector=label_selector,
        )
    )

    _validate_imagestreams_with_label(
        imagestreams=imagestreams,
        label_selector=label_selector,
        expected_count=expected_imagestream_count,
    )
