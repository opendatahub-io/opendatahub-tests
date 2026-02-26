"""ImageStream health checks for workbench-related images."""

from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.image_stream import ImageStream
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger

pytestmark = [pytest.mark.smoke]
LOGGER = get_logger(name=__name__)
IMPORT_SUCCESS_CONDITION_TYPE = "ImportSuccess"


def _validate_imagestream_tag_health(
    imagestream_name: str,
    tag_name: str,
    tag_data: dict[str, Any],
) -> list[str]:
    """
    Validate one ImageStream status tag and return all discovered errors.

    A tag is considered healthy when it has at least one resolved item in
    `status.tags[].items`, each item points to a digest-based image reference,
    and an optional `ImportSuccess` condition (when present) is `True`.

    Args:
        imagestream_name: Name of the parent ImageStream (for error reporting).
        tag_name: Name of the ImageStream tag being validated.
        tag_data: Raw `status.tags[]` payload for the tag.

    Returns:
        List of validation error messages. Empty list means the tag is healthy.
    """
    errors: list[str] = []

    raw_tag_items = tag_data.get("items")
    tag_items = raw_tag_items if isinstance(raw_tag_items, list) else []
    import_conditions = [
        condition
        for condition in (tag_data.get("conditions") or [])
        if condition.get("type") == IMPORT_SUCCESS_CONDITION_TYPE
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
    """
    Validate ImageStreams selected by label and fail the test if unhealthy.

    This helper enforces:
    - expected ImageStream count for the selector
    - every tag declared in `spec.tags` appears in `status.tags`
    - per-tag resolution/import checks via `_validate_imagestream_tag_health`

    Args:
        imagestreams: ImageStreams fetched for the label selector.
        label_selector: Label selector used to fetch ImageStreams.
        expected_count: Expected number of matching ImageStreams.

    Raises:
        pytest.fail: When any validation error is found.
    """
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
        errors.extend([
            f"ImageStream {imagestream_name} spec tag {missing_tag} is missing from status.tags "
            f"(label selector: {label_selector})"
            for missing_tag in missing_status_tags
        ])

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
        pytest.fail("\n".join(errors))


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
