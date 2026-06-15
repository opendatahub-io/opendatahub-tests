"""ImageStream health checks for workbench-related images."""

from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.image_stream import ImageStream
from packaging.version import InvalidVersion, Version
from pytest_testconfig import config as py_config

pytestmark = [pytest.mark.smoke]
LOGGER = structlog.get_logger(name=__name__)
IMPORT_SUCCESS_CONDITION_TYPE = "ImportSuccess"
LATEST_TAGS_COUNT = 2


RHOAI_VERSIONING_START = Version(version="3.4")


def _tag_sort_key(version: Version) -> tuple[int, Version]:
    """Return a sort key reflecting the chronological ordering of ImageStream tag versions.

    Three naming schemes have been used historically:
      - Tier 2 (newest): RHOAI major.minor >= 3.4 (e.g. 3.4, 3.5)
      - Tier 1 (middle): year-based versions with major >= 2000 (e.g. 2024.1, 2025.2)
      - Tier 0 (oldest): legacy versions before the year scheme (e.g. 1.2)
    """
    if version >= RHOAI_VERSIONING_START and version.major < 2000:
        return (2, version)
    if version.major >= 2000:
        return (1, version)
    return (0, version)


def _get_latest_n_tag_names(imagestream_data: dict[str, Any], n: int = LATEST_TAGS_COUNT) -> set[str]:
    """Return the names of the N most recent tags by version.

    Handles mixed version schemes: RHOAI versions (3.4, 3.5) are ranked above
    legacy year-based versions (2025.2, 2025.1, 2024.2, ...).

    Args:
        imagestream_data: Raw ImageStream dict (from `instance.to_dict()`).
        n: Number of latest tags to return.

    Returns:
        Set of tag name strings for the N highest-versioned tags.
    """
    spec_tags = imagestream_data.get("spec", {}).get("tags", [])
    all_tag_names = {str(tag.get("name")) for tag in spec_tags if tag.get("name")}
    versioned: list[tuple[tuple[int, Version], str]] = []
    for tag in spec_tags:
        name = tag.get("name")
        if not name:
            continue
        try:
            version = Version(version=name)
        except InvalidVersion:
            continue
        versioned.append((_tag_sort_key(version=version), name))
    versioned.sort(reverse=True)
    if not versioned:
        # Fail safe: never return an empty filter that disables validation.
        return all_tag_names
    return {name for _, name in versioned[:n]}


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
    tags_to_validate: set[str] | None = None,
) -> None:
    """
    Validate ImageStreams selected by label and fail the test if unhealthy.

    This helper enforces:
    - expected ImageStream count for the selector
    - every tag declared in `spec.tags` appears in `status.tags` (scoped to ``tags_to_validate`` when provided)
    - per-tag resolution/import checks via `_validate_imagestream_tag_health`

    Args:
        imagestreams: ImageStreams fetched for the label selector.
        label_selector: Label selector used to fetch ImageStreams.
        expected_count: Expected number of matching ImageStreams.
        tags_to_validate: If provided, only validate these tag names. When ``None``, all tags are validated.

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
        if tags_to_validate is not None:
            spec_tag_names &= tags_to_validate

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
            if tags_to_validate is not None and tag_name not in tags_to_validate:
                continue
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
    "label_selector, expected_imagestream_count, latest_tags_only",
    [
        pytest.param(
            "opendatahub.io/notebook-image=true,platform.opendatahub.io/part-of=workbenches",
            11,
            True,
            id="test_notebook_imagestreams",
        ),
        pytest.param(
            "opendatahub.io/runtime-image=true,platform.opendatahub.io/part-of=workbenches",
            7,
            False,
            id="test_runtime_imagestreams",
        ),
        pytest.param(
            "opendatahub.io/notebook-image=true,platform.opendatahub.io/part-of=trainer",
            3,
            False,
            id="test_trainer_imagestreams",
        ),
    ],
)
def test_workbench_imagestreams_health(
    admin_client: DynamicClient,
    label_selector: str,
    expected_imagestream_count: int,
    latest_tags_only: bool,
) -> None:
    """
    Given workbench-related ImageStreams in the applications namespace.
    When ImageStreams are listed by the expected workbench labels.
    Then all expected ImageStreams exist and the validated tags are imported and resolved.

    For notebook ImageStreams, only the latest 2 version tags are validated
    (these are guaranteed to be mirrored on disconnected clusters).
    Runtime and trainer ImageStreams validate all tags unconditionally.
    """
    imagestreams = list(
        ImageStream.get(
            client=admin_client,
            namespace=py_config["applications_namespace"],
            label_selector=label_selector,
        )
    )

    tags_to_validate: set[str] | None = None
    if latest_tags_only:
        tags_to_validate = set()
        for imagestream in imagestreams:
            imagestream_data: dict[str, Any] = imagestream.instance.to_dict()
            tags_to_validate |= _get_latest_n_tag_names(imagestream_data=imagestream_data)

    _validate_imagestreams_with_label(
        imagestreams=imagestreams,
        label_selector=label_selector,
        expected_count=expected_imagestream_count,
        tags_to_validate=tags_to_validate,
    )


@pytest.mark.skip_on_disconnected
@pytest.mark.parametrize(
    "label_selector, expected_imagestream_count",
    [
        pytest.param(
            "opendatahub.io/notebook-image=true,platform.opendatahub.io/part-of=workbenches",
            11,
            id="test_notebook_imagestreams",
        ),
    ],
)
def test_workbench_imagestreams_older_tags_health(
    admin_client: DynamicClient,
    label_selector: str,
    expected_imagestream_count: int,
) -> None:
    """
    Given workbench-related ImageStreams in the applications namespace.
    When ImageStreams are listed by the expected workbench labels.
    Then older version tags (beyond the latest 2) are also imported and resolved.

    This test is skipped on disconnected clusters where only the latest 2 tags
    are expected to be mirrored.
    """
    imagestreams = list(
        ImageStream.get(
            client=admin_client,
            namespace=py_config["applications_namespace"],
            label_selector=label_selector,
        )
    )

    older_tags: set[str] = set()
    for imagestream in imagestreams:
        imagestream_data: dict[str, Any] = imagestream.instance.to_dict()
        latest_tags = _get_latest_n_tag_names(imagestream_data=imagestream_data)
        all_tag_names = {
            str(spec_tag.get("name"))
            for spec_tag in imagestream_data.get("spec", {}).get("tags", [])
            if spec_tag.get("name")
        }
        older_tags |= all_tag_names - latest_tags

    if not older_tags:
        pytest.fail("No older tags found beyond the latest 2 versions across any notebook ImageStream")

    _validate_imagestreams_with_label(
        imagestreams=imagestreams,
        label_selector=label_selector,
        expected_count=expected_imagestream_count,
        tags_to_validate=older_tags,
    )
