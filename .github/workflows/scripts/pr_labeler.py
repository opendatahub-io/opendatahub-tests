from __future__ import annotations

import os
import re
import sys
from github import Github
from github.PullRequest import PullRequest
from github.Repository import Repository
from simple_logger.logger import get_logger

LOGGER = get_logger(name="pr_labeler")


def get_pr_size(pr: PullRequest) -> int:
    additions: int = 0

    for file in pr.get_files():
        additions += file.additions + file.deletions

    LOGGER.info(f"PR size: {additions}")
    return additions


def get_size_label(size: int) -> str:
    size_labels: dict[tuple[int, int], str] = {
        (0, 20): "size/xs",
        (21, 50): "size/s",
        (51, 100): "size/m",
        (101, 300): "size/l",
        (301, 1000): "size/xl",
        (1001, sys.maxsize): "size/xxl",
    }

    for (min_size, max_size), label in size_labels.items():
        if min_size <= size <= max_size:
            return label
    return "size/xxl"


def set_pr_size(pr: PullRequest) -> None:
    size: int = get_pr_size(pr=pr)
    size_label: str = get_size_label(size=size)

    for label in pr.labels:
        if label.name == size_label:
            LOGGER.info(f"Label {label.name} already set")
            return

        if label.name.lower().startswith("size/"):
            LOGGER.info(f"Removing label {label.name}")
            pr.remove_from_labels(label.name)

    LOGGER.info(f"New label: {size_label}")
    pr.add_to_labels(size_label)


def add_remove_pr_labels(pr: PullRequest, event_name: str, event_action: str, comment_body: str = "") -> None:
    wip_str: str = "wip"
    lgtm_str: str = "lgtm"
    verified_str: str = "verified"
    hold_str: str = "hold"
    label_prefix: str = "/"

    LOGGER.info(
        f"add_remove_pr_label comment_body: {comment_body} event_name:{event_name} event_action: {event_action}"
    )

    pr_labels = [label.name for label in pr.labels]
    LOGGER.info(f"PR labels: {pr_labels}")

    # Remove labels on new commit
    if event_action == "synchronize":
        LOGGER.info("Synchronize event")
        for label in pr_labels:
            if label.lower() in (lgtm_str, verified_str):
                LOGGER.info(f"Removing label {label}")
                pr.remove_from_labels(label)
        return

    elif event_name == "issue_comment":
        LOGGER.info("Issue comment event")
        supported_labels: set[str] = {
            f"{label_prefix}{wip_str}",
            f"{label_prefix}{lgtm_str}",
            f"{label_prefix}{verified_str}",
            f"{label_prefix}{hold_str}",
        }

        # Searches for `supported_labels` in PR comment and splits to tuples;
        # index 0 is label, index 1 (optional) `cancel`
        user_requested_labels: list[tuple[str, str]] = re.findall(
            rf"({'|'.join(supported_labels)})\s*(cancel)?", comment_body.lower()
        )

        LOGGER.info(f"User labels: {user_requested_labels}")

        # In case of the same label appears multiple times, the last one is used
        labels: dict[str, dict[str, bool]] = {}
        for _label in user_requested_labels:
            labels[_label[0].replace(label_prefix, "")] = {"cancel": _label[1] == "cancel"}

        LOGGER.info(f"Processing labels: {labels}")
        for label, action in labels.items():
            label_in_pr = any([label == _label.lower() for _label in pr_labels])
            if action["cancel"] or event_action == "deleted":
                if label_in_pr:
                    LOGGER.info(f"Removing label {label}")
                    pr.remove_from_labels(label)
            elif not label_in_pr:
                LOGGER.info(f"Adding label {label}")
                pr.add_to_labels(label)

        return

    LOGGER.warning("`add_remove_pr_label` called without a supported event")


def main() -> None:
    labels_action_name: str = "add-remove-labels"
    pr_size_action_name: str = "add-pr-size-label"
    supported_actions: set[str] = {pr_size_action_name, labels_action_name}
    action: str | None = os.getenv("ACTION")

    if not action or action not in supported_actions:
        sys.exit("`ACTION` is not set in workflow or is not supported. Supported actions: {supported_actions}")

    github_token: str | None = os.getenv("GITHUB_TOKEN")
    if not github_token:
        sys.exit("`GITHUB_TOKEN` is not set")

    repo_name: str = os.environ["GITHUB_REPOSITORY"]

    pr_number: int = int(os.getenv("GITHUB_PR_NUMBER", 0))
    if not pr_number:
        sys.exit("`GITHUB_PR_NUMBER` is not set")

    event_action: str | None = os.getenv("GITHUB_EVENT_ACTION")
    if not event_action:
        sys.exit("`GITHUB_EVENT_ACTION` is not set")

    event_name: str | None = os.getenv("GITHUB_EVENT_NAME")
    if not event_name:
        sys.exit("`GITHUB_EVENT_NAME` is not set")

    LOGGER.info(f"pr number: {pr_number}, event_action: {event_action}, event_name: {event_name}, action: {action}")

    comment_body: str = ""
    if action == labels_action_name and event_name == "issue_comment":
        comment_body = os.getenv("COMMENT_BODY") or comment_body
        if not comment_body:
            sys.exit("`COMMENT_BODY` is not set")

    gh_client: Github = Github(github_token)
    repo: Repository = gh_client.get_repo(repo_name)
    pr: PullRequest = repo.get_pull(pr_number)

    if action == pr_size_action_name:
        set_pr_size(pr=pr)

    if action == labels_action_name:
        add_remove_pr_labels(
            pr=pr,
            event_name=event_name,
            event_action=event_action,
            comment_body=comment_body,
        )


if __name__ == "__main__":
    main()
