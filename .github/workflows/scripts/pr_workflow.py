from __future__ import annotations

import os
import re
import sys
from github import Github
from github.PullRequest import PullRequest
from github.Repository import Repository
from simple_logger.logger import get_logger

LOGGER = get_logger(name="pr_labeler")


WIP_STR: str = "wip"
LGTM_STR: str = "lgtm"
VERIFIED_STR: str = "verified"
HOLD_STR: str = "hold"
LABEL_PREFIX: str = "/"

SUPPORTED_LABELS: set[str] = {
    f"{LABEL_PREFIX}{WIP_STR}",
    f"{LABEL_PREFIX}{LGTM_STR}",
    f"{LABEL_PREFIX}{VERIFIED_STR}",
    f"{LABEL_PREFIX}{HOLD_STR}",
}

CANCEL_ACTION: str = "cancel"
WELCOME_COMMENT: str = f"""
The following are automatically added/executed:
 * PR size label.
 * Run [pre-commit](https://pre-commit.ci/)
 * Run [tox](https://tox.wiki/)

Available user actions:
 * To mark a PR as `WIP`, add `/wip` in a comment. To remove it from the PR comment `/wip cancel` to the PR.
 * To block merging of a PR, add `/hold` in a comment. To un-block merging of PR comment `/hold cancel`.
 * To mark a PR as approved, add `/lgtm` in a comment. To remove, add `/lgtm cancel`.
        `lgtm` label removed on each new commit push.
 * To mark PR as verified comment `/verified` to the PR, to un-verify comment `/verified cancel` to the PR.
        `verified` label removed on each new commit push.

<details>
<summary>Supported labels</summary>

{SUPPORTED_LABELS}
</details>
    """


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
    if comment_body and WELCOME_COMMENT in comment_body:
        LOGGER.info(f"Welcome message found in PR {pr.title}. Not processing")
        return

    LOGGER.info(
        f"add_remove_pr_label comment_body: {comment_body} event_name:{event_name} event_action: {event_action}"
    )

    pr_labels = [label.name for label in pr.labels]
    LOGGER.info(f"PR labels: {pr_labels}")

    # Remove labels on new commit
    if event_action == "synchronize":
        LOGGER.info("Synchronize event")
        for label in pr_labels:
            if label.lower() in (LGTM_STR, VERIFIED_STR):
                LOGGER.info(f"Removing label {label}")
                pr.remove_from_labels(label)
        return

    elif event_name == "issue_comment":
        LOGGER.info("Issue comment event")

        # Searches for `supported_labels` in PR comment and splits to tuples;
        # index 0 is label, index 1 (optional) `cancel`
        user_requested_labels: list[tuple[str, str]] = re.findall(
            rf"({'|'.join(SUPPORTED_LABELS)})\s*(cancel)?", comment_body.lower()
        )

        LOGGER.info(f"User labels: {user_requested_labels}")

        # In case of the same label appears multiple times, the last one is used
        labels: dict[str, dict[str, bool]] = {}
        for _label in user_requested_labels:
            labels[_label[0].replace(LABEL_PREFIX, "")] = {CANCEL_ACTION: _label[1] == CANCEL_ACTION}

        LOGGER.info(f"Processing labels: {labels}")
        for label, action in labels.items():
            label_in_pr = any([label == _label.lower() for _label in pr_labels])
            if action[CANCEL_ACTION] or event_action == "deleted":
                if label_in_pr:
                    LOGGER.info(f"Removing label {label}")
                    pr.remove_from_labels(label)
            elif not label_in_pr:
                LOGGER.info(f"Adding label {label}")
                pr.add_to_labels(label)

        return

    LOGGER.warning("`add_remove_pr_label` called without a supported event")


def add_welcome_comment(pr: PullRequest) -> None:
    pr.create_issue_comment(body=WELCOME_COMMENT)


def main() -> None:
    labels_action_name: str = "add-remove-labels"
    pr_size_action_name: str = "add-pr-size-label"
    welcome_comment_action_name: str = "add-welcome-comment"
    supported_actions: set[str] = {pr_size_action_name, labels_action_name, welcome_comment_action_name}
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

    if action == welcome_comment_action_name:
        add_welcome_comment(pr=pr)


if __name__ == "__main__":
    main()
