from __future__ import annotations

import os
import re
import sys

from github.PullRequest import PullRequest
from github.Repository import Repository

from constants import (
    ALL_LABELS_DICT,
    CANCEL_ACTION,
    CHANGED_REQUESTED_BY_LABEL_PREFIX,
    COMMENTED_BY_LABEL_PREFIX,
    DEFAULT_LABEL_COLOR,
    LABEL_PREFIX,
    LGTM_BY_LABEL_PREFIX,
    LGTM_LABEL_STR,
    SIZE_LABEL_PREFIX,
    SUPPORTED_LABELS,
    VERIFIED_LABEL_STR,
    WELCOME_COMMENT,
)
from github import Github, UnknownObjectException
from simple_logger.logger import get_logger

LOGGER = get_logger(name="pr_labeler")


class PrBaseClass:
    class SupportedActions:
        add_remove_labels_action_name: str = "add-remove-labels"
        pr_size_action_name: str = "add-pr-size-label"
        welcome_comment_action_name: str = "add-welcome-comment"
        supported_actions: set[str] = {
            pr_size_action_name,
            add_remove_labels_action_name,
            welcome_comment_action_name,
        }

    def __init__(self) -> None:
        self.repo: Repository
        self.pr: PullRequest

        self.repo_name = os.environ["GITHUB_REPOSITORY"]
        self.pr_number = int(os.getenv("GITHUB_PR_NUMBER", 0))
        self.action = os.getenv("ACTION")
        self.event_action = os.getenv("GITHUB_EVENT_ACTION")
        self.event_name = os.getenv("GITHUB_EVENT_NAME")
        self.github_token = os.getenv("GITHUB_TOKEN")

        self.verify_base_config()
        self.set_gh_config()

    def verify_base_config(self) -> None:
        if not self.action or self.action not in self.SupportedActions.supported_actions:
            sys.exit(
                "`ACTION` is not set in workflow or is not supported. "
                f"Supported actions: {self.SupportedActions.supported_actions}"
            )

        if not self.pr_number:
            sys.exit("`GITHUB_PR_NUMBER` is not set")

        if not self.event_action:
            sys.exit("`GITHUB_EVENT_ACTION` is not set")

        if not self.event_name:
            sys.exit("`GITHUB_EVENT_NAME` is not set")

        if not self.github_token:
            sys.exit("`GITHUB_TOKEN` is not set")

        LOGGER.info(
            f"pr number: {self.pr_number}, event_action: {self.event_action}, "
            f"event_name: {self.event_name}, action: {self.action}"
        )

    def set_gh_config(self) -> None:
        gh_client: Github = Github(login_or_token=self.github_token)
        self.repo = gh_client.get_repo(full_name_or_id=self.repo_name)
        self.pr = self.repo.get_pull(number=self.pr_number)


class PrLabeler(PrBaseClass):
    def __init__(self) -> None:
        super().__init__()
        self.user_login = os.getenv("GITHUB_USER_LOGIN")
        self.review_state = os.getenv("GITHUB_EVENT_REVIEW_STATE")
        self.comment_body = os.getenv("COMMENT_BODY", "")
        self.last_commit = list(self.pr.get_commits())[-1]
        self.last_commit_sha = self.last_commit.sha

        self.verify_labeler_config()

    def verify_labeler_config(self) -> None:
        if self.action == self.SupportedActions.add_remove_labels_action_name and self.event_name in (
            "issue_comment",
            "pull_request_review",
        ):
            if not self.user_login:
                sys.exit("`GITHUB_USER_LOGIN` is not set")

            if self.event_name == "issue_comment" and not self.comment_body:
                sys.exit("`COMMENT_BODY` is not set")

            if self.event_name == "pull_request_review" and not self.review_state:
                sys.exit("`GITHUB_EVENT_REVIEW_STATE` is not set")

    def run_pr_label_action(self) -> None:
        if self.action == self.SupportedActions.pr_size_action_name:
            self.set_pr_size()

        if self.action == self.SupportedActions.add_remove_labels_action_name:
            self.add_remove_pr_labels()

        if self.action == self.SupportedActions.welcome_comment_action_name:
            self.add_welcome_comment()

    def get_pr_size(self) -> int:
        additions: int = 0

        for file in self.pr.get_files():
            additions += file.additions + file.deletions

        LOGGER.info(f"PR size: {additions}")
        return additions

    @staticmethod
    def get_size_label(size: int) -> str:
        xxl_size = f"{SIZE_LABEL_PREFIX}xxl"

        size_labels: dict[tuple[int, int], str] = {
            (0, 20): f"{SIZE_LABEL_PREFIX}xs",
            (21, 50): f"{SIZE_LABEL_PREFIX}s",
            (51, 100): f"{SIZE_LABEL_PREFIX}m",
            (101, 300): f"{SIZE_LABEL_PREFIX}l",
            (301, 1000): f"{SIZE_LABEL_PREFIX}xl",
            (1001, sys.maxsize): xxl_size,
        }

        for (min_size, max_size), label in size_labels.items():
            if min_size <= size <= max_size:
                return label
        return xxl_size

    def add_pr_label(self, label: str) -> None:
        self.set_label_in_repository(label=label)

        LOGGER.info(f"New label: {label}")
        self.pr.add_to_labels(label)

    def set_label_in_repository(self, label: str) -> None:
        label_color = [
            label_color for label_prefix, label_color in ALL_LABELS_DICT.items() if label.startswith(label_prefix)
        ]
        label_color = label_color[0] if label_color else DEFAULT_LABEL_COLOR

        repo_labels = {_label.name: _label.color for _label in self.repo.get_labels()}
        LOGGER.info(f"repo labels: {repo_labels}")

        try:
            if _repo_label := self.repo.get_label(name=label):
                if _repo_label.color != label_color:
                    LOGGER.info(f"Edit repository label: {label}, color: {label_color}")
                    _repo_label.edit(name=_repo_label.name, color=label_color)

        except UnknownObjectException:
            LOGGER.info(f"Add repository label: {label}, color: {label_color}")
            self.repo.create_label(name=label, color=label_color)

    def set_pr_size(self) -> None:
        size: int = self.get_pr_size()
        size_label: str = self.get_size_label(size=size)

        for label in self.pr.labels:
            if label.name == size_label:
                LOGGER.info(f"Label {label.name} already set")
                return

            if label.name.lower().startswith(SIZE_LABEL_PREFIX):
                LOGGER.info(f"Removing label {label.name}")
                self.pr.remove_from_labels(label=label.name)

        self.add_pr_label(label=size_label)

    @property
    def pr_labels(self) -> list[str]:
        pr_labels = [label.name for label in self.pr.labels]
        LOGGER.info(f"PR labels: {pr_labels}")

        return pr_labels

    def add_remove_pr_labels(self) -> None:
        if self.comment_body and WELCOME_COMMENT in self.comment_body:
            LOGGER.info(f"Welcome message found in PR {self.pr.title}. Not processing")
            return

        LOGGER.info(
            f"add_remove_pr_label comment_body: {self.comment_body} event_name:{self.event_name} "
            f"event_action: {self.event_action} review_state {self.review_state}"
        )

        # Remove labels on new commit
        if self.event_action == "synchronize":
            LOGGER.info("Synchronize event")
            for label in self.pr_labels:
                LOGGER.info(f"Processing label: {label}")
                if (
                    label.lower() == VERIFIED_LABEL_STR
                    or label.lower().startswith(LGTM_BY_LABEL_PREFIX)
                    or label.lower().startswith(CHANGED_REQUESTED_BY_LABEL_PREFIX)
                ):
                    LOGGER.info(f"Removing label {label}")
                    self.pr.remove_from_labels(label=label)

            return

        elif self.event_name == "issue_comment":
            self.issue_comment_label_actions()

            return

        elif self.event_name == "pull_request_review":
            self.pull_request_review_label_actions()

            return

        LOGGER.warning("`add_remove_pr_label` called without a supported event")

    def pull_request_review_label_actions(
        self,
    ) -> None:
        LOGGER.info(f"{self.event_name} event, state: {self.review_state}")

        lgtm_label = f"{LGTM_BY_LABEL_PREFIX}{self.user_login}"
        change_requested_label = f"{CHANGED_REQUESTED_BY_LABEL_PREFIX}{self.user_login}"

        label_to_remove = None
        label_to_add = None

        if self.review_state == "approved":
            label_to_remove = change_requested_label
            label_to_add = lgtm_label

        elif self.review_state == "changes_requested":
            label_to_add = change_requested_label
            label_to_remove = lgtm_label

        elif self.review_state == "commented":
            label_to_add = f"{COMMENTED_BY_LABEL_PREFIX}{self.user_login}"

        if label_to_add and label_to_add not in self.pr_labels:
            self.add_pr_label(label=label_to_add)

        if label_to_remove and label_to_remove in self.pr_labels:
            LOGGER.info(f"Removing review label {label_to_add}")
            self.pr.remove_from_labels(label=label_to_remove)

    def issue_comment_label_actions(
        self,
    ) -> None:
        LOGGER.info(f"{self.event_name} event")
        # Searches for `supported_labels` in PR comment and splits to tuples;
        # index 0 is label, index 1 (optional) `cancel`
        if user_requested_labels := re.findall(
            rf"({'|'.join(SUPPORTED_LABELS)})\s*(cancel)?", self.comment_body.lower()
        ):
            LOGGER.info(f"User labels: {user_requested_labels}")

            # In case of the same label appears multiple times, the last one is used
            labels: dict[str, dict[str, bool]] = {}
            for _label in user_requested_labels:
                labels[_label[0].replace(LABEL_PREFIX, "")] = {CANCEL_ACTION: _label[1] == CANCEL_ACTION}

            LOGGER.info(f"Processing labels: {labels}")
            for label, action in labels.items():
                if label == LGTM_LABEL_STR:
                    label = f"{LGTM_BY_LABEL_PREFIX}{self.user_login}"

                label_in_pr = any([label == _label.lower() for _label in self.pr_labels])
                LOGGER.info(f"Processing label: {label}, action: {action}")

                if action[CANCEL_ACTION] or self.event_action == "deleted":
                    if label_in_pr:
                        LOGGER.info(f"Removing label {label}")
                        self.pr.remove_from_labels(label=label)

                elif not label_in_pr:
                    self.add_pr_label(label=label)

        else:
            commented_by_label = f"{COMMENTED_BY_LABEL_PREFIX}{self.user_login}"
            if commented_by_label not in self.pr_labels:
                self.add_pr_label(label=commented_by_label)

    def add_welcome_comment(self) -> None:
        self.pr.create_issue_comment(body=WELCOME_COMMENT)


def main() -> None:
    labeler = PrLabeler()
    labeler.run_pr_label_action()


if __name__ == "__main__":
    main()
