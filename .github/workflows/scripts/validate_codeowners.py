import logging
import os
import sys
from datetime import date, timedelta

import github.Auth
from codeowners_utils import CODEOWNERS_PATH, get_owner_changes, parse_owners, read_codeowners
from github.GithubException import GithubException
from github.MainClass import Github
from github.Repository import Repository

LOGGER = logging.getLogger("codeowners_eligibility")
ELIGIBILITY_WINDOW_DAYS = 365


def expand_owner_to_users(github_client: Github, owner: str, repository_organization: str) -> set[str]:
    """Return the user login(s) represented by an owner reference."""
    if "/" not in owner:
        return {owner}

    organization_name, team_slug = owner.split("/", maxsplit=1)
    if organization_name.casefold() != repository_organization.casefold():
        raise ValueError("CODEOWNERS teams must belong to the repository organization.")

    team = github_client.get_organization(login=organization_name).get_team_by_slug(slug=team_slug)
    return {member.login.casefold() for member in team.get_members()}


def has_recent_contribution(github_client: Github, repository: Repository, username: str, since: date) -> bool:
    """Check whether a user authored or reviewed a repository PR in the past year."""
    authored_query = f"repo:{repository.full_name} is:pr author:{username} created:>={since.isoformat()}"
    if github_client.search_issues(query=authored_query).totalCount > 0:
        return True

    reviewed_query = f"repo:{repository.full_name} is:pr reviewed-by:{username} updated:>={since.isoformat()}"
    for issue in github_client.search_issues(query=reviewed_query):
        for review in repository.get_pull(issue.number).get_reviews():
            if (
                review.user
                and review.user.login.casefold() == username
                and review.submitted_at
                and review.submitted_at.date() >= since
            ):
                return True

    return False


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    repository = os.getenv("GITHUB_REPOSITORY")
    token = os.getenv("GITHUB_TOKEN")
    base_sha = os.getenv("PR_BASE_SHA")
    head_sha = os.getenv("PR_HEAD_SHA")
    missing_variables = [
        variable_name
        for variable_name, value in (
            ("GITHUB_REPOSITORY", repository),
            ("GITHUB_TOKEN", token),
            ("PR_BASE_SHA", base_sha),
            ("PR_HEAD_SHA", head_sha),
        )
        if not value
    ]
    if missing_variables:
        LOGGER.error("Missing required environment variable(s): %s", ", ".join(missing_variables))
        return 1

    assert repository is not None
    assert token is not None
    assert base_sha is not None
    assert head_sha is not None

    try:
        base_contents = read_codeowners(commit_sha=base_sha) or ""
        head_contents = read_codeowners(commit_sha=head_sha)
    except RuntimeError as error:
        LOGGER.error("Unable to read CODEOWNERS versions for this pull request: %s", error)
        return 1

    if head_contents is None:
        LOGGER.error("The pull request deletes %s; restore the file before adding or changing owners.", CODEOWNERS_PATH)
        return 1

    head_owners = parse_owners(contents=head_contents)
    if not head_owners:
        LOGGER.error("%s has no valid owner entries.", CODEOWNERS_PATH)
        return 1

    added_owners, _removed_owners = get_owner_changes(base_contents=base_contents, head_contents=head_contents)
    added_owners = sorted(added_owners)
    if not added_owners:
        LOGGER.info("No new users or teams were added to %s.", CODEOWNERS_PATH)
        return 0

    LOGGER.info("Checking newly added owner(s): %s", ", ".join(f"@{owner}" for owner in added_owners))
    github_client = Github(auth=github.Auth.Token(token))
    try:
        repository_api = github_client.get_repo(full_name_or_id=repository)
        users_to_check: set[str] = set()
        for owner in added_owners:
            users_to_check.update(
                expand_owner_to_users(
                    github_client=github_client,
                    owner=owner,
                    repository_organization=repository.split("/", maxsplit=1)[0],
                )
            )

        since = date.today() - timedelta(days=ELIGIBILITY_WINDOW_DAYS)
        ineligible_users: list[str] = []
        for username in sorted(users_to_check):
            if not has_recent_contribution(
                github_client=github_client,
                repository=repository_api,
                username=username,
                since=since,
            ):
                ineligible_users.append(username)
    except GithubException, ValueError:
        LOGGER.error("Unable to evaluate one or more newly added CODEOWNERS references.")
        return 1

    if ineligible_users:
        LOGGER.error("Eligibility requires at least one authored or reviewed PR in the past year.")
        LOGGER.error("One or more newly added CODEOWNERS users do not meet the eligibility requirement.")
        return 1

    LOGGER.info("All newly added CODEOWNERS users meet the eligibility requirement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
