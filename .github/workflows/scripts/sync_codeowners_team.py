import logging
import os
import subprocess
import sys

import github.Auth
from codeowners_utils import CODEOWNERS_PATH, get_owner_changes, parse_owners, read_codeowners
from github.GithubException import GithubException, UnknownObjectException
from github.MainClass import Github
from github.Organization import Organization
from github.Repository import Repository
from github.Team import Team

LOGGER = logging.getLogger("codeowners_team_sync")
TARGET_ORG = "opendatahub-io"
TARGET_TEAM = "opendatahub-tests-contributors"
ZERO_SHA = "0" * 40


def expand_owner_to_users(github_client: Github, owner: str) -> set[str]:
    """Return the user login(s) represented by a newly added owner reference."""
    if "/" not in owner:
        return {owner}

    organization_name, team_slug = owner.split("/", maxsplit=1)
    source_team = github_client.get_organization(login=organization_name).get_team_by_slug(slug=team_slug)
    members = {member.login.casefold() for member in source_team.get_members()}
    LOGGER.info("Expanded @%s to %d team member(s).", owner, len(members))
    return members


def expand_owners_to_users(github_client: Github, owners: set[str], owner_cache: dict[str, set[str]]) -> set[str]:
    """Return all users covered by a set of CODEOWNERS references."""
    users: set[str] = set()
    for owner in sorted(owners):
        if owner not in owner_cache:
            owner_cache[owner] = expand_owner_to_users(github_client=github_client, owner=owner)
        users.update(owner_cache[owner])
    return users


def add_user_to_team(
    github_client: Github,
    organization: Organization,
    team: Team,
    username: str,
    token: str,
) -> None:
    """Add a user to the contributors team unless they are already a member."""
    try:
        user = github_client.get_user(login=username)
    except UnknownObjectException:
        LOGGER.info("@%s is not an opendatahub-io member; skipping.", username)
        return

    if not organization.has_in_members(member=user):
        LOGGER.info("@%s is not an opendatahub-io member; skipping.", username)
        return

    try:
        team.get_team_membership(member=username)
        LOGGER.info("@%s is already in @%s/%s; skipping.", username, TARGET_ORG, TARGET_TEAM)
    except UnknownObjectException:
        gh_environment = os.environ.copy()
        gh_environment["GH_TOKEN"] = token
        result = subprocess.run(
            [
                "gh",
                "api",
                "--method",
                "PUT",
                f"/orgs/{TARGET_ORG}/teams/{TARGET_TEAM}/memberships/{username}",
                "-f",
                "role=member",
            ],
            check=False,
            capture_output=True,
            env=gh_environment,
            text=True,
        )
        if result.returncode != 0:
            error_details = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"Unable to add @{username} to the contributors team: {error_details}")
        LOGGER.info("Added @%s to @%s/%s.", username, TARGET_ORG, TARGET_TEAM)


def create_removal_issue(
    github_client: Github,
    repository: Repository,
    removed_users: list[str],
    removed_owner_references: list[str],
    after_sha: str,
) -> None:
    """Create one issue asking the contributors team to review fully removed owners."""
    short_sha = after_sha[:12]
    title = f"Review CODEOWNERS removals ({after_sha})"
    duplicate_query = f'repo:{repository.full_name} is:issue in:title "Review CODEOWNERS removals" "{after_sha}"'
    if github_client.search_issues(query=duplicate_query).totalCount > 0:
        LOGGER.info("An issue for CODEOWNERS removals in commit %s already exists.", short_sha)
        return

    removed_users_markdown = "\n".join(f"- `@{username}`" for username in removed_users)
    removed_references_markdown = "\n".join(f"- `@{owner}`" for owner in removed_owner_references)
    commit_url = f"{repository.html_url}/commit/{after_sha}"
    body = (
        f"@{TARGET_ORG}/{TARGET_TEAM}\n\n"
        f"The following user(s) are no longer covered by any owner entry in `{CODEOWNERS_PATH}` after "
        f"[commit `{short_sha}`]({commit_url}):\n\n"
        f"{removed_users_markdown}\n\n"
        f"Removed owner reference(s):\n\n{removed_references_markdown}\n\n"
        "Please review whether team membership changes are needed. "
        "This workflow does not remove members from the team."
    )
    repository.create_issue(title=title, body=body)
    LOGGER.info("Created issue for CODEOWNERS removals in commit %s.", short_sha)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    repository_name = os.getenv("GITHUB_REPOSITORY")
    token = os.getenv("GITHUB_TOKEN")
    before_sha = os.getenv("PUSH_BEFORE_SHA")
    after_sha = os.getenv("PUSH_AFTER_SHA")
    missing_variables = [
        variable_name
        for variable_name, value in (
            ("GITHUB_REPOSITORY", repository_name),
            ("GITHUB_TOKEN", token),
            ("PUSH_BEFORE_SHA", before_sha),
            ("PUSH_AFTER_SHA", after_sha),
        )
        if not value
    ]
    if missing_variables:
        LOGGER.error("Missing required environment variable(s): %s", ", ".join(missing_variables))
        return 1

    assert repository_name is not None
    assert token is not None
    assert before_sha is not None
    assert after_sha is not None

    try:
        before_contents = "" if before_sha == ZERO_SHA else read_codeowners(commit_sha=before_sha) or ""
        after_contents = read_codeowners(commit_sha=after_sha) or ""
    except RuntimeError as error:
        LOGGER.error("Unable to read CODEOWNERS versions for this push: %s", error)
        return 1

    added_owners, removed_owner_references = get_owner_changes(
        base_contents=before_contents,
        head_contents=after_contents,
    )
    if not added_owners and not removed_owner_references:
        LOGGER.info("No CODEOWNERS user or team references were added or removed.")
        return 0

    github_client = Github(auth=github.Auth.Token(token))
    repository = github_client.get_repo(full_name_or_id=repository_name)
    operation_failed = False
    owner_cache: dict[str, set[str]] = {}

    if added_owners:
        try:
            target_organization = github_client.get_organization(login=TARGET_ORG)
            target_team = target_organization.get_team_by_slug(slug=TARGET_TEAM)
        except GithubException as error:
            LOGGER.error("Unable to access the contributors team: %s", error)
            operation_failed = True
        else:
            users_to_add: set[str] = set()
            for owner in sorted(added_owners):
                try:
                    users_to_add.update(
                        expand_owners_to_users(
                            github_client=github_client,
                            owners={owner},
                            owner_cache=owner_cache,
                        )
                    )
                except GithubException as error:
                    LOGGER.error("Unable to expand newly added owner @%s: %s", owner, error)
                    operation_failed = True

            for username in sorted(users_to_add):
                try:
                    add_user_to_team(
                        github_client=github_client,
                        organization=target_organization,
                        team=target_team,
                        username=username,
                        token=token,
                    )
                except (GithubException, RuntimeError) as error:
                    LOGGER.error("Unable to add newly listed CODEOWNERS user @%s: %s", username, error)
                    operation_failed = True

    if removed_owner_references:
        try:
            before_owners = parse_owners(contents=before_contents)
            after_owners = parse_owners(contents=after_contents)
            before_users = expand_owners_to_users(
                github_client=github_client,
                owners=before_owners,
                owner_cache=owner_cache,
            )
            after_users = expand_owners_to_users(
                github_client=github_client,
                owners=after_owners,
                owner_cache=owner_cache,
            )
            removed_users = sorted(before_users - after_users)
        except (GithubException, RuntimeError) as error:
            LOGGER.error("Unable to verify whether CODEOWNERS removals remove users completely: %s", error)
            operation_failed = True
        else:
            if removed_users:
                try:
                    create_removal_issue(
                        github_client=github_client,
                        repository=repository,
                        removed_users=removed_users,
                        removed_owner_references=sorted(removed_owner_references),
                        after_sha=after_sha,
                    )
                except (GithubException, RuntimeError) as error:
                    LOGGER.error("Unable to create an issue for fully removed CODEOWNERS users: %s", error)
                    operation_failed = True
            else:
                LOGGER.info("No users lost all CODEOWNERS coverage; no issue created.")

    return 1 if operation_failed else 0


if __name__ == "__main__":
    sys.exit(main())
