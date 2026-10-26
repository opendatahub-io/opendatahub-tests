import re
import subprocess

CODEOWNERS_PATH = ".github/CODEOWNERS"
OWNER_REFERENCE = re.compile(r"@(?P<owner>[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)?)\Z")


def read_codeowners(commit_sha: str) -> str | None:
    """Read CODEOWNERS at a commit, returning None only when the file is absent."""
    commit_check = subprocess.run(
        ["git", "cat-file", "-e", f"{commit_sha}^{{commit}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if commit_check.returncode != 0:
        raise RuntimeError(f"Commit {commit_sha} is not available in the checkout.")

    file_check = subprocess.run(
        ["git", "cat-file", "-e", f"{commit_sha}:{CODEOWNERS_PATH}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if file_check.returncode != 0:
        return None

    result = subprocess.run(
        ["git", "show", f"{commit_sha}:{CODEOWNERS_PATH}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Unable to read {CODEOWNERS_PATH} at commit {commit_sha}.")
    return result.stdout


def parse_owners(contents: str) -> set[str]:
    """Return normalized user and team references from CODEOWNERS entries."""
    owners: set[str] = set()
    for line in contents.splitlines():
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue

        fields = stripped_line.split()
        for field in fields[1:]:
            if field.startswith("#"):
                break
            match = OWNER_REFERENCE.fullmatch(string=field)
            if match:
                owners.add(match.group("owner").casefold())

    return owners


def get_owner_changes(base_contents: str, head_contents: str) -> tuple[set[str], set[str]]:
    """Return owners added and removed between two CODEOWNERS contents."""
    base_owners = parse_owners(contents=base_contents)
    head_owners = parse_owners(contents=head_contents)
    return head_owners - base_owners, base_owners - head_owners
