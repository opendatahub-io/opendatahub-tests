# Cutting a release branch

On the day of code freeze for an RHOAI release, this repository is branched so that the
release keeps a stable set of tests while `main` continues to move. Three things have to
happen, and all three are required before component test pipelines pick up the new release.

1. [Cut the release branch](#1-cut-the-release-branch) from `main`.
2. [Publish a container image](#2-publish-the-container-image) tagged for that branch.
3. [Register the image tag with the test pipelines](#3-register-the-image-tag-with-the-test-pipelines).

> This document is public. Do not add internal hostnames, repository URLs, file paths or
> ticket links to it. Step 3 is deliberately described without them.

## Naming convention

The branch name and the image tag are always the same string.

| Release kind | Branch and tag | Examples |
| --- | --- | --- |
| GA | `<major>.<minor>` | `2.25`, `3.4`, `3.5` |
| Early access | `<major>.<minor>ea<n>` | `3.4ea1`, `3.5ea2`, `3.6ea1` |

No `v` prefix, no patch component, and no separator before `ea`. Existing branches are the
source of truth if you are unsure:

```bash
git ls-remote --heads https://github.com/opendatahub-io/opendatahub-tests.git \
  | grep -oE 'refs/heads/[0-9]+\.[0-9]+(ea[0-9]+)?$' | sed 's|refs/heads/||' | sort -V
```

`main` is not branch-specific: it always builds and pushes the `latest` tag.

## Before you start

- Push access to `opendatahub-io/opendatahub-tests`, or a maintainer who can run the action
  for you.
- Access to the internal QE CI configuration for step 3.
- Agreement on the release identifier with whoever owns the release schedule. Cutting under
  the wrong name means redoing all three steps.

## 1. Cut the release branch

### Automated

Actions tab -> **Cut Release Branch** -> **Run workflow**.

| Input | Value to use |
| --- | --- |
| `branch_name` | The release branch, for example `3.6` or `3.6ea1`. |
| `source_ref` | `main`, unless you were told to cut from a specific commit. |
| `image_tag` | Leave empty. It defaults to `branch_name`, which is what you want. |
| `image_name` | Leave at `opendatahub-tests`. |
| `registry` | Leave at `quay.io/opendatahub`. |
| `build_image` | `true`. This also completes step 2. |

The run summary reports the branch, the commit it was cut from, and the image tag. If the
branch already exists, the workflow warns and skips creation rather than failing, so it is
safe to re-run after a failed image build.

Or from the CLI:

```bash
gh workflow run cut-release-branch.yml \
  --repo opendatahub-io/opendatahub-tests \
  -f branch_name=3.6 \
  -f source_ref=main \
  -f build_image=true
```

Pinning `source_ref` to an explicit commit is worth doing if `main` is busy on code freeze
day, so the branch point is not whatever landed while you were filling in the form:

```bash
gh api repos/opendatahub-io/opendatahub-tests/commits/main --jq .sha
```

### Manual

```bash
git fetch upstream
git push upstream upstream/main:refs/heads/3.6
```

Substitute the release name for `3.6`, and your own remote name if it is not `upstream`.
Then confirm:

```bash
git ls-remote --heads https://github.com/opendatahub-io/opendatahub-tests.git 3.6
```

Nothing else needs to change on the new branch. The workflows are inherited from `main`,
and the on-merge workflow already derives its tag from the base branch name.

## 2. Publish the container image

Component pipelines pull `quay.io/opendatahub/opendatahub-tests:<branch>`. Until that tag
exists, nothing can run against the new release.

Images are only ever published by a GitHub workflow. Nobody pushes to
`quay.io/opendatahub/opendatahub-tests` by hand, and the registry credentials live in
repository secrets rather than with individuals. The manual path below is manual in the
sense that you trigger the build yourself; the push still happens in CI.

### Automated

Covered by step 1 when `build_image` is `true`. To build an image for a branch that already
exists, re-run **Cut Release Branch** with the same `branch_name`; creation is skipped and
only the image is rebuilt.

### Manual: trigger the on-merge build with an empty PR

The on-merge workflow builds and pushes a tag named after the PR's base branch, so merging
any PR into the release branch produces the image. An empty commit is enough:

```bash
git fetch upstream
git checkout -b chore/trigger-image-build-3.6 upstream/3.6
git commit --allow-empty -s -m "chore: trigger release image build for 3.6"
git push origin chore/trigger-image-build-3.6

gh pr create \
  --repo opendatahub-io/opendatahub-tests \
  --base 3.6 \
  --head "$(gh api user --jq .login):chore/trigger-image-build-3.6" \
  --title "chore: trigger release image build for 3.6" \
  --body "Empty commit to trigger the on-merge image build for the 3.6 release branch."
```

Merge the PR. **Build and Push Container Image On PR Merge** then pushes the `3.6` tag.
Watch it with:

```bash
gh run list --repo opendatahub-io/opendatahub-tests \
  --workflow build-push-container-on-merge.yml --limit 5
```

### If both workflows fail

Do not build and push the image from a workstation. Raise it with the repository
maintainers instead: the credentials are held as repository secrets, and a hand-pushed
image would be missing the `io.opendatahub.tests.required-images` labels that the workflow
adds, which consumers rely on. See
[Consuming the image manifest](CONSUMING_IMAGE_MANIFEST.md) for what those labels are used
for.

To sanity check a build without publishing anything, build the image locally and do not
push it:

```bash
git fetch upstream
git checkout upstream/3.6
podman build -t opendatahub-tests:3.6-local -f Dockerfile .
```

### Verify the image

```bash
skopeo inspect docker://quay.io/opendatahub/opendatahub-tests:3.6 \
  | jq -r '.Digest, .Labels["io.opendatahub.tests.required-images.sha256"]'
```

Both values should be non-empty. An empty label means the manifest step failed and the
image was published without it; the workflow logs a warning rather than failing the build,
so check the run summary as well.

## 3. Register the image tag with the test pipelines

Component test pipelines do not read the image tag from this repository. They resolve it
from the internal QE CI configuration, which holds one file per release under a shared
framework entry for `opendatahub-tests`. Until that file exists, pipelines for the new
release keep using whatever tag the previous release pinned, and your new image is never
pulled.

Adding the file is normally the only change needed for a release cut. It carries a single
override: the `opendatahub-tests` image tag for that release.

Two things to get right:

- **The release identifier and the image tag are different strings for early access
  releases.** The CI configuration identifies releases in a dotted, hyphenated form, while
  the image tag has no separators. For example a CI release identified as `3.6-ea.1` pins
  the image tag `3.6ea1`. For GA releases the two match, so `3.5` pins `3.5`.
- **Start from the previous release's file rather than from scratch.** Some releases carry
  extra overrides on top of the tag, and copying forward avoids silently dropping them.

The repository, path, file format and review process are internal. Ask the RHOAI QE team
for the location and for the most recent release's change to use as a template.

## After the cut

- Announce the new branch and image tag to the teams that own component pipelines.
- Fixes that belong on both `main` and the release branch land on `main` first, then get
  backported. Comment `/cherry-pick <branch>` on the merged PR and the bot opens the
  backport PR for you.
- Every merge into the release branch rebuilds and overwrites its tag, so the tag always
  reflects the branch head rather than the state at code freeze.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| **Cut Release Branch** is missing from the Actions tab | `workflow_dispatch` workflows are only offered for the default branch. Confirm the workflow file exists on `main`. |
| Branch creation fails with a permissions error | The workflow uses a bot token and falls back to `GITHUB_TOKEN`. If branch protection covers release-branch names, the token needs `contents: write`; escalate to a repository admin. |
| Workflow reports the branch already exists | Expected on a re-run. Creation is skipped and the image is rebuilt. Verify the reported commit is the one you intended. |
| Validation error on an input | Inputs are checked against their allowed format, and `main` is rejected as a `branch_name`. See [GitHub workflows](GITHUB_WORKFLOWS.md) for the rules. |
| Image build succeeded but the tag is missing on quay | Check the push step in the run summary. Registry credentials are repository secrets, so a credential problem shows as a push failure rather than a build failure. |
| Pipelines still run the previous release's tests | Step 3 is missing or pins the wrong tag. Check the release identifier against the image tag; the early access forms differ. |
