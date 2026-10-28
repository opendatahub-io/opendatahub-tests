# Custom workflows

## Supported workflows

### Automatic

- Add PR size label
- Run `tox`
- Close stale PRs
- Check for offensive language
- Assign the PR to the author

### On user action

- Add to or remove a label from PR; supported labels: `wip`, `lgtm`, `verified`, and `hold`.  
- To add a new label, add `/<label name>` in a comment.  
- To remove a label, add `/<label name> cancel` in a comment.  
  `verified` and `lgtm` are removed on new commits.
- To build and push image to quay, add `/build-push-pr-image` in a comment.
  This would create an image with tag pr-<pr_number> to quay repository. This image tag,
  however would be deleted on PR merge or close action.

### Manual (`workflow_dispatch`)

- `Cut Release Branch` creates a release branch from `main` (or any other ref) and pushes a
  matching container image tag. Run it from the Actions tab during code freeze. See the
  [Release Process Guide](RELEASE_PROCESS.md) for the full code freeze checklist, including
  the steps this workflow does not cover.
  Only active members of the `opendatahub-io/opendatahub-tests-maintainers` team can run it;
  the check uses the `RHODS_CI_BOT_PAT` secret and fails closed. If any job fails, the workflow
  opens an issue linking to the failed run.
  Inputs:

  | Input | Default | Description |
  | --- | --- | --- |
  | `branch_name` | (required) | Release branch to create, for example `3.6`. |
  | `source_ref` | `main` | Branch, tag or commit SHA to cut from. |
  | `image_tag` | `branch_name` | Image tag override. |
  | `image_name` | `opendatahub-tests` | Image name, without the registry prefix. |
  | `build_image` | `true` | Build and push the image after the branch is created. |

  Surrounding whitespace is stripped from every input. `branch_name`, `source_ref`,
  `image_tag` and `image_name` are then checked against the format each one is
  allowed to take, and `main` is rejected as a `branch_name`. An empty `image_tag` falls
  back to `branch_name`.

  If the branch already exists, branch creation is skipped with a warning and the image is
  still rebuilt from that branch. With `build_image: false` the branch is created but no
  image is built or published.

## Reusable workflows

`build-push-image.yml` builds the `Dockerfile` and pushes it to a registry, including the
`io.opendatahub.tests.required-images` manifest labels. It is called by `Cut Release Branch`.
The on-merge workflow keeps its own inline build-and-push steps rather than calling this
reusable workflow, so it is unaffected by changes here. Call it with:

```yaml
jobs:
  build-push:
    uses: ./.github/workflows/build-push-image.yml
    with:
      ref: <git ref to build>
      tag: <image tag>
    secrets:
      QUAY_USERNAME: ${{ secrets.QUAY_USERNAME }}
      QUAY_PASSWORD: ${{ secrets.QUAY_PASSWORD }}
```

## How to add a new workflow

1. Create a new file in `.github/workflows` directory.
2. Add relevant steps to the workflow.
3. Code should be implemented in Python and placed in `.github/scripts` directory.
4. Make sure that the workflow is triggered only on relevant events.
5. Set `ACTION` environment variable in the workflow and use it in the code to identify the relevant workflow.

## To be added

- Block merging if not all defined checks pass. For example: a `verified` label was added and at least 2 approvals.
- When a PR is opened, add reviewers (requires updates to OWNERS file(s))
- When a PR is ready to be merged (all checks passed), add `ready-to-merge` label
- If a label is missing from the repository (i.e was manually deleted), add it back (label colors should be defined as well)
- Tests
