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
  matching container image tag. Run it from the Actions tab during code freeze.
  Inputs:

  | Input | Default | Description |
  | --- | --- | --- |
  | `branch_name` | (required) | Release branch to create, for example `3.6`. |
  | `source_ref` | `main` | Branch, tag or commit SHA to cut from. |
  | `image_tag` | `branch_name` | Image tag override. |
  | `image_name` | `opendatahub-tests` | Image name, without the registry prefix. |
  | `registry` | `quay.io/opendatahub` | Registry and namespace to push to. |
  | `build_image` | `true` | Build and push the image after the branch is created. |

  If the branch already exists, branch creation is skipped with a warning and the image is
  still rebuilt from that branch.

## Reusable workflows

`build-push-image.yml` builds the `Dockerfile` and pushes it to a registry. It is called by
the on-merge workflow and by `Cut Release Branch`, so both produce identical images,
including the `io.opendatahub.tests.required-images` manifest labels. Call it with:

```yaml
uses: ./.github/workflows/build-push-image.yml
with:
  ref: <git ref to build>
  tag: <image tag>
secrets: inherit
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
