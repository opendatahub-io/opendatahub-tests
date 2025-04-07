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
  To add a new label, add `/<label name>` in a comment.  
  To remove a label, add `/<label name> cancel` in a comment.  
  `verified` and `lgtm` are removed on new commits.

## How to add a new workflow
1. Create a new file in `.github/workflows` directory.
2. Add relevant steps to the workflow.
3. Code should be implemented in Python and placed in `.github/scripts` directory.
4. Make sure that the workflow is triggered only on relevant events.
5. Set `ACTION` environment variable in the workflow and use it in the code to identify the relevant workflow.


## To be added
- Block merging if not all defined checks pass. For example: a `verified` label was added and at least 2 approvals.
- When a PR is opened, add reviewers (requires updates to OWNERS file(s))
- When a PR is reviewed/commented by a user who's not the PR owner, add `reviewed|commented|approved-by-<username>` label
- When a PR is ready to be merged (all checks passed), add `ready-to-merge` label
- If a label is missing from the repository (i.e was manually deleted), add it back (label colors should be defined as well)
- Tests
