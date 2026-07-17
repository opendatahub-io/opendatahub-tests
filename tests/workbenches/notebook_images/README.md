# Notebook Images Tests

Tests for validating notebook container images used by OpenDataHub/RHOAI workbenches.

## N-1 Upgrade Survival (`upgrade/`)

Verifies that workbenches launched on N-1 (source-version) images remain healthy after a RHOAI platform upgrade.

A single parameterized test module covers all IDEs:

- `upgrade/test_upgrade_workbench.py` -- JupyterLab, Code Server, RStudio, Elyra (parameterized via `get_workbench_image_specs()`)
- `upgrade/test_upgrade_jupyter_elyra.py` -- Elyra-specific extension and runtime config checks

Pre-upgrade validation creates dashboard-faithful Notebook CRs, waits for controller reconciliation (kube-rbac-proxy, ReferenceGrant, HTTPRoute), captures a rich baseline (image selection, digest, restart counts, Notebook generation), and writes a PVC marker file.

Post-upgrade validation checks:

- Pod not recreated (creationTimestamp preserved)
- Image selection annotation unchanged
- Running container digest unchanged
- Container restart counts unchanged
- Notebook CR generation unchanged
- StatefulSet health (readyReplicas, no pending rollout)
- PVC marker file still readable
- Log cleanliness and in-pod HTTP health (JupyterLab and Code Server only)
- Jupyter kernel in-memory state survived (JupyterLab only)
- Elyra extensions and runtime configs preserved (Elyra only)

## Dashboard Image Bump (`upgrade/test_bump_jupyterlab.py`)

After the platform upgrade, applies the same JSON patch the Dashboard uses to bump a JupyterLab workbench from N-1 to N, then verifies the workbench restarts healthy with PVC data intact.

Post-upgrade tests: workbench exists, JSON patch applied + pod rollout, image annotation + digest updated, log/HTTP health, PVC marker file survived.

### Running

```bash
# Pre-upgrade (on N-1 cluster)
uv run pytest --pre-upgrade tests/workbenches/notebook_images/upgrade/

# Post-upgrade (on upgraded cluster)
uv run pytest --post-upgrade tests/workbenches/notebook_images/upgrade/

# Target a single IDE via keyword
uv run pytest --post-upgrade tests/workbenches/notebook_images/upgrade/ -k jupyterlab
```

Optional overrides via pytest-testconfig:

```bash
# Pin a specific ImageStream tag (useful on already-upgraded clusters)
uv run pytest --pre-upgrade tests/workbenches/notebook_images/upgrade/ --tc workbench_image_tag=3.4

# Force stable (3.x major.minor) or legacy EUS (year.release) tag selection
uv run pytest --pre-upgrade tests/workbenches/notebook_images/upgrade/ --tc workbench_upgrade_track=stable
```

### Notes

- Uses namespace `upgrade-notebook-images` (separate from `upgrade-workbenches` controller tests).
- Code Server and RStudio are skipped on upstream clusters.
- RStudio runs only on the EUS upgrade track when a legacy RStudio ImageStream is present.
- RStudio skips the in-container HTTP probe (serves via nginx, not Jupyter Server).
- On clusters where RStudio images are built in-cluster, the test can instantiate `rstudio-server-rhel9` BuildConfig when the tag is not yet imported.
