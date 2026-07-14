# Notebook Images Tests

Tests for validating notebook container images used by OpenDataHub/RHOAI workbenches.

## N-1 Upgrade Survival (`upgrade/`)

Verifies that workbenches launched on N-1 (source-version) images remain healthy after a RHOAI platform upgrade.

Per-IDE test modules:

- `upgrade/test_upgrade_jupyterlab.py` - JupyterLab (`s2i-minimal-notebook` / `jupyter-minimal-notebook`)
- `upgrade/test_upgrade_codeserver.py` - Code Server (`code-server-notebook`)
- `upgrade/test_upgrade_rstudio.py` - RStudio (legacy EUS track only)
- `upgrade/test_upgrade_jupyter_elyra.py` - Elyra Jupyterlab extension (`jupyter-datascience-notebook` / `s2i-generic-data-science-notebook`)

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
- Elyra extension remains installed (Elyra only)
- Elyra runtimes are unchanged and do not get deleted (Elyra only)

### Running

```bash
# Pre-upgrade (on N-1 cluster) — 3 tests (1 per IDE)
uv run pytest --pre-upgrade tests/workbenches/notebook_images/upgrade/

# Post-upgrade (on upgraded cluster) — 27 tests (9 per IDE)
uv run pytest --post-upgrade tests/workbenches/notebook_images/upgrade/

# Target a single IDE
uv run pytest --post-upgrade tests/workbenches/notebook_images/upgrade/test_upgrade_jupyterlab.py

# Target Jupyter workbenches with Elyra
uv run pytest tests/workbenches/notebook_images/upgrade/test_upgrade_jupyter_elyra.py
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
