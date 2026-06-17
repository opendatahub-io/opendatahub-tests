# Notebook Images Tests

Tests for validating notebook container images used by OpenDataHub/RHOAI workbenches.

## N-1 Upgrade Survival (`test_upgrade_n_minus_1.py`)

Verifies that workbenches launched on N-1 (source-version) images remain healthy after a RHOAI platform upgrade.

Representative IDEs:

- JupyterLab (`s2i-minimal-notebook` / `jupyter-minimal-notebook`)
- Code Server (`code-server-notebook`)
- RStudio (`rstudio-rhel9`)

Post-upgrade validation follows the log → HTTP → log sequence from Jan Stourac's criteria (RHAIENG-5382) for JupyterLab and Code Server. RStudio skips the in-container HTTP probe because it serves via nginx, not Jupyter Server. A PVC marker file confirms data survives the upgrade.

### Running

```bash
# Pre-upgrade (on N-1 cluster)
uv run pytest --pre-upgrade tests/workbenches/notebook_images/

# Post-upgrade (on upgraded cluster)
uv run pytest --post-upgrade tests/workbenches/notebook_images/
```

Optional: set `workbench_image_tag` in test config when the ImageStream tag differs from the operator CSV version.

### Notes

- Uses namespace `upgrade-notebook-images` (separate from `upgrade-workbenches` controller tests).
- Code Server is skipped on upstream clusters (`jupyter-minimal-notebook` only).
- RStudio is skipped on upstream clusters. On downstream clusters RStudio uses the ImageStream `latest` tag when no year-based version tags exist. RHOAI 2.25 builds RStudio in-cluster via `rstudio-server-rhel9` BuildConfig (requires `rhel-subscription-secret`); the test triggers that build when needed and waits for the imported image before launching the workbench.
