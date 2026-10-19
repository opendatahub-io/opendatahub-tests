# Notebook Images Tests

Tests for validating notebook container images used by OpenDataHub/RHOAI workbenches.

## N-1 Upgrade Survival (`upgrade/`)

Verifies that workbenches launched on the source-version image remain healthy after a RHOAI platform upgrade.

A single parameterized test module covers the IDEs available on this branch:

- `upgrade/test_upgrade_workbench.py` -- JupyterLab, JupyterLab with Elyra, and Code Server (parameterized via `get_workbench_image_specs()`)
- `upgrade/test_upgrade_jupyter_elyra.py` -- Elyra extension and runtime-config checks on the datascience image
- `upgrade/test_bump_jupyterlab.py` -- Dashboard-equivalent image bump from the source tag to the current tag

Pre-upgrade validation creates a Notebook CR with `notebooks.opendatahub.io/inject-oauth`, waits until the controller injects the `oauth-proxy` sidecar, captures a baseline (image selection, digest, restart counts, Notebook generation, pod identity), and writes a PVC marker file.

Post-upgrade validation checks that the running workbench was not rolled:

- Pod not recreated (`creationTimestamp` and pod UID preserved)
- Image selection annotation unchanged
- Running container digest unchanged
- Container restart counts unchanged
- Notebook CR generation unchanged
- StatefulSet health (`readyReplicas`, no pending rollout)
- PVC marker file still readable
- Log cleanliness and in-pod HTTP health
- Jupyter kernel in-memory state survived (JupyterLab and Elyra)
- Elyra extensions and runtime configs preserved (`test_upgrade_jupyter_elyra.py`)

This contract applies to a z-stream upgrade (for example 2.25.8 to 2.25.9) and to a 2.x to 3.x upgrade before auth migration restarts the workbench.

## Dashboard Image Bump (`upgrade/test_bump_jupyterlab.py`)

After the platform upgrade, applies the same JSON patch the Dashboard uses to bump a JupyterLab workbench from the source image to the current image, then verifies the workbench restarts healthy with PVC data intact.

On 2.x the current ImageStream tag is the newest `YYYY.N` tag (for example `2025.2`), not `{major}.{minor}`. A z-stream upgrade therefore resolves the same image before and after the upgrade. The bump test skips in that case because the patch is a no-op and the pod is not recreated. Pin an older tag during pre-upgrade to exercise a real bump:

```bash
uv run pytest --pre-upgrade tests/workbenches/notebook_images/upgrade/ --tc workbench_image_tag=2025.1
```

`workbench_image_tag` selects the source image only. The bump target always follows the running product: newest `YYYY.N` on 2.x, or `{major}.{minor}` on 3.x.

The digest check accepts the ImageStream manifest-list digest and the platform child digest from `dockerImageManifests`.

### Running

```bash
# Pre-upgrade (on the source cluster)
uv run pytest --pre-upgrade tests/workbenches/notebook_images/upgrade/

# Post-upgrade (on the upgraded cluster)
uv run pytest --post-upgrade tests/workbenches/notebook_images/upgrade/

# Target a single IDE via keyword
uv run pytest --post-upgrade tests/workbenches/notebook_images/upgrade/ -k jupyterlab
```

Optional overrides via pytest-testconfig (`key=value`):

```bash
# Pin the source ImageStream tag
uv run pytest --pre-upgrade tests/workbenches/notebook_images/upgrade/ --tc workbench_image_tag=2025.1

# Force the legacy EUS (year.release) tag selection for the source image
uv run pytest --pre-upgrade tests/workbenches/notebook_images/upgrade/ --tc workbench_upgrade_track=eus
```

On product major less than 3 the default track is already `eus`, which selects the newest `YYYY.N` tag.

### Notes

- Uses namespace `upgrade-notebook-images` (separate from `upgrade-workbenches` controller tests).
- Code Server is skipped on upstream clusters.
- When the integrated image registry is unavailable, image resolution falls back to the digest-pinned `dockerImageReference`.
