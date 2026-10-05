# Container Image Check Rules

The **PR Container Image Checks** workflow enforces three rules on every PR
that touches Python files. Each rule has a unique code, a fix, and a
suppression comment.

## IMG001: Stray image

**Severity:** Warning (will become a blocking error after all components migrate)

**What:** A hardcoded container image string was found outside of any registered
`image_constants.py` file. Images needed for disconnected testing must be
centralized so they appear in the OCI manifest label on the built container.

**Why:** Disconnected environments use the manifest to discover which images to
mirror. Images not in the manifest won't be available in air-gapped clusters.

**Fix:** Move the image to your component's `image_constants.py` or
`utilities/image_constants.py` if shared across components. Then register the
class in `scripts/generate_image_manifest.py` under `IMAGE_CLASS_MAP`.

**Suppress:** Add `# noqa: IMG001` to the line. Use only for images that are
not needed in disconnected environments (e.g., test-only images that are skipped
on air-gapped clusters).

**Find your stray images:**

```sh
python scripts/check_stray_images.py | grep 'tests/<your_component>/'
```

## IMG002: Missing digest

**Severity:** Error (blocks the PR)

**What:** An image in an `image_constants.py` file uses a mutable `:tag`
reference instead of an immutable `@sha256:` digest pin.

**Why:** Tags can be overwritten at any time. A tag that worked yesterday can
point to a different image today, breaking test reproducibility and potentially
introducing untested behavior.

**Fix:** Replace `:tag` with `@sha256:<digest>`. Get the digest:

```sh
skopeo inspect docker://quay.io/org/image:tag | jq -r .Digest
```

Then use it:

```python
MY_IMAGE: str = "quay.io/org/image@sha256:abc123..."
```

**Suppress:** Add `# noqa: IMG002` to the line. Use only when digest pinning
is genuinely not possible (e.g., the image doesn't publish digests).

## IMG003: DockerHub image

**Severity:** Warning (does not block the PR)

**What:** An image is sourced from `docker.io` (DockerHub).

**Why:** DockerHub enforces strict pull rate limits (100 pulls/6h for
unauthenticated, 200 for free accounts). CI jobs and disconnected mirroring
hit these limits frequently, causing flaky failures.

**Fix:** Use an equivalent image from `quay.io` or `registry.redhat.io` which
have no pull rate limits for authenticated Red Hat users.

**Suppress:** Add `# noqa: IMG003` to the line. Use only when no alternative
registry hosts the image.

## Combining suppressions

Multiple rules can be suppressed on the same line with a comma-separated list:

```python
MY_IMAGE = "docker.io/org/image:v1"  # noqa: IMG001, IMG003
```

## Running locally

```sh
python scripts/check_stray_images.py
python scripts/check_image_digests.py

python scripts/check_stray_images.py --diff-base main
python scripts/check_image_digests.py --diff-base main
```
