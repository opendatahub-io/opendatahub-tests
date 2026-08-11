# ModelCar Image Build and Push Guide

This document describes how to build and push OCI ModelCar images used by the `opendatahub-tests` integration suite. ModelCar images are minimal OCI containers that package model weights for deployment via KServe `storageUri: oci://...`.

## Prerequisites

- `podman` (or `docker`) installed
- `skopeo` installed (for inspecting image digests)
- `jq` installed (for parsing JSON output)
- Access to `quay.io/opendatahub` push credentials
- Model files prepared locally

### Registry Credentials

The shared robot account for pushing images:

- **Username:** `opendatahub+modelcar`
- **Password:** Request in `#rhoai-devtestops-requests` Channel.

Login:

```bash
podman login quay.io -u "opendatahub+modelcar"
```

## Directory Structure

Create a workspace directory with one subdirectory per image. Each subdirectory contains a `Dockerfile` and a `models/` directory with the model file(s) to package:

```text
<workspace>/
├── modelcar-<runtime>/
│   ├── Dockerfile
│   └── models/
│       └── <model-files>
└── ...
```

## Dockerfile Pattern

All ModelCar images follow the same minimal pattern:

```dockerfile
FROM registry.access.redhat.com/ubi9/ubi-micro@sha256:<digest>

COPY ./models/ /models/

USER 1000
```

Key rules:

- Base image **must** be UBI-micro with `@sha256:` digest pinning
- Model files go under `/models/`
- Run as non-root (`USER 1000`)
- No runtime dependencies -- these are pure data containers

To get the latest UBI-micro digest:

```bash
skopeo inspect docker://registry.access.redhat.com/ubi9/ubi-micro:latest | jq -r '.Digest'
```

## Build and Push

From within the `modelcar-<runtime>/` directory, replace `<version-tag>` with your image version (e.g., `v1`, `v2`, `v1.1`):

**Step 1:** Remove any existing local manifest

```bash
podman manifest rm --ignore quay.io/opendatahub/modelcar-<runtime>:<version-tag>
```

**Step 2:** Build multi-arch manifest

```bash
podman build --no-cache --platform linux/amd64,linux/arm64,linux/s390x,linux/ppc64le \
    -f Dockerfile --manifest quay.io/opendatahub/modelcar-<runtime>:<version-tag>
```

> **Note:** You can add or remove architectures from the `--platform` list as needed.

**Step 3:** Inspect manifest to verify architectures

```bash
podman manifest inspect quay.io/opendatahub/modelcar-<runtime>:<version-tag>
```

**Step 4:** Push all architectures

```bash
podman manifest push --all quay.io/opendatahub/modelcar-<runtime>:<version-tag>
```

**Step 5:** Get the `@sha256:` digest

```bash
skopeo inspect docker://quay.io/opendatahub/modelcar-<runtime>:<version-tag> | jq -r '.Digest'
```

**Example:** Building `modelcar-mlserver-sklearn` version `v2`:

```bash
podman manifest rm --ignore quay.io/opendatahub/modelcar-mlserver-sklearn:v2
podman build --no-cache --platform linux/amd64,linux/arm64,linux/s390x,linux/ppc64le \
    -f Dockerfile --manifest quay.io/opendatahub/modelcar-mlserver-sklearn:v2
podman manifest inspect quay.io/opendatahub/modelcar-mlserver-sklearn:v2
podman manifest push --all quay.io/opendatahub/modelcar-mlserver-sklearn:v2
skopeo inspect docker://quay.io/opendatahub/modelcar-mlserver-sklearn:v2 | jq -r '.Digest'
```

## Update Test References

After pushing new images, update `utilities/image_constants.py` with the new digests. All image references **must** use `@sha256:` digest pinning (no mutable tags). The constants are in the `SharedImages` class.

## Rebuilding Images

Common reasons to rebuild:

1. **Model update** -- new model version or weights
2. **Base image update** -- UBI-micro CVE fix (update the `@sha256:` in `FROM`)
3. **New model format** -- adding a new runtime/model combination

Since ModelCar images contain only data files (no binaries), a single manifest works across all architectures.

> **Note:** After updating any image digest in `utilities/image_constants.py`, always verify that all affected test cases pass before submitting a PR.

## Adding a New ModelCar Image

### Step 1: Create the Quay repository via app-interface

New `quay.io/opendatahub/` repositories are managed through the [app-interface](https://gitlab.cee.redhat.com/service/app-interface) GitLab repo.

1. Clone or update your fork of app-interface:

```bash
git clone git@gitlab.cee.redhat.com:<your-username>/app-interface.git
cd app-interface
git remote add upstream git@gitlab.cee.redhat.com:service/app-interface.git
git fetch upstream && git checkout -b add-modelcar-<runtime> upstream/master
```

2. Edit `data/services/rhoai/quay/opendatahub.yml` and add a new entry:

```yaml
- name: modelcar-<runtime>
  description: '<Runtime> ModelCar image'
  public: true
```

3. Commit and push, then create a Merge Request targeting `master`:

```bash
git add data/services/rhoai/quay/opendatahub.yml
git commit -m "Add quay.io/opendatahub/modelcar-<runtime> repository"
git push origin add-modelcar-<runtime>
```

4. Once the MR is merged, the Quay repository will be created automatically.

### Step 2: Build and push the image

Follow the [Build and Push](#build-and-push) steps above.

### Step 3: Register in opendatahub-tests

1. Add the new constant to `utilities/image_constants.py` under the `SharedImages` class
2. Register the new constant in `scripts/generate_image_manifest.py` if applicable
3. Write or update tests to use the new image
