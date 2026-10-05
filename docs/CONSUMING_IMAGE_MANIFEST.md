# Consuming the Image Manifest

The `odh-tests` container image embeds a manifest of all container images required by tests
as OCI labels. Disconnected or air-gapped environments can extract this manifest to discover
which images need to be mirrored.

## Labels

| Label                                          | Content                                    |
| ---------------------------------------------- | ------------------------------------------ |
| `io.opendatahub.tests.required-images`         | Compact JSON manifest grouped by component |
| `io.opendatahub.tests.required-images.sha256`  | SHA-256 hex digest of the manifest JSON    |

## Prerequisites

- `skopeo`
- `jq`
- `sha256sum` (coreutils) or `shasum -a 256` on macOS

## Extracting the image list

Inspect the image and extract the manifest label. The command exits with an error if
the label is missing:

```sh
IMAGE=quay.io/opendatahub/opendatahub-tests:latest
LABEL=io.opendatahub.tests.required-images
INSPECT=$(skopeo inspect "docker://$IMAGE")
MANIFEST=$(printf '%s' "$INSPECT" | jq -r ".Labels[\"$LABEL\"]")

if [ "$MANIFEST" = "null" ] || [ -z "$MANIFEST" ]; then
    echo "ERROR: no image manifest label found on $IMAGE"
    exit 1
fi
```

### Option 1: JSON (grouped by component)

Saves the manifest as pretty-printed JSON with images grouped by component:

```sh
printf '%s' "$MANIFEST" | jq . > required-images.json
```

Output:

```json
{
  "ai_safety": [
    "quay.io/minio/minio@sha256:14cea4...",
    "quay.io/trustyai_testing/vllm_emulator@sha256:c4bdd5..."
  ],
  "shared": [
    "ghcr.io/project-zot/zot:v2.1.8",
    "quay.io/opendatahub/openvino_model_server@sha256:564664..."
  ]
}
```

### Option 2: Plain text (one image per line)

Flattens all images across components into a plain list, one per line:

```sh
printf '%s' "$MANIFEST" | jq -r '.[][]' > required-images.txt
```

Output:

```text
quay.io/minio/minio@sha256:14cea4...
quay.io/trustyai_testing/vllm_emulator@sha256:c4bdd5...
ghcr.io/project-zot/zot:v2.1.8
quay.io/opendatahub/openvino_model_server@sha256:564664...
```

## Verifying the checksum

After extracting the manifest, verify it was not corrupted or tampered with.
If the image was built before checksum support was added, the checksum label
will be missing and verification is skipped.

On macOS, replace `sha256sum` with `shasum -a 256` in the commands below:

```sh
EXPECTED=$(printf '%s' "$INSPECT" | jq -r ".Labels[\"$LABEL.sha256\"]")

if [ "$EXPECTED" = "null" ] || [ -z "$EXPECTED" ]; then
    echo "SKIP: no checksum label found -- image was built without checksum support"
else
    ACTUAL=$(printf '%s' "$MANIFEST" | sha256sum | cut -d' ' -f1)
    if [ "$ACTUAL" = "$EXPECTED" ]; then
        echo "OK: checksum verified ($ACTUAL)"
    else
        echo "FAIL: checksum mismatch (expected=$EXPECTED, actual=$ACTUAL)"
    fi
fi
```

## Full example

Extract the manifest, verify the checksum, and save both output formats:

```sh
IMAGE=quay.io/opendatahub/opendatahub-tests:latest
LABEL=io.opendatahub.tests.required-images

INSPECT=$(skopeo inspect "docker://$IMAGE")
MANIFEST=$(printf '%s' "$INSPECT" | jq -r ".Labels[\"$LABEL\"]")

if [ "$MANIFEST" = "null" ] || [ -z "$MANIFEST" ]; then
    echo "ERROR: no image manifest label found on $IMAGE"
    exit 1
fi

EXPECTED=$(printf '%s' "$INSPECT" | jq -r ".Labels[\"$LABEL.sha256\"]")

if [ "$EXPECTED" = "null" ] || [ -z "$EXPECTED" ]; then
    echo "SKIP: no checksum label found -- image was built without checksum support"
else
    ACTUAL=$(printf '%s' "$MANIFEST" | sha256sum | cut -d' ' -f1)
    if [ "$ACTUAL" = "$EXPECTED" ]; then
        echo "OK: checksum verified ($ACTUAL)"
    else
        echo "FAIL: checksum mismatch (expected=$EXPECTED, actual=$ACTUAL)"
        exit 1
    fi
fi

printf '%s' "$MANIFEST" | jq .    > required-images.json
printf '%s' "$MANIFEST" | jq -r '.[][]' > required-images.txt

echo "Wrote required-images.json and required-images.txt"
```
