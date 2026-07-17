IMAGE_BUILD_CMD = $(shell which podman 2>/dev/null || which docker)
IMAGE_REGISTRY ?= "quay.io"
REGISTRY_NAMESPACE ?= "opendatahub"
IMAGE_NAME="opendatahub-tests"
IMAGE_TAG ?= "latest"

FULL_OPERATOR_IMAGE ?= "$(IMAGE_REGISTRY)/$(REGISTRY_NAMESPACE)/$(IMAGE_NAME):$(IMAGE_TAG)"

all: check

check:
	python3 -m pip install pip tox --upgrade
	tox

build:
	@MANIFEST=$$(uv run python scripts/generate_image_manifest.py --compact 2>/tmp/manifest-err.log); \
	if [ -n "$$MANIFEST" ]; then \
		CHECKSUM=$$(printf '%s' "$$MANIFEST" | shasum -a 256 | cut -d' ' -f1); \
		echo "Image manifest generated (sha256:$$CHECKSUM)"; \
		$(IMAGE_BUILD_CMD) build \
			--label "io.opendatahub.tests.required-images=$$MANIFEST" \
			--label "io.opendatahub.tests.required-images.sha256=$$CHECKSUM" \
			-t $(FULL_OPERATOR_IMAGE) .; \
	else \
		echo "WARNING: Image manifest generation failed — building without manifest labels"; \
		if [ -s /tmp/manifest-err.log ]; then cat /tmp/manifest-err.log >&2; fi; \
		$(IMAGE_BUILD_CMD) build -t $(FULL_OPERATOR_IMAGE) .; \
	fi

push:
	$(IMAGE_BUILD_CMD) push $(FULL_OPERATOR_IMAGE)

build-and-push-container: build push

.PHONY: \
	check \
	build \
	push \
	build-and-push-container \
