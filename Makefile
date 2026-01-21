IMAGE_BUILD_CMD = $(shell which podman 2>/dev/null || which docker)
IMAGE_REGISTRY ?= "quay.io"
REGISTRY_NAMESPACE ?= "opendatahub"
IMAGE_NAME="opendatahub-tests"
IMAGE_TAG ?= "3.3"

FULL_OPERATOR_IMAGE ?= "$(IMAGE_REGISTRY)/$(REGISTRY_NAMESPACE)/$(IMAGE_NAME):$(IMAGE_TAG)"

all: check

check:
	python3 -m pip install pip tox --upgrade
	tox

build:
	$(IMAGE_BUILD_CMD) build -t $(FULL_OPERATOR_IMAGE) .

push:
	$(IMAGE_BUILD_CMD) push $(FULL_OPERATOR_IMAGE)

build-and-push-container: build push

.PHONY: \
	check \
	build \
	push \
	build-and-push-container \
