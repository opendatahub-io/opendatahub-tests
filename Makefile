IMAGE_BUILD_CMD = $(shell which podman 2>/dev/null || which docker)
IMAGE_REGISTRY ?= "quay.io"
REGISTRY_NAMESPACE ?= "opendatahub"
IMAGE_NAME="opendatahub-tests"
IMAGE_TAG ?= "latest"

# CPU build flag - set to "true" to build CPU image
# This will be passed from Jenkins as: make build CPU=true
CPU ?= false

# Conditional Dockerfile and tag selection based on CPU flag
ifeq ($(CPU),true)
    DOCKERFILE = Dockerfile.CPU
    EFFECTIVE_IMAGE_TAG = power
else
    DOCKERFILE = Dockerfile
    EFFECTIVE_IMAGE_TAG = $(IMAGE_TAG)
endif

FULL_OPERATOR_IMAGE ?= "$(IMAGE_REGISTRY)/$(REGISTRY_NAMESPACE)/$(IMAGE_NAME):$(EFFECTIVE_IMAGE_TAG)"

all: check

check:
	python3 -m pip install pip tox --upgrade
	tox

build:
	$(IMAGE_BUILD_CMD) build -f $(DOCKERFILE) -t $(FULL_OPERATOR_IMAGE) .

push:
	$(IMAGE_BUILD_CMD) push $(FULL_OPERATOR_IMAGE)

build-and-push-container: build push

.PHONY: \
	check \
	build \
	push \
	build-and-push-container \
