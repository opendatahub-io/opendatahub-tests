# Pytest args handling
PYTEST_ARGS ?= tests --tc-file=tests/global_config.py --tc-format=python

#OPENSHIFT_PYTHON_WRAPPER LOG LEVEL
export OPENSHIFT_PYTHON_WRAPPER_LOG_LEVEL=DEBUG


IMAGE_BUILD_CMD = $(shell which podman 2>/dev/null || which docker)
IMAGE_REGISTRY ?= "quay.io"
REGISTRY_NAMESPACE ?= "redhat_msi"
OPERATOR_IMAGE_NAME="opendatahub-tests"
IMAGE_TAG ?= "latest"

FULL_OPERATOR_IMAGE ?= "$(IMAGE_REGISTRY)/$(REGISTRY_NAMESPACE)/$(OPERATOR_IMAGE_NAME):$(IMAGE_TAG)"
POETRY_BIN = poetry

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
