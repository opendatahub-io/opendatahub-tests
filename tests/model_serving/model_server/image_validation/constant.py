"""Constants for model server image validation tests."""

from utilities.constants import RuntimeTemplates

# Placeholder storage URI so the model server controller creates the Deployment/Pod with
# the model server and storage-initializer images. No actual model or inference is
# required; pod phase does not need to be Ready.
PLACEHOLDER_STORAGE_URI = "s3://dummy-bucket/dummy/"

# Namespace used by the model server image validation tests.
MODEL_SERVER_IMAGE_VALIDATION_NAMESPACE = "model-server-image-validation"

# Model server runtime template used to create the InferenceService predictor pod.
MODEL_SERVER_RUNTIME_TEMPLATE = RuntimeTemplates.OVMS_KSERVE

# Init container injected by the kserve controllers to download the model from storage;
# its image is one of the images required to be validated by RHOAIENG-23498.
STORAGE_INITIALIZER_CONTAINER_NAME = "storage-initializer"

# Control plane deployments created by the `kserve` DSC component in the applications
# namespace. Deployed by the Model Server and Metrics team (Serving Orchestration).
KSERVE_CONTROL_PLANE_DEPLOYMENTS: list[dict[str, str]] = [
    {"name": "odh-model-controller", "label_selector": "control-plane=odh-model-controller"},
    {"name": "kserve-controller-manager", "label_selector": "control-plane=kserve-controller-manager"},
]
