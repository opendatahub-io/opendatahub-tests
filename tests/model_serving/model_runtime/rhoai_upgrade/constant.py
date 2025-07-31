import os

from pytest_testconfig import config as py_config

WORKING_TESTS_DIR: str = os.path.dirname(__file__)

APPLICATIONS_NAMESPACE: str = py_config["applications_namespace"]

SERVING_RUNTIME_TEMPLATE_NAME: str = "kserve-ovms-serving-runtime-template"

SERVING_RUNTIME_INSTANCE_NAME: str = "kserve-ovms-serving-runtime-instance"

SERVING_RUNTIME_TEMPLATE_FILE_NAME: str = "kserve_ovms_runtime_template.yaml"

SERVING_RUNTIME_TEMPLATE_FILE_PATH: str = os.path.join(WORKING_TESTS_DIR, SERVING_RUNTIME_TEMPLATE_FILE_NAME)
