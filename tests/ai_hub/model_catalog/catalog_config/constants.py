import time

MODEL_WITH_SINGLE_TAG: str = "test/model-single-hw-tag"
MODEL_WITH_EMPTY_TAG: str = "test/model-empty-hw-tag"
MODEL_WITHOUT_TAG: str = "test/model-no-hw-tag"

SINGLE_HW_TAG: str = "Hardware Tag1"
_CURRENT_TIME = int(time.time() * 1000)
CUSTOM_YAML_WITH_HARDWARE_TAGS: str = f"""source: Custom Hardware Tag Test
models:
- name: {MODEL_WITH_SINGLE_TAG}
  description: Model with a hardware tag.
  readme: |-
    # Single hardware tag model
  provider: Test Provider
  logo: placeholder
  license: apache-2.0
  licenseLink: https://www.apache.org/licenses/LICENSE-2.0.txt
  tasks:
    - text-generation
  customProperties:
    hardware_tag:
      metadataType: MetadataStringValue
      string_value: "{SINGLE_HW_TAG}"
  artifacts:
    - uri: oci://registry.example.io/test-single-hw:1.0
  createTimeSinceEpoch: "{_CURRENT_TIME - 10000!s}"
  lastUpdateTimeSinceEpoch: "{_CURRENT_TIME!s}"

- name: {MODEL_WITH_EMPTY_TAG}
  description: Model with an empty hardware tag.
  readme: |-
    # Empty hardware tag model
  provider: Test Provider
  logo: placeholder
  license: apache-2.0
  licenseLink: https://www.apache.org/licenses/LICENSE-2.0.txt
  tasks:
    - text-generation
  customProperties:
    hardware_tag:
      metadataType: MetadataStringValue
      string_value: ""
  artifacts:
    - uri: oci://registry.example.io/test-empty-hw:1.0
  createTimeSinceEpoch: "{_CURRENT_TIME - 10000!s}"
  lastUpdateTimeSinceEpoch: "{_CURRENT_TIME!s}"

- name: {MODEL_WITHOUT_TAG}
  description: Model without any hardware tag.
  readme: |-
    # No hardware tag model
  provider: Test Provider
  logo: placeholder
  license: apache-2.0
  licenseLink: https://www.apache.org/licenses/LICENSE-2.0.txt
  tasks:
    - text-generation
  artifacts:
    - uri: oci://registry.example.io/test-no-hw:1.0
  createTimeSinceEpoch: "{_CURRENT_TIME - 10000!s}"
  lastUpdateTimeSinceEpoch: "{_CURRENT_TIME!s}"
"""
