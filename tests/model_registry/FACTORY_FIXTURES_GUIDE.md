# Model Registry Factory Fixtures Guide

This guide explains how to use the new factory fixtures for Model Registry testing, which address the issues of long dependency chains, limited parametrization, and single-instance restrictions.

## Overview

The factory fixtures provide a flexible way to create Model Registry instances with:
- **Multiple instances**: Create multiple registries with unique names
- **Custom configurations**: Easily customize settings without complex fixture chains
- **Reduced dependencies**: Self-contained factory functions
- **Parametrization**: Easy configuration through data classes

## Quick Start

### Basic Usage

```python
import pytest
from pytest_testconfig import config as py_config
from tests.model_registry.conftest import create_dsc_component_patch

@pytest.mark.parametrize(
    "updated_dsc_component_state_scope_class",
    [
        pytest.param(
            create_dsc_component_patch(py_config["model_registry_namespace"]),
            id="basic-test"
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("updated_dsc_component_state_scope_class")
class TestMyModelRegistry:
    def test_single_registry(self, default_model_registry_factory):
        # Create a registry with default settings
        registry = default_model_registry_factory("my-test")

        # Use the registry
        assert registry.instance.name is not None
        assert registry.rest_endpoint is not None

    def test_multiple_registries(self, default_model_registry_factory):
        # Create multiple registries
        registry1 = default_model_registry_factory("test-1")
        registry2 = default_model_registry_factory("test-2")

        # They have different names and endpoints
        assert registry1.instance.name != registry2.instance.name
        assert registry1.rest_endpoint != registry2.rest_endpoint
```

## Available Factory Fixtures

### Core Factories

#### `model_registry_db_factory`
Creates database bundles (Service, PVC, Secret, Deployment).

```python
def test_custom_db(self, model_registry_db_factory, model_registry_namespace):
    config = ModelRegistryDBConfig(
        name_prefix="custom-db",
        namespace=model_registry_namespace,
        storage_size="10Gi"
    )
    db_bundle = model_registry_db_factory(config)
    assert db_bundle.deployment.name.startswith("custom-db")
```

#### `model_registry_instance_factory`
Creates complete Model Registry instances with their dependencies.

```python
def test_custom_instance(self, model_registry_instance_factory, model_registry_namespace):
    config = ModelRegistryConfig(
        name="custom-mr",
        namespace=model_registry_namespace,
        use_oauth_proxy=False,
        use_istio=True,
        mysql_config=ModelRegistryDBConfig(
            name_prefix="custom-db",
            namespace=model_registry_namespace
        )
    )
    registry = model_registry_instance_factory(config)
    assert registry.config.use_istio is True
```

#### `model_registry_client_factory`
Creates Model Registry clients for given endpoints.

```python
def test_with_client(self, default_model_registry_factory, model_registry_client_factory):
    registry = default_model_registry_factory("test")
    client = model_registry_client_factory(registry.rest_endpoint)
    # Use client for API calls
```

### Convenience Factories

#### `default_model_registry_factory`
Creates registries with standard OAuth configuration.

```python
def test_default_config(self, default_model_registry_factory):
    registry = default_model_registry_factory("my-test")
    assert registry.config.use_oauth_proxy is True
```

#### `oauth_model_registry_factory`
Creates registries with OAuth proxy configuration.

#### `istio_model_registry_factory`
Creates registries with Istio configuration.

#### `simple_model_registry_factory`
Simplified interface with override support.

```python
def test_simple_overrides(self, simple_model_registry_factory):
    registry = simple_model_registry_factory({
        "name": "simple-test",
        "storage_size": "8Gi",
        "use_oauth_proxy": False
    })
```

#### `multi_instance_factory`
Creates multiple instances at once.

```python
def test_batch_creation(self, multi_instance_factory):
    instances = multi_instance_factory(3, "batch-test")
    assert len(instances) == 3
    # All instances have unique names
```

#### `standalone_db_factory`
Creates standalone database instances.

## Configuration Classes

### `ModelRegistryDBConfig`
Configures database resources.

```python
db_config = ModelRegistryDBConfig(
    name_prefix="my-db",
    namespace="test-namespace",
    storage_size="10Gi",
    mysql_image="custom-mysql:8.0",
    port=3306,
    teardown=True  # Clean up after test
)
```

### `ModelRegistryConfig`
Configures Model Registry instances.

```python
mr_config = ModelRegistryConfig(
    name="my-registry",
    namespace="test-namespace",
    use_oauth_proxy=True,
    use_istio=False,
    mysql_config=db_config,
    teardown=True
)
```

## DSC Setup Requirements

**Critical**: All tests using factory fixtures must properly set up the DSC (Data Science Cluster) component to enable Model Registry and configure the namespace.

### Required Pattern

```python
@pytest.mark.parametrize(
    "updated_dsc_component_state_scope_class",
    [
        pytest.param(
            create_dsc_component_patch(py_config["model_registry_namespace"]),
            id="test-id"
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("updated_dsc_component_state_scope_class")
class TestMyClass:
    # Your tests here
```

### Helper Functions

#### `create_dsc_component_patch(namespace)`
Creates DSC component patch for given namespace.

#### `ModelRegistryTestHelper`
Provides common parametrization patterns.

```python
# Standard parametrization
params = ModelRegistryTestHelper.get_standard_dsc_parametrization("my-namespace")

# OAuth vs Istio testing
params = ModelRegistryTestHelper.get_oauth_vs_istio_parametrization("my-namespace")

# Multiple namespaces
params = ModelRegistryTestHelper.create_multi_namespace_parametrization([
    "namespace-1", "namespace-2", "namespace-3"
])
```

## Bundle Classes

### `ModelRegistryDBBundle`
Contains all database resources with cleanup methods.

```python
db_bundle = model_registry_db_factory(config)
# Access components
db_bundle.service
db_bundle.pvc
db_bundle.secret
db_bundle.deployment

# Get MySQL config
mysql_config = db_bundle.get_mysql_config()

# Cleanup (automatic via teardown, or manual)
db_bundle.cleanup()
```

### `ModelRegistryInstanceBundle`
Contains Model Registry instance and related resources.

```python
registry = model_registry_instance_factory(config)
# Access components
registry.instance          # ModelRegistry resource
registry.db_bundle         # Database bundle
registry.service           # Service
registry.rest_endpoint     # REST endpoint URL
registry.grpc_endpoint     # gRPC endpoint URL
registry.config            # Original configuration

# Cleanup
registry.cleanup()
```

## Context Managers

For temporary resources that need guaranteed cleanup:

```python
from tests.model_registry.conftest import temporary_model_registry

def test_with_context_manager(self, model_registry_instance_factory):
    config = ModelRegistryConfig(name="temp-mr", namespace="test-ns")

    with temporary_model_registry(config, model_registry_instance_factory) as registry:
        # Registry will be automatically cleaned up
        client = ModelRegistryClient(...)
        # ... test logic ...
    # Registry is cleaned up here
```

## Migration from Existing Fixtures

### Before (Old Pattern)
```python
@pytest.mark.parametrize(
    "updated_dsc_component_state_scope_class",
    [pytest.param({"component_patch": {...}})],
    indirect=True,
)
@pytest.mark.usefixtures("updated_dsc_component_state_scope_class")
class TestOldPattern:
    def test_single_instance(
        self,
        model_registry_instance,
        model_registry_client,
        model_registry_instance_rest_endpoint,
    ):
        # Limited to single instance
        # Long dependency chain
        pass
```

### After (New Pattern)
```python
@pytest.mark.parametrize(
    "updated_dsc_component_state_scope_class",
    [pytest.param(create_dsc_component_patch(py_config["model_registry_namespace"]))],
    indirect=True,
)
@pytest.mark.usefixtures("updated_dsc_component_state_scope_class")
class TestNewPattern:
    def test_multiple_instances(
        self,
        default_model_registry_factory,
        model_registry_client_factory,
    ):
        # Create multiple instances
        registry1 = default_model_registry_factory("test-1")
        registry2 = default_model_registry_factory("test-2")

        # Create clients
        client1 = model_registry_client_factory(registry1.rest_endpoint)
        client2 = model_registry_client_factory(registry2.rest_endpoint)

        # Test both independently
```

## Common Patterns

### Testing Multiple Configurations

```python
@pytest.mark.parametrize(
    *ModelRegistryTestHelper.get_oauth_vs_istio_parametrization(py_config["model_registry_namespace"])
)
@pytest.mark.usefixtures("updated_dsc_component_state_scope_class", "is_model_registry_oauth")
class TestMultipleConfigs:
    def test_auth_configurations(
        self,
        model_registry_instance_factory,
        model_registry_namespace,
        is_model_registry_oauth,
    ):
        use_oauth = is_model_registry_oauth.get("use_oauth_proxy", True)
        config = ModelRegistryConfig(
            name="auth-test",
            namespace=model_registry_namespace,
            use_oauth_proxy=use_oauth,
            use_istio=not use_oauth,
        )
        registry = model_registry_instance_factory(config)
        # Test the configuration
```

### Testing Scale

```python
def test_scale_testing(self, multi_instance_factory):
    # Create many instances to test scale
    instances = multi_instance_factory(10, "scale-test")

    # Test all instances
    for i, instance in enumerate(instances):
        assert f"scale-test-{i}" in instance.instance.name
        # ... perform tests ...
```

### Testing Custom Resources

```python
def test_custom_resources(self, model_registry_instance_factory, model_registry_namespace):
    config = ModelRegistryConfig(
        name="custom-test",
        namespace=model_registry_namespace,
        mysql_config=ModelRegistryDBConfig(
            name_prefix="custom-db",
            namespace=model_registry_namespace,
            storage_size="20Gi",
            mysql_image="custom-mysql:8.0",
        )
    )
    registry = model_registry_instance_factory(config)
    # Test custom configuration
```

## Best Practices

1. **Always use DSC parametrization**: Every test class must properly set up the DSC component
2. **Use unique names**: Provide unique name prefixes to avoid conflicts
3. **Leverage bundles**: Use the bundle objects to access related resources
4. **Cleanup automatically**: Use `teardown=True` for automatic cleanup
5. **Use context managers**: For temporary resources that need guaranteed cleanup
6. **Test multiple scenarios**: Use the helper functions for common parametrization patterns

## Example Test File

See `tests/model_registry/test_factory_fixtures_example.py` for a complete example demonstrating all patterns and usage scenarios.

## Troubleshooting

### Common Issues

1. **DSC not parametrized**: Ensure you use the correct DSC parametrization pattern
2. **Namespace mismatch**: Make sure the namespace in your config matches the DSC setup
3. **Name conflicts**: Use unique name prefixes to avoid resource conflicts
4. **Missing teardown**: Set `teardown=True` to ensure cleanup

### Debug Tips

- Check the test logs for resource creation/cleanup messages
- Verify the DSC component is properly enabled
- Ensure the namespace exists and is active
- Use `kubectl get` commands to check resource states
