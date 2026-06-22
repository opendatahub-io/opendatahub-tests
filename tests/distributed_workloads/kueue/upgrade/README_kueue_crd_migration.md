# Kueue CRD v1beta1→v1beta2 Migration Tests

This directory contains tests for Kueue Custom Resource Definition (CRD) migration from v1beta1 to v1beta2 during RHOAI upgrades.

## Test Overview

The Kueue CRD migration tests address **Gap G3** from RHOAIENG-63117 audit report by verifying that:

1. **Pre-upgrade**: Kueue resources can be created using v1beta1 API
2. **Post-upgrade**: Resources are accessible via v1beta2 API with all fields preserved
3. **Conversion**: Bidirectional API compatibility (v1beta1↔v1beta2) works correctly
4. **Functional**: Workload admission behavior remains intact after upgrade

## Test Files

### Core Migration Tests

- **`test_kueue_crd_migration.py`** - Main migration test suite
  - Pre-upgrade: Creates ClusterQueue, LocalQueue, ResourceFlavor using v1beta1 API
  - Post-upgrade: Validates v1beta2 conversion and field preservation
  - Baseline comparison: Ensures no data loss during migration

### Advanced Webhook Tests

- **`test_kueue_webhook_conversion.py`** - Conversion webhook testing
  - Round-trip conversion compatibility (v1beta1→v1beta2→v1beta1)
  - Edge cases: complex data structures, unicode, special characters
  - Concurrent operations testing
  - Field defaults and status preservation

### Test Configuration

- **`conftest.py`** - Kueue-specific pytest fixtures and configuration
- **`__init__.py`** - Module initialization

### Utilities

- **`utilities/kueue_utils_v1beta1.py`** - v1beta1 resource classes and helpers

## Test Resources Created

### ResourceFlavor (v1beta1)

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: migration-test-cpu-flavor
spec:
  nodeLabels:
    node.kubernetes.io/instance-type: test-gpu-node
    topology.zone: test-zone-a
  tolerations:
    - key: nvidia.com/gpu
      operator: Equal
      value: "true"
      effect: NoSchedule
```

### ClusterQueue (v1beta1)

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: migration-test-cluster-queue
spec:
  namespaceSelector:
    matchLabels:
      kueue-migration-test: enabled
  stopPolicy: Hold
  cohort: migration-test-cohort
  resourceGroups:
    - coveredResources: [cpu, memory]
      flavors:
        - name: test-cpu-flavor
          resources:
            - name: cpu
              nominalQuota: "4"
            - name: memory
              nominalQuota: "8Gi"
```

### LocalQueue (v1beta1)

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: migration-test-local-queue
  namespace: kueue-crd-migration-test
spec:
  clusterQueue: migration-test-cluster-queue
```

## Running the Tests

### Prerequisites

1. RHOAI cluster with Kueue operator installed
2. Admin and unprivileged kubeconfig access
3. pytest with upgrade test infrastructure configured

### Pre-upgrade Tests

```bash
# Run Kueue CRD migration pre-upgrade tests
pytest tests/distributed_workloads/kueue/upgrade/test_kueue_crd_migration.py \
  --pre-upgrade \
  -v

# Run conversion webhook pre-upgrade tests
pytest tests/distributed_workloads/kueue/upgrade/test_kueue_webhook_conversion.py \
  --pre-upgrade \
  -v
```

### Post-upgrade Tests

```bash
# Run Kueue CRD migration post-upgrade tests
pytest tests/distributed_workloads/kueue/upgrade/test_kueue_crd_migration.py \
  --post-upgrade \
  -v

# Run conversion webhook post-upgrade tests
pytest tests/distributed_workloads/kueue/upgrade/test_kueue_webhook_conversion.py \
  --post-upgrade \
  -v
```

### Combined Upgrade Test (Full Cycle)

```bash
# Pre-upgrade phase
pytest tests/distributed_workloads/kueue/upgrade/test_kueue_*.py \
  --pre-upgrade \
  -v

# << PERFORM RHOAI UPGRADE HERE >>

# Post-upgrade phase
pytest tests/distributed_workloads/kueue/upgrade/test_kueue_*.py \
  --post-upgrade \
  -v
```

## Test Validation

### Pre-upgrade Validation

- ✅ Resources created successfully with v1beta1 API
- ✅ Resource specifications captured in baseline ConfigMap
- ✅ Functional behavior verified (ClusterQueue Active, workload admission)

### Post-upgrade Validation

- ✅ Resources accessible via v1beta2 API
- ✅ All fields preserved during migration (compared against baseline)
- ✅ Bidirectional API compatibility (both v1beta1 and v1beta2 work)
- ✅ Functional behavior maintained (admission, queue policies)
- ✅ Conversion webhook handles edge cases correctly

## Integration with CI Pipeline

The tests integrate with the existing opendatahub-tests upgrade infrastructure:

### Test Discovery

- Tests marked with `@pytest.mark.pre_upgrade` run during pre-upgrade phase
- Tests marked with `@pytest.mark.post_upgrade` run during post-upgrade phase
- Dependency management ensures proper test execution order

### Resource Management

- `teardown_resources` fixture controls cleanup based on `SKIP_RESOURCE_TEARDOWN` env var
- Baseline data persisted in ConfigMap survives upgrade for comparison
- Namespace `kueue-crd-migration-test` labeled for Kueue management

### Error Handling

- Must-gather collection on test failures
- Comprehensive logging for debugging migration issues
- Graceful handling of missing baselines or CRD unavailability

## Debugging Migration Issues

### Common Issues

**Issue**: Pre-upgrade tests fail with "Kueue CRDs not available"

```bash
# Solution: Verify Kueue operator installation
kubectl get crd -l app.kubernetes.io/name=kueue
kubectl get pods -n openshift-kueue-operator
```

**Issue**: Post-upgrade tests fail with "Baseline ConfigMap not found"

```bash
# Solution: Verify pre-upgrade tests created baseline
kubectl get cm kueue-crd-migration-baseline -n kueue-crd-migration-test
kubectl describe cm kueue-crd-migration-baseline -n kueue-crd-migration-test
```

**Issue**: Field comparison fails after upgrade

```bash
# Solution: Check conversion webhook logs
kubectl logs -n openshift-kueue-operator -l control-plane=controller-manager
```

### Must-Gather Collection

```bash
# Collect debugging information on failure
pytest tests/distributed_workloads/kueue/upgrade/test_kueue_crd_migration.py \
  --post-upgrade \
  --collect-must-gather \
  -v
```

## Acceptance Criteria

- ✅ **Task**: CRD v1beta1→v1beta2 migration test created and functional
- ✅ **API Compatibility**: Both v1beta1 and v1beta2 APIs work post-upgrade
- ✅ **Field Preservation**: All resource fields maintained during conversion
- ✅ **Functional Behavior**: Workload admission and queue policies work post-upgrade
- ✅ **CI Integration**: Tests run in upgrade pipeline with proper dependencies
- ✅ **Edge Cases**: Conversion webhook handles complex data structures correctly
- ✅ **Documentation**: Comprehensive test documentation and debugging guide

## Contributing

When modifying these tests:

1. **Follow existing patterns**: Use established fixture and baseline patterns
2. **Update both test files**: Migration tests and webhook tests should stay in sync
3. **Test edge cases**: Consider unicode, special characters, and large configurations
4. **Update documentation**: Keep this README current with any changes
5. **Validate CI integration**: Ensure tests run correctly in upgrade pipeline

## Related Files

- `utilities/kueue_utils.py` - v1beta2 Kueue utilities (existing)
- `utilities/kueue_utils_v1beta1.py` - v1beta1 Kueue utilities (new)
- `tests/workbenches/notebooks_server/controller/upgrade/conftest.py` - Upgrade test fixtures
- `conftest.py` - Global upgrade test configuration
- `[RHOAIENG-63117] Kueue Upgrade Test Coverage Audit Report.md` - Original audit report
