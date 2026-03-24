# Cluster Health Tests

This directory contains foundational health check tests for OpenDataHub/RHOAI clusters. These tests serve as prerequisites to ensure the cluster and operators are in a healthy state before running more complex integration tests.

## Directory Structure

```text
cluster_health/
├── test_cluster_health.py      # Cluster node health validation
└── test_operator_health.py     # Operator and pod health validation
```

### Current Test Suites

- **`test_cluster_health.py`** - Validates that all cluster nodes are healthy and schedulable
- **`test_operator_health.py`** - Validates that DSCInitialization, DataScienceCluster resources are ready, and all pods in operator/application namespaces are running

## Test Markers

Tests use the following markers defined in `pytest.ini`:

- `@pytest.mark.cluster_health` - Tests that verify the cluster is healthy to begin testing
- `@pytest.mark.operator_health` - Tests that verify OpenDataHub/RHOAI operators are healthy and functioning correctly

## Test Details

### Cluster Node Health (`test_cluster_health.py`)

- **`test_cluster_node_healthy`** - Asserts all cluster nodes have `KubeletReady: True` condition and are schedulable (not cordoned)

### Operator Health (`test_operator_health.py`)

- **`test_data_science_cluster_initialization_healthy`** - Validates the DSCInitialization resource reaches `READY` status (120s timeout)
- **`test_data_science_cluster_healthy`** - Validates the DataScienceCluster resource reaches `READY` status (120s timeout)
- **`test_pods_cluster_healthy`** - Validates all pods in operator and application namespaces reach Running/Completed state (180s timeout). Parametrized across `operator_namespace` and `applications_namespace` from global config

## Running Tests

### Run All Cluster Health Tests

```bash
uv run pytest tests/cluster_health/
```

### Run by Marker

```bash
# Run cluster node health tests
uv run pytest -m cluster_health

# Run operator health tests
uv run pytest -m operator_health

# Run both
uv run pytest -m "cluster_health or operator_health"
```
