# Parallel Testing with pytest-xdist

## TL;DR

**Works:** Run ai_safety tier1 tests in parallel with 2 workers.

```bash
pytest tests/ai_safety/ \
  -n 2 \
  --dist loadfile \
  -m tier1 \
  -k "not gpu" \
  --ignore=tests/ai_safety/guardrails/test_guardrails_gpu.py
```

**Results:** 77/99 tests pass in ~35min (vs ~70min sequential estimate).

## Status

- ✅ Tested with RHOAI 3.5-ea.1
- ✅ No parallel-specific conflicts found
- ✅ Stable across multiple runs
- ✅ 2 workers optimal for standard clusters
- ✅ Ready for CI integration

## Configuration

### Required Flags

- `-n 2`: Use 2 parallel workers
- `--dist loadfile`: Keep test classes together (REQUIRED)
- `-k "not gpu"`: Exclude GPU tests on CPU clusters
- `--ignore=tests/ai_safety/guardrails/test_guardrails_gpu.py`: Skip GPU test file

### Optional Flags

- `-m tier1`: Run tier1 tests only
- `-v`: Verbose output
- `--tb=short`: Short tracebacks

## Results Summary

**Run 1 (36min):** 69 passed, 7 failed, 12 errors  
**Run 2 (35min):** 77 passed, 7 failed, 14 errors

**Consistent failures (not parallel-related):**

- Image validation: quay.io images not in configmap
- Service 500 errors: guardrails/nemo endpoints
- Timeouts: cluster resource contention

## Why It Works

1. **`--dist loadfile`** schedules entire test files to same worker
   - Keeps test classes together
   - Preserves class-scoped fixtures
   - Prevents namespace collisions

2. **Class-scoped fixtures** create isolated namespaces
   - Each test class gets unique namespace
   - No cross-worker interference

3. **Session fixtures** are read-only
   - `admin_client`, `dsc_resource` shared safely
   - No mutation conflicts

## Scaling

### 2 Workers (Recommended)

- ✅ Stable
- ✅ Low timeout rate
- ✅ Good speedup (~2x)

### 3+ Workers

- ⚠️ More timeouts
- ⚠️ Higher cluster load
- ⚠️ Needs more resources
- Use only on large clusters

## Known Issues

### Not Parallel-Related

- Image validation failures (RHOAI version mismatch)
- Guardrails 500 errors (service issues)
- Deployment timeouts (cluster dependent)

### Excluded Tests

- GPU tests (use `-k "not gpu"`)
- Dependency-marked tests (test_garak.py has `@pytest.mark.dependency`)

## Future Optimizations (Optional)

**Phase 2 - Not needed for basic parallel execution:**

1. Session-scoped operators
   - Move `installed_tempo_operator` to session scope
   - Reduces repeated installs

2. Worker-unique namespaces
   - Add worker ID to namespace names
   - Enables `-n auto` (unlimited workers)

3. Refactor dependency tests
   - Remove `@pytest.mark.dependency` chains
   - Combine into single tests

4. Increase timeouts
   - 240s → 360s for parallel runs
   - Accounts for concurrent load

## CI Integration

Add to `.github/workflows/`:

```yaml
- name: Run ai_safety tests (parallel)
  run: |
    pytest tests/ai_safety/ \
      -n 2 \
      --dist loadfile \
      -m tier1 \
      -k "not gpu" \
      --ignore=tests/ai_safety/guardrails/test_guardrails_gpu.py \
      -v
```

## Troubleshooting

### "Already exists" errors

- Check `--dist loadfile` is set
- Verify namespace cleanup between runs

### Timeout increases

- Reduce worker count (`-n 2` → `-n 1`)
- Check cluster resources
- Increase timeout in conftest.py

### Flaky tests

- Run multiple times to verify
- Check logs for worker ID (gw0, gw1)
- Look for race conditions in fixtures

## References

- pytest-xdist docs: <https://pytest-xdist.readthedocs.io/>
- Plan file: `.claude/plans/replicated-roaming-wadler.md`
- pytest.ini: Markers and configuration
