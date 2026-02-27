# llmd_configs

One config class per LLMInferenceService test scenario. Each class is the single source of truth for its deployment.

## Hierarchy

```
LLMISvcConfig (config_base.py)              # Base — defaults, helpers
├── CpuConfig (config_base.py)              # CPU image, env, resources
│   ├── TinyLlamaOciConfig                  # OCI storage
│   ├── TinyLlamaS3Config                   # S3 storage
│   └── Opt125mHfConfig                     # HuggingFace storage
└── GpuConfig (config_base.py)              # GPU resources
    ├── QwenS3Config                        # S3 storage
    │   ├── PrefillDecodeConfig             # prefill-decode disaggregation
    │   └── EstimatedPrefixCacheConfig      # estimated prefix cache
    └── QwenHfConfig                        # HuggingFace storage
        └── PrecisePrefixCacheConfig        # precise prefix cache
```

Model+storage classes are in `config_models.py`. Feature configs are in their own files.

## Usage

```python
@pytest.mark.parametrize("llmisvc", [TinyLlamaOciConfig], indirect=True)
def test_something(self, llmisvc):
    ...
```

Override inline with `with_overrides()`:

```python
@pytest.mark.parametrize("llmisvc", [TinyLlamaOciConfig.with_overrides(replicas=2)], indirect=True)
```
