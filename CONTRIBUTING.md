# Welcome to opendatahub-tests contributing guide

Thank you for contributing to our project!  

## New contributor guide

To get an overview of the project, read the [README](README.md).

## Issues

### Create a new issue

If you find a problem with the code, [search if an issue already exists](https://github.com/opendatahub-io/opendatahub-tests/issues).  
If you open a pull request to fix the problem, an issue will ba automatically created.  
If a related issue doesn't exist, you can open a new issue using a relevant [issue form](https://github.com/opendatahub-io/opendatahub-tests/issues/new/choose).

## Pull requests

To contribute code to the project:

- Fork the project and work on your forked repository
- Before submitting a new pull request, make sure you have [pre-commit](https://pre-commit.com/) package and installed

```bash
pre-commit install
```

- When submitting a pull request, make sure to fill all the required, relevant fields for your PR.  
  Make sure the title is descriptive and short.

## General

- Add typing to new code; typing is enforced using [mypy](https://mypy-lang.org/)
  - Rules are defined in [our pyproject.toml file](//pyproject.toml#L10)

## Check the code
Checks tools are used to check the code are defined in .pre-commit-config.yaml file
To install pre-commit:

```bash
pip install pre-commit --user
pre-commit install
pre-commit install --hook-type commit-msg
```

Run pre-commit:

```bash
pre-commit run --all-files
```

## Adding new runtime
To add a new runtime, you need to:  
1. Add a new file under [manifests](utilities/manifests) directory.
2. Add `<runtime>_INFERENCE_CONFIG` dict with:
```code
    "support_multi_default_queries": True|False,  # Optioanl, if set to True, `default_query_model` should contains a dict with corresponding inference_type
    "default_query_model": {
        "query_input": <default query to be sent to the model>,
        "query_output": <expected output>,
        "use_regex": True|False, # Optional, if set to True, `query_output` should be a regex
    },
    "<query type, for example: all-tokens>": {
        "<protocol, for example HTTP>": {
            "endpoint": "<model endpoint>",
            "header": "<model required headers>",
            "body": '{<model expected body}',
            "response_fields_map": {
                "response_output": <output field in response>,
                "response": <response field in response - optional>,
            },
        },
```
3. See [caikit_standalone](utilities/manifests/caikit_standalone.py) for an example
