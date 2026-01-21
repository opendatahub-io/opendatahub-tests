# Project Structure

The project is structured as follows:
- [tests](../tests): Base directory for pytest tests
  - Each component has its own directory
  - Each feature has its own directory
- [utilities](../utilities): Base directory for utility functions
  - Each module contains a set of utility functions related to a specific topic, for example:  
    - [infra](../utilities/infra.py): Infrastructure-related (cluster resources) utility functions
    - [constants](../utilities/constants.py): Constants used in the project
- [docs](../docs): Documentation
- [py_config](../tests/global_config.py) contains tests-specific configuration which can be controlled from the command line.  

Please refer to [pytest-testconfig](https://github.com/wojole/pytest-testconfig) for more information.


# Contribution
To contribute code to the project:

## Pull requests
- Fork the project and work on your forked repository
- Before submitting a new pull request:
  - Make sure you follow the [Style guide](STYLE_GUIDE.md)
  - Make sure you have [pre-commit](https://pre-commit.com/) package installed
  - Make sure you have [tox](https://tox.readthedocs.io/en/latest/) package installed
- PRs that are not ready for review (but needed to be pushed for any reason) should have [WIP] in the title and labelled as "wip".
  - When a PR is ready for review, remove the [WIP] from the title and remove the "wip" label.
- PRs should be relatively small; if needed the PRs should be split and depended on each other.
  - Small PRs will get quicker review.
  - Small PRs comments will be fixed quicker and merged quicker.
  - Both the reviewer and the committer will benefit from this.
- When a refactor is needed as part of the current PR, the refactor should be done in another PR and the current PR should be rebased on it.
- Please address each comment in code review
  - If a comment was addressed and accepted, please comment as done and resolve.
  - If a comment was addressed and rejected or additional discussion is needed, add your input and do not resolve the comment.  
  - To
  minimize the number of comments, please try to address all comments in one PR.
- Before a PR can be merged:
  - PRs must be verified and marked with "verified" label.
  - PRs must be reviewed by at least two reviewers other than the committer.
  - All CI checks must pass.

## Branching strategy
The project follows RHOAI [release lifecyle strategy](https://access.redhat.com/support/policy/updates/rhoai-sm/lifecycle).  
If needed, once your PR is merged to `main`, cherry-pick your PR to the relevant branch(es).


## Python
- Reduce duplicate code, before writing new function search for it, probably someone already wrote it or one that should serve your needs.
  - The project uses external packages that may already have a functionality that does what you need.
- When using a variable more than once save it and reuse.
- Keep functions and fixtures close to where they're used, if needed move them later for more modules to use them.
- Call functions using argument names to make it clear what is being passed and easier refactoring.
- Imports: Always use absolute paths
- Imports: when possible, avoid importing a module but rather import specific functions
- Do not import from `conftest.py` files. These files must contain fixtures only, and not utility functions, constants etc.
- Flexible code is good, but:
  - Should not come at the expense of readability; remember that someone else will need to look/use/maintain the code.
  - Do not prepare code for the future just because it may be useful.
  - Every function, variable, fixture, etc. written in the code - must be used, or else removed.
- Log enough to make you debug and understand the flow easy, but do not spam the log with unuseful info.  
Error logs should be detailed with what failed, status and so on.


## Interacting with Kubernetes/OpenShift APIs
The project utilizes [openshift-python-wrapper](https://github.com/RedHatQE/openshift-python-wrapper).
Please refer to the [documentation](https://github.com/RedHatQE/openshift-python-wrapper/blob/main/README.md)  
and the [examples](https://github.com/RedHatQE/openshift-python-wrapper/tree/main/examples) for more information.

For any missing resources, please generate a new resource using
[class_generator tool](https://github.com/RedHatQE/openshift-python-wrapper/blob/main/class_generator/README.md) and
create a PR against wrapper. Calls to cluster resources from tests, utils and fixtures must always use
openshift-python-wrapper resource or oc command
(when wrapper resource is not relevant. e.g. must-gather generation)


## Conftest
- Top level [conftest.py](../conftest.py) contains pytest native fixtures.
- General tests [conftest.py](../tests/conftest.py) contains fixtures that are used in multiple tests by multiple teams.
- If needed, create new `conftest.py` files in the relevant directories.


## Fixtures
- Ordering: Always call pytest native fixtures first, then session-scoped fixtures and then any other fixtures.
- Fixtures should handle setup (and the teardown, if needed) needed for the test(s), including the creation of resources for example.
- Fixtures should do one thing only.  
For example, instead of:

```python
@pytest.fixture()
def model_inference_service():
    with ServingRuntime(name=...) as serving_runtime:
      with InferenceService(name=..) as inference_service:
        yield inference_service
```

Do:

```python
@pytest.fixture()
def model_runtime():
    with ServingRuntime(name=...) as serving_runtime:
      yield serving_runtime

@pytest.fixture(model_runtime)
def model_inference_service(model_runtime):
    with InferenceService(name=..) as inference_service:
        yield inference_service

```

- Pytest reports failures in fixtures as ERROR
- A fixture name should be a noun that describes what the fixture provides (i.e. returns or yields), rather than a verb.  
For example:  
  - If a test needs a storage secret, the fixture should be called 'storage_secret' and not 'create_secret'.
  - If a test needs a directory to store user data, the fixture should be called 'user_data_dir' and not 'create_directory'.
  - If a vllm test needs a serving runtime, the fixture should be called 'vllm_serving_runtime' and not 'create_serving_runtime'.
- Note fixture scope, test execution times can be reduced by selecting the right scope.  
Pytest default fixture invocation is "function", meaning the code in the fixture will be executed every time the fixture is called.  
Broader scopes (class, module etc) will invoke the code only once within the given scope and all tests within the scope will use the same instance.
- Use request.param to pass parameters from test/s to fixtures; use a dict structure for readability.  For example:

```code
@pytest.mark.parametrize(
"storage_secret",
[
pytest.param(
{"name": "my-secret", "model-dir": "/path/to/model"},
),

def test_secret(storage_secret):

    pass

@pytest.fixture()
def storage_secret(request):
secret = Secret(name=request.param["name"], model_dir=request.param["model-dir"])
```


## Tests
- Pytest reports failures in fixtures as FAILED
- Each test should have a clear purpose and should be easy to understand.
- Each test should verify a single aspect of the product.
- Preferably, each test should be independent of other tests.
- When there's a dependency between tests use pytest dependency plugin to mark the relevant hierarchy between tests (https://github.com/RKrahl/pytest-dependency)
- When adding a new test, apply relevant marker(s) which may apply.  
Check [pytest.ini](../pytest.ini) for available markers; additional markers can always be added when needed.
- Classes are good to group related tests together, for example when they share a fixture.  
You should NOT group unrelated tests in one class (because it is misleading the reader).
- All the tests should be properly documented. Every test (or test class), should have a docstring explaning what the test does so that anyone (engineers from other components, managers, PMs, or non-technical users) can have a basic understanding of what the code is trying to test without having to dive into the technical details of related functions or fixtures.


## Check the code
### pre-commit

When submitting a pull request, make sure to fill all the required, relevant fields for your PR.  
Make sure the title is descriptive and short.  
Checks tools are used to check the code are defined in .pre-commit-config.yaml file
To install pre-commit:

```bash
pre-commit install -t pre-commit -t commit-msg
```

Run pre-commit:

```bash
pre-commit run --all-files
```

### tox
CI uses [tox](https://tox.readthedocs.io/en/latest/) and will run the code under tox.ini  

Run tox:

```bash
tox
```

## Adding new runtime
To add a new runtime, you need to:  
1. Add a new file under [manifests](../utilities/manifests) directory.
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
3. See [caikit_standalone](../utilities/manifests/caikit_standalone.py) for an example

## AI Usage
If using AI tooling to assist you in the process of writing or reviewing code:
1. Understand what you are doing --as a developer, you are ultimately responsible for the code. Always assume the code produced by the AI tools is unsafe and incorrect, and always double-check it.
2. We support [AGENTS.md](../AGENTS.md), an [open format](https://agents.md/) for guiding coding agents. If you use any proprietary tool that does not support `AGENTS.md` (e.g. Claude Code, Qwen Code, Gemini Code), you can create a symlink:
```bash
ln -s AGENTS.md CLAUDE.md
```
