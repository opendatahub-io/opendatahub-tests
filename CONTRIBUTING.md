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

If you use Visual Studio Code as your IDE, we recommend using the [Mypy Type Checker](https://marketplace.visualstudio.com/items?itemName=ms-python.mypy-type-checker) extension.
After installing it, make sure to update the `Mypy-type-checkers: Args` setting
to `"mypy-type-checker.args" = ["--config-file=pyproject.toml"]`.

### Debugging in Visual Studio Code

If you use Visual Studio Code and want to debug your test execution with its "Run and Debug" feature, you'll want to use
a `launch.json` file similar to this one:

```
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "justMyCode": false,  #set to false if you want to debug dependent libraries too
            "name": "uv_pytest_debugger",
            "type": "debugpy",
            "request": "launch",
            "program": ".venv/bin/pytest",  #or your path to pytest's bin in the venv
            "python": "${command:python.interpreterPath}",  #make sure uv's python interpreter is selected in vscode
            "console": "integratedTerminal",
            "args": "path/to/test.py"  #the args for pytest, can be a list, in this example runs a single file
        }
    ]
}
```
