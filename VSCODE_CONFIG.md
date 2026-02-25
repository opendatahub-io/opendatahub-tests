# Helpful VS Code Settings and Extensions

The following are some helpful tips on how to set up your VS Code environment for working with this repository.

## Mypy Type Checker in Visual Studio Code

If you use Visual Studio Code as your IDE, we recommend using the [Mypy Type Checker](https://marketplace.visualstudio.com/items?itemName=ms-python.mypy-type-checker) extension.
After installing it, make sure to update the `Mypy-type-checkers: Args` setting
to `"mypy-type-checker.args" = ["--config-file=pyproject.toml"]`.

## Debugging in Visual Studio Code

If you use Visual Studio Code and want to debug your test execution with its "Run and Debug" feature, you'll want to use
a `launch.json` file similar to this one:

```json
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
