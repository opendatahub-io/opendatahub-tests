# Helpful VS Code Settings and Extensions

The following are some helpful tips on how to set up your VS Code environment for working with this repository.

## Pyrefly Type Checker in Visual Studio Code

If you use Visual Studio Code as your IDE, we recommend using the [Pyrefly](https://marketplace.visualstudio.com/items?itemName=meta.pyrefly) extension.
After installing it, the extension will automatically use the configuration from `pyproject.toml`.

The Pyrefly extension provides:

- Fast inline type checking (15x faster than mypy)
- Auto-completion and hover tooltips
- Go-to-definition and find references
- Type inference without annotations

**Installation:**

```bash
code --install-extension meta.pyrefly
```

Alternatively, use the `.vscode/settings.json` workspace configuration provided in this repository.

## PyCharm Type Checking with Pyrefly

PyCharm 2026.1.2 and later have native Pyrefly integration.

**Enable Pyrefly:**

1. Click the **Type widget** at the bottom of the PyCharm window
2. Select "Use Pyrefly" from the dropdown
3. PyCharm will install Pyrefly automatically if not present

**Note:** Currently works for local interpreter configurations only (Docker, WSL, SSH not yet supported).

The integration provides fast type diagnostics, quick documentation, and inlay hints.

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
