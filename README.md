# py2jupyter

Execute Python `.py` files on remote Jupyter servers.

[![PyPI version](https://badge.fury.io/py/py2jupyter.svg)](https://pypi.org/project/py2jupyter/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-petitoff%2Fpy2jupyter-blue)](https://github.com/petitoff/py2jupyter)

## Installation

```bash
pip install py2jupyter
```

## Quick Start

```python
from py2jupyter import JupyterRunner

# Connect to your Jupyter server
runner = JupyterRunner("http://localhost:8888", token="your_token")

# Run a .py file on the remote server
result = runner.run_file("script.py")
print(result.stdout)

# Or run code directly
result = runner.run("print('Hello from Jupyter!')")
print(result.stdout)
```

## Features

- 🐍 Execute standard `.py` files on remote Jupyter kernels
- 📤 Pass parameters to your scripts
- 📥 Capture stdout, stderr, and rich outputs
- 🔄 Kernel management (start, stop, restart)
- 🔌 Connect to existing kernels
- ⚡ Context manager support for automatic cleanup

## Usage Examples

### Basic Usage

```python
from py2jupyter import JupyterRunner

runner = JupyterRunner("http://jupyter-server:8888", token="xxx")

# Execute code
result = runner.run("x = 42; print(f'The answer is {x}')")
print(result.stdout)  # The answer is 42

# Clean up
runner.stop_kernel()
```

### Using Context Manager (Recommended)

```python
from py2jupyter import JupyterRunner

with JupyterRunner("http://localhost:8888", token="xxx") as runner:
    result = runner.run_file("my_script.py")
    print(result.stdout)
# Kernel automatically stopped
```

### Passing Parameters to Scripts

```python
from py2jupyter import JupyterRunner

with JupyterRunner("http://localhost:8888", token="xxx") as runner:
    # Parameters are injected as variables in your script
    result = runner.run_file(
        "train_model.py",
        params={
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 32,
        }
    )
    print(result.stdout)
```

Your `train_model.py` can use these variables directly:

```python
# train_model.py
print(f"Training with lr={learning_rate}, epochs={epochs}")
# ... your training code
```

### Handling Errors

```python
from py2jupyter import JupyterRunner, ExecutionError

runner = JupyterRunner("http://localhost:8888", token="xxx")

result = runner.run("1/0")
if result.has_error:
    print(f"Error: {result.error_name}: {result.error}")
    # Error: ZeroDivisionError: division by zero
```

### Connecting to Existing Kernel

```python
runner = JupyterRunner("http://localhost:8888", token="xxx", auto_start_kernel=False)

# List available kernels
kernels = runner.list_kernels()
print(kernels)

# Connect to a specific kernel
runner.connect_to_kernel("existing-kernel-id")
result = runner.run("print('Using existing kernel!')")
```

### Managing Kernels

```python
runner = JupyterRunner("http://localhost:8888", token="xxx")

# Start a specific kernel type
runner.start_kernel("python3")

# Restart kernel (clears state)
runner.restart_kernel()

# Stop kernel when done
runner.stop_kernel()
```

## ExecutionResult

The `run()` and `run_file()` methods return an `ExecutionResult` object:

| Attribute | Type | Description |
|-----------|------|-------------|
| `stdout` | str | Standard output |
| `stderr` | str | Standard error |
| `success` | bool | Whether execution succeeded |
| `error` | str | Error message (if failed) |
| `error_name` | str | Exception type (e.g., 'ValueError') |
| `error_traceback` | list | Full traceback |
| `data` | dict | Rich output (text/plain, text/html, etc.) |
| `execution_count` | int | Jupyter cell execution count |

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `url` | (required) | Jupyter server URL |
| `token` | None | Authentication token |
| `kernel_name` | "python3" | Kernel specification to use |
| `auto_start_kernel` | True | Start kernel automatically on first run |

## Getting Your Jupyter Token

### From Jupyter Notebook/Lab

When you start Jupyter, it displays a URL with the token:
```
http://localhost:8888/?token=abc123...
```

### Generate a permanent token

```bash
jupyter server --generate-config
# Edit ~/.jupyter/jupyter_server_config.py
# Set: c.ServerApp.token = 'your-secret-token'
```

## Requirements

- Python >= 3.8
- A running Jupyter server (Notebook, Lab, or Hub)

## License

MIT