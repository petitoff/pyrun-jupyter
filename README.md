# pyrun-jupyter

Run local Python code and projects on a remote Jupyter kernel, then bring the
results back to your machine.

[![PyPI](https://img.shields.io/pypi/v/pyrun-jupyter)](https://pypi.org/project/pyrun-jupyter/)
[![Tests](https://github.com/petitoff/pyrun-jupyter/actions/workflows/test.yml/badge.svg)](https://github.com/petitoff/pyrun-jupyter/actions/workflows/test.yml)
[![Python](https://img.shields.io/pypi/pyversions/pyrun-jupyter)](https://pypi.org/project/pyrun-jupyter/)
[![License: PolyForm Noncommercial](https://img.shields.io/badge/license-PolyForm%20Noncommercial-orange.svg)](LICENSE)

`pyrun-jupyter` is useful when your code lives locally but the compute is
attached to a Jupyter server—for example, a machine with an NVIDIA GPU. It can:

- execute Python source code or a local `.py` file;
- synchronize a multi-file project and run one of its entrypoints;
- inject JSON-compatible parameters into the executed program;
- collect stdout, stderr, rich Jupyter output, and tracebacks;
- download selected artifacts such as checkpoints, plots, or metrics;
- start, connect to, restart, and stop Jupyter kernels;
- be used as either a Python library or a CLI.

> [!IMPORTANT]
> This package is a Jupyter client, not a GPU or Kaggle provisioner. The target
> must expose a reachable, Jupyter Server-compatible HTTP and WebSocket API.
> Enabling a Kaggle accelerator, authenticating to Kaggle, and exposing a
> reachable endpoint are outside this package. A Kaggle notebook page URL alone
> is not necessarily a usable Jupyter API endpoint.

## Requirements

- Python 3.10 or newer on the client;
- a reachable Jupyter Server, JupyterLab, or compatible deployment;
- permission to use `/api/kernels` and kernel WebSocket channels;
- third-party dependencies required by your program already installed in the
  remote kernel environment.

GPU acceleration is provided by the remote environment. `pyrun-jupyter` does
not install CUDA, PyTorch, TensorFlow, drivers, or model dependencies.

## Installation

For use as a library:

```bash
uv add pyrun-jupyter
```

or:

```bash
python -m pip install pyrun-jupyter
```

For an isolated CLI installation:

```bash
uv tool install pyrun-jupyter
pyrun-jupyter --help
```

## Quick start

Set connection details once. Supplying the token through the environment keeps
it out of command history and process arguments.

```bash
export PYRUN_JUPYTER_URL="https://jupyter.example.com"
export PYRUN_JUPYTER_TOKEN="your-secret-token"
```

Run a local project remotely and download its outputs:

```bash
pyrun-jupyter run-project ./trainer train.py \
  --params '{"epochs": 10, "learning_rate": 0.001}' \
  --timeout 7200 \
  --artifact 'outputs/*.safetensors' \
  --artifact 'outputs/metrics.json' \
  --artifact-dir ./artifacts
```

The equivalent library API is:

```python
import os

from pyrun_jupyter import JupyterRunner


with JupyterRunner(
    os.environ["PYRUN_JUPYTER_URL"],
    token=os.environ.get("PYRUN_JUPYTER_TOKEN"),
) as runner:
    result = runner.run_project(
        "./trainer",
        "train.py",
        params={"epochs": 10, "learning_rate": 0.001},
        timeout=7200,
        artifact_paths=[
            "outputs/*.safetensors",
            "outputs/metrics.json",
        ],
        local_artifact_dir="./artifacts",
    )

if result.has_error:
    print(f"{result.error_name}: {result.error}")
else:
    print(result.stdout)
    print(result.data["artifacts"])
```

## Running a project

`run_project()` is the recommended interface for normal Python projects. It
uploads the directory to a clean remote workspace, adds that workspace to
`sys.path`, changes the remote working directory, executes the entrypoint as
`__main__`, and optionally downloads artifacts.

```text
trainer/
├── train.py
├── model.py
├── pyproject.toml
└── outputs/
```

Imports such as `from model import Model` continue to work after synchronization.
The default exclusions include `.git`, virtual environments, Python caches,
test caches, and `*.egg-info` directories. Add project-specific exclusions from
the CLI by repeating `--exclude`:

```bash
pyrun-jupyter run-project ./trainer train.py \
  --exclude data \
  --exclude '*.ckpt' \
  --artifact 'outputs/**/best.ckpt'
```

Artifact paths may be exact paths or recursive glob patterns relative to the
remote project root. Their directory structure is preserved under
`--artifact-dir`.

### Parameters

CLI parameters can be JSON or comma-separated `key=value` pairs:

```bash
pyrun-jupyter run-project ./trainer train.py --params '{"epochs": 20, "amp": true}'
pyrun-jupyter run-project ./trainer train.py --params 'epochs=20,amp=true'
```

They are available as globals in the entrypoint:

```python
# train.py
print(f"Training for {epochs} epochs; mixed precision: {amp}")
```

For complex nested values, prefer JSON.

## Running code and individual files

```bash
pyrun-jupyter run "import torch; print(torch.cuda.is_available())" --timeout 120
pyrun-jupyter run-file ./scripts/evaluate.py --params '{"split": "test"}' --timeout 1800
```

```python
from pyrun_jupyter import JupyterRunner


with JupyterRunner("https://jupyter.example.com", token="...") as runner:
    result = runner.run("print(6 * 7)")
    print(result.stdout)

    evaluation = runner.run_file(
        "./scripts/evaluate.py",
        params={"split": "test"},
        timeout=1800,
    )
```

Use `run_project()` instead of `run_file()` when the entrypoint imports other
local modules or relies on local files and relative paths.

## CLI reference

```text
pyrun-jupyter run CODE [connection options]
pyrun-jupyter run-file FILE [--params PARAMS] [connection options]
pyrun-jupyter run-project PROJECT_DIR ENTRYPOINT [project options] [connection options]
```

Common connection options:

| Option | Environment variable | Default |
| --- | --- | --- |
| `--url` | `PYRUN_JUPYTER_URL` | required |
| `--token` | `PYRUN_JUPYTER_TOKEN` | none |
| `--kernel` | — | `python3` |
| `--timeout` | — | `60` seconds |

Run `pyrun-jupyter COMMAND --help` for the complete list of command-specific
options. Commands exit with status `0` on successful remote execution and `1`
on connection, validation, timeout, or execution failure.

## Execution results and errors

Library calls return an `ExecutionResult`; errors reported by the remote Python
process are represented in the result instead of being raised.

| Attribute | Description |
| --- | --- |
| `success` / `has_error` | Remote execution status |
| `stdout` / `stderr` | Captured stream output |
| `error_name` / `error` | Remote exception type and message |
| `error_traceback` | Remote traceback lines |
| `data` | Rich `execute_result` output and downloaded artifact paths |
| `display_data` | Additional Jupyter display outputs |
| `execution_count` | Jupyter execution counter |

Connection, kernel-management, and file-transfer failures raise subclasses of
`PyrunJupyterError` (`ConnectionError`, `KernelError`, `ExecutionError`,
`FileTransferError`, and `RemoteFileNotFoundError`):

```python
from pyrun_jupyter import JupyterRunner, PyrunJupyterError


try:
    with JupyterRunner("https://jupyter.example.com", token="...") as runner:
        result = runner.run("1 / 0")
except PyrunJupyterError as exc:
    print(f"Jupyter connection failed: {exc}")
else:
    if result.has_error:
        print("\n".join(result.error_traceback))
```

## Kernel lifecycle

The context manager starts a kernel and stops it when the block exits:

```python
with JupyterRunner("https://jupyter.example.com", token="...") as runner:
    print(runner.kernel_id)
    runner.restart_kernel()
```

To connect to a kernel explicitly:

```python
runner = JupyterRunner(
    "https://jupyter.example.com",
    token="...",
    auto_start_kernel=False,
)
runner.connect_to_kernel("existing-kernel-id")
result = runner.run("print('connected')")
runner.stop_kernel()
```

`stop_kernel()` terminates the remote kernel, not only the client connection.

## Lower-level file transfer

The public API also exposes Jupyter Contents API helpers:

```python
runner.upload_file("./data.csv", "data/input.csv")
runner.upload_directory("./src", "project/src", pattern="**/*.py")
runner.download_file("outputs/model.pt", "./artifacts/model.pt")
runner.download_files(["outputs/model.pt", "outputs/metrics.json"], "./artifacts")
```

Kernel-based helpers are available when the Contents API does not map to the
kernel filesystem:

```python
runner.upload_via_kernel("./dataset.parquet", "project/data/dataset.parquet")
runner.upload_directory_via_kernel("./src", "project/src")
runner.download_kernel_files(
    ["model.pt", "metrics.json"],
    local_dir="./artifacts",
    working_dir="project/outputs",
    flatten=False,
)
```

Both directions use the same transport: raw binary WebSocket buffers in 4 MiB
chunks, carried over a dedicated Jupyter Comm rather than encoded into source
code or program output. Neither side ever holds more than a single chunk in
memory, and no payload is base64-encoded.

- **Upload.** Each chunk is acknowledged by the kernel before the next is sent.
  The kernel writes to a temporary file, verifies the complete transfer against
  a SHA-256 sent with the final message, and only then replaces the destination
  atomically.
- **Download.** The client pulls one chunk at a time, so the kernel never queues
  more than a single buffer. Bytes land in a local temporary file, are verified
  against the size and SHA-256 the kernel reports on completion, and only then
  replace the destination atomically.

An aborted or failed transfer leaves no partial file on either side. You can
tune the chunk size and the per-chunk timeout for a particular proxy:

```python
runner.upload_via_kernel(
    "./large-model.safetensors",
    "models/large-model.safetensors",
    chunk_size=8 * 1024 * 1024,
    timeout=300,
)

runner.download_kernel_files(
    ["outputs/large-model.safetensors"],
    local_dir="./artifacts",
    chunk_size=8 * 1024 * 1024,
    timeout=300,
)
```

A file that does not exist on the kernel raises `RemoteFileNotFoundError`, a
subclass of `FileTransferError`. `download_kernel_files()` catches it per file,
warns, and continues, so a missing artifact does not abandon the rest of the
batch.

## Operational notes

- Kernel transfers are binary in both directions: bounded raw chunks, never
  base64, and never the complete file in memory. This is the path used by
  `run_project()`, so project synchronization and artifact retrieval both stream.
- The `upload_file()` / `download_file()` / `upload_directory()` /
  `download_files()` helpers go through the Jupyter Contents API, whose REST
  representation for binary content is base64 (`{"format": "base64"}`). That
  encoding is the API's, not a fallback chosen here, and it does buffer the whole
  file. Prefer the kernel helpers above for large artifacts.
- Project dependencies are not synchronized or installed automatically.
- The default project workspace is recreated for every `run_project()` call.
  Treat a custom `remote_dir` as managed, disposable storage.
- On timeout, the client stops waiting and sends a best-effort interrupt to the
  kernel. Remote termination is not guaranteed by every Jupyter deployment.
- A single `JupyterRunner` instance is intended for sequential execution.
- Use HTTPS/WSS for remote servers and give tokens only the permissions needed
  for kernel and file operations.

## Development

The repository uses [`uv`](https://docs.astral.sh/uv/) and commits `uv.lock` for
repeatable development and CI environments.

```bash
git clone https://github.com/petitoff/pyrun-jupyter.git
cd pyrun-jupyter
uv sync --extra dev --locked

uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src/pyrun_jupyter --ignore-missing-imports
uv build
uv run twine check dist/*
```

When dependencies change, update the lockfile with `uv lock` and commit it with
`pyproject.toml`.

## Release status

The package is currently classified as **Alpha**. Before relying on it for
unattended or high-value workloads, test it against the exact Jupyter deployment,
proxy, authentication method, artifact sizes, and expected runtime you use.

## License

Source code distributed with this release is available under the
[PolyForm Noncommercial License 1.0.0](LICENSE). Noncommercial, academic, and
educational use is permitted under its terms. Commercial use requires a separate
written license from the copyright holder.

This is a source-available license, not an OSI-approved open-source license.
Earlier copies received under the MIT License remain governed by the license
that accompanied those copies.
