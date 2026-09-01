"""
pyrun-jupyter - Execute Python code on remote Jupyter servers.

Main exports:
    JupyterRunner - Remote Jupyter code executor
        Methods:
            run(code) -> ExecutionResult
            run_file(path, params) -> ExecutionResult
            run_project(project_dir, entrypoint, artifact_paths) -> ExecutionResult
            upload_directory_via_kernel(local_dir, remote_dir) -> list[str]
            download_kernel_files(paths, local_dir, working_dir) -> list[Path]
            upload_file(local, remote)
            download_file(remote, local)

    ExecutionResult - Code execution result
        Attributes:
            stdout, stderr, success, has_error, error, error_name, error_traceback, data

Usage:
    from pyrun_jupyter import JupyterRunner

    with JupyterRunner("http://localhost:8888", token="xxx") as runner:
        result = runner.run("print('hello')")
        print(result.stdout)
"""

from .contents import FileTransferError as FileTransferError
from .exceptions import ConnectionError as ConnectionError
from .exceptions import ExecutionError as ExecutionError
from .exceptions import KernelError as KernelError
from .exceptions import PyrunJupyterError as PyrunJupyterError
from .result import ExecutionResult as ExecutionResult
from .runner import JupyterRunner as JupyterRunner

__version__: str
__all__ = [
    "ConnectionError",
    "ExecutionError",
    "ExecutionResult",
    "FileTransferError",
    "JupyterRunner",
    "KernelError",
    "PyrunJupyterError",
]
