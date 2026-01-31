"""
pyrun-jupyter - Execute Python code on remote Jupyter servers.

Main exports:
    JupyterRunner - Remote Jupyter code executor
        Methods:
            run(code) -> ExecutionResult
            run_file(path, params) -> ExecutionResult
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

from .runner import JupyterRunner
from .result import ExecutionResult
from .contents import FileTransferError
from .exceptions import (
    PyrunJupyterError,
    ConnectionError,
    KernelError,
    ExecutionError,
)

__version__: str
__all__: list[str]
