"""
pyrun-jupyter - Execute Python code on remote Jupyter servers.

This library provides JupyterRunner for remote code execution with file transfer support.

Main Class:
    JupyterRunner - Connect to Jupyter server and execute code/files

    Key Methods:
        Code Execution:
            - run(code: str) -> ExecutionResult
            - run_file(path, params) -> ExecutionResult
            - run_project(project_dir, entrypoint, artifact_paths) -> ExecutionResult

        File Transfer (Contents API):
            - upload_file(local, remote)
            - upload_directory(local_dir, remote_dir)
            - download_file(remote, local)
            - download_files(paths, local_dir)
            - list_files(path) -> list[dict]
            - file_exists(path) -> bool
            - delete_file(path)

        File Transfer (Kernel-based, for Kaggle etc.):
            - upload_via_kernel(local, remote) -> bool
            - upload_directory_via_kernel(local_dir, remote_dir) -> list[str]
            - download_kernel_files(paths, local_dir, working_dir) -> list[Path]

        Kernel Management:
            - start_kernel(name) -> str
            - stop_kernel()
            - restart_kernel()
            - connect_to_kernel(kernel_id)
            - list_kernels() -> list[dict]

Result Class:
    ExecutionResult - Contains execution output and errors
        - stdout: str - Standard output
        - stderr: str - Standard error
        - success: bool - True if no errors
        - has_error: bool - True if errors occurred
        - error: str | None - Error message
        - error_name: str | None - Exception type
        - error_traceback: list[str] - Full traceback
        - data: dict - Rich outputs (text/html, image/png, etc.)

Basic Usage:
    from pyrun_jupyter import JupyterRunner

    with JupyterRunner("http://localhost:8888", token="xxx") as runner:
        result = runner.run("print('Hello!')")
        print(result.stdout)

        File Transfer (for Kaggle and similar):
    with JupyterRunner(kaggle_url) as runner:
        # Upload project
        runner.upload_directory_via_kernel("./src", "project")

        # Run code
        runner.run("import os; os.chdir('project'); exec(open('train.py').read())")

        # Download results
        runner.download_kernel_files(["model.pth"], "./output", "project")

Project Workflow:
    with JupyterRunner(kaggle_url) as runner:
        result = runner.run_project(
            "./my_project",
            "train.py",
            artifact_paths=["outputs/*.pth", "metrics.json"],
        )
        print(result.data["artifacts"])

Exceptions:
    - PyrunJupyterError: Base exception
    - ConnectionError: Server connection failed
    - KernelError: Kernel operation failed
    - ExecutionError: Code execution failed
    - FileTransferError: File upload/download failed
    - RemoteFileNotFoundError: Requested file is missing on the remote kernel
"""

from .contents import FileTransferError, RemoteFileNotFoundError
from .exceptions import (
    ConnectionError,
    ExecutionError,
    KernelError,
    PyrunJupyterError,
)
from .result import ExecutionResult
from .runner import JupyterRunner

__version__ = "0.6.0"
__all__ = [
    "ConnectionError",
    "ExecutionError",
    "ExecutionResult",
    "FileTransferError",
    "JupyterRunner",
    "KernelError",
    "PyrunJupyterError",
    "RemoteFileNotFoundError",
]
