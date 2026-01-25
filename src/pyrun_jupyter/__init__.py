"""
pyrun-jupyter - Execute Python .py files on remote Jupyter servers.

Usage:
    from pyrun_jupyter import JupyterRunner

    runner = JupyterRunner("http://jupyter-server:8888", token="your_token")
    result = runner.run_file("script.py")
    print(result.stdout)
    
    runner.upload_file("local_data.csv", "data/input.csv")
    runner.download_file("output/model.pt", "local/model.pt")
"""

from .runner import JupyterRunner
from .result import ExecutionResult
from .contents import FileTransferError
from .exceptions import (
    Py2JupyterError,
    ConnectionError,
    KernelError,
    ExecutionError,
)

__version__ = "0.2.0"
__all__ = [
    "JupyterRunner",
    "ExecutionResult",
    "Py2JupyterError",
    "ConnectionError",
    "KernelError",
    "ExecutionError",
    "FileTransferError",
]
