"""
py2jupyter - Execute Python .py files on remote Jupyter servers.

Usage:
    from py2jupyter import JupyterRunner

    runner = JupyterRunner("http://jupyter-server:8888", token="your_token")
    result = runner.run_file("script.py")
    print(result.stdout)
"""

from .runner import JupyterRunner
from .result import ExecutionResult
from .exceptions import (
    Py2JupyterError,
    ConnectionError,
    KernelError,
    ExecutionError,
)

__version__ = "0.1.0"
__all__ = [
    "JupyterRunner",
    "ExecutionResult",
    "Py2JupyterError",
    "ConnectionError",
    "KernelError",
    "ExecutionError",
]
