"""Custom exceptions for pyrun_jupyter."""

from typing import List, Optional


class PyrunJupyterError(Exception):
    """Base exception for pyrun_jupyter."""


class ConnectionError(PyrunJupyterError):
    """Raised when connection to Jupyter server fails."""


class KernelError(PyrunJupyterError):
    """Raised when kernel operations fail."""


class ExecutionError(PyrunJupyterError):
    """Raised when code execution fails."""

    def __init__(
        self,
        message: str,
        ename: Optional[str] = None,
        evalue: Optional[str] = None,
        traceback: Optional[List[str]] = None,
    ) -> None:
        super().__init__(message)
        self.ename = ename
        self.evalue = evalue
        self.traceback = traceback or []
