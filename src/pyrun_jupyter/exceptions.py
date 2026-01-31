"""Custom exceptions for pyrun_jupyter."""


class PyrunJupyterError(Exception):
    """Base exception for pyrun_jupyter."""
    pass


class ConnectionError(PyrunJupyterError):
    """Raised when connection to Jupyter server fails."""
    pass


class KernelError(PyrunJupyterError):
    """Raised when kernel operations fail."""
    pass


class ExecutionError(PyrunJupyterError):
    """Raised when code execution fails."""
    
    def __init__(self, message: str, ename: str = None, evalue: str = None, traceback: list = None):
        super().__init__(message)
        self.ename = ename
        self.evalue = evalue
        self.traceback = traceback or []
