"""Custom exceptions for py2jupyter."""


class Py2JupyterError(Exception):
    """Base exception for py2jupyter."""
    pass


class ConnectionError(Py2JupyterError):
    """Raised when connection to Jupyter server fails."""
    pass


class KernelError(Py2JupyterError):
    """Raised when kernel operations fail."""
    pass


class ExecutionError(Py2JupyterError):
    """Raised when code execution fails."""
    
    def __init__(self, message: str, ename: str = None, evalue: str = None, traceback: list = None):
        super().__init__(message)
        self.ename = ename
        self.evalue = evalue
        self.traceback = traceback or []
