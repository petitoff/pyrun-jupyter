"""Main JupyterRunner class for executing Python code on remote Jupyter servers."""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

from .kernel import KernelManager
from .contents import ContentsManager, FileTransferError
from .websocket import KernelWebSocket
from .result import ExecutionResult
from .exceptions import PyrunJupyterError, KernelError, ExecutionError


class JupyterRunner:
    """Execute Python .py files on a remote Jupyter server.
    
    This class provides a simple interface to run Python code or files
    on a remote Jupyter kernel, collecting output and handling errors.
    
    Example:
        >>> runner = JupyterRunner("http://localhost:8888", token="your_token")
        >>> result = runner.run_file("script.py")
        >>> print(result.stdout)
        
        >>> result = runner.run("print('Hello from Jupyter!')")
        >>> print(result.stdout)
        Hello from Jupyter!
    """
    
    def __init__(
        self,
        url: str,
        token: str = None,
        kernel_name: str = "python3",
        auto_start_kernel: bool = True,
        reuse_kernel: bool = True,
    ):
        """Initialize JupyterRunner.
        
        Args:
            url: Jupyter server URL (e.g., http://localhost:8888)
            token: Authentication token for the server
            kernel_name: Name of kernel to use (default: python3)
            auto_start_kernel: Whether to start a kernel automatically on first run
            reuse_kernel: Whether to reuse existing kernel on reconnection (default: True)
        """
        self.url = url.rstrip("/")
        self.token = token
        self.kernel_name = kernel_name
        self.auto_start_kernel = auto_start_kernel
        self.reuse_kernel = reuse_kernel
        
        self._kernel_manager = KernelManager(self.url, token)
        self._contents_manager = ContentsManager(self.url, token)
        self._kernel_id: Optional[str] = None
        self._websocket: Optional[KernelWebSocket] = None
        
        # Validate connection on initialization
        self._validate_connection()
    
    @property
    def kernel_id(self) -> Optional[str]:
        """Get current kernel ID."""
        return self._kernel_id
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to a kernel."""
        return self._kernel_id is not None and self._websocket is not None
    
    def start_kernel(self, name: str = None) -> str:
        """Start a new kernel.
        
        Args:
            name: Kernel name (uses default if not specified)
            
        Returns:
            Kernel ID
        """
        kernel_name = name or self.kernel_name
        kernel_info = self._kernel_manager.start_kernel(kernel_name)
        self._kernel_id = kernel_info["id"]
        
        # Create WebSocket connection
        ws_url = self._kernel_manager.get_websocket_url(self._kernel_id)
        self._websocket = KernelWebSocket(ws_url)
        self._websocket.connect()
        
        return self._kernel_id
    
    def stop_kernel(self) -> None:
        """Stop the current kernel."""
        if self._websocket:
            self._websocket.close()
            self._websocket = None
        
        if self._kernel_id:
            self._kernel_manager.stop_kernel(self._kernel_id)
            self._kernel_id = None
    
    def restart_kernel(self) -> None:
        """Restart the current kernel."""
        if self._kernel_id:
            self._kernel_manager.restart_kernel(self._kernel_id)
    
    def list_kernels(self) -> list:
        """List all running kernels on the server."""
        return self._kernel_manager.list_kernels()
    
    def connect_to_kernel(self, kernel_id: str) -> None:
        """Connect to an existing kernel.
        
        Args:
            kernel_id: ID of the kernel to connect to
        """
        # Verify kernel exists
        self._kernel_manager.get_kernel_info(kernel_id)
        
        self._kernel_id = kernel_id
        ws_url = self._kernel_manager.get_websocket_url(kernel_id)
        self._websocket = KernelWebSocket(ws_url)
        self._websocket.connect()
    
    def _validate_connection(self) -> None:
        """Validate connection to Jupyter server.
        
        Raises:
            ConnectionError: If cannot connect to server or authentication fails
        """
        try:
            # Try to list kernels to validate connection
            self._kernel_manager.list_kernels()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Jupyter server at {self.url}: {e}")
    
    def _find_reusable_kernel(self) -> Optional[str]:
        """Find an existing kernel that can be reused.
        
        Returns:
            Kernel ID if found, None otherwise
        """
        if not self.reuse_kernel:
            return None
        
        try:
            kernels = self._kernel_manager.list_kernels()
            for kernel in kernels:
                if kernel.get("name") == self.kernel_name:
                    return kernel.get("id")
        except Exception:
            pass
        return None
    
    def _ensure_kernel(self) -> None:
        """Ensure a kernel is running, start one if needed.
        
        If reuse_kernel is True, will try to connect to existing kernel first.
        """
        if self._kernel_id:
            return
        
        if self.auto_start_kernel:
            # Try to reuse existing kernel first
            existing_kernel = self._find_reusable_kernel()
            if existing_kernel:
                self.connect_to_kernel(existing_kernel)
            else:
                self.start_kernel()
        else:
            raise KernelError("No kernel running. Call start_kernel() first.")
    
    def run(self, code: str, timeout: float = 60.0) -> ExecutionResult:
        """Execute Python code on the remote kernel.
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
            
        Returns:
            ExecutionResult with stdout, stderr, and any errors
        """
        self._ensure_kernel()
        return self._websocket.execute(code, timeout=timeout)
    
    def run_file(
        self,
        filepath: Union[str, Path],
        params: Dict[str, Any] = None,
        timeout: float = 60.0,
    ) -> ExecutionResult:
        """Execute a Python file on the remote kernel.
        
        Args:
            filepath: Path to the .py file to execute
            params: Optional parameters to inject as variables before execution
            timeout: Maximum execution time in seconds
            
        Returns:
            ExecutionResult with stdout, stderr, and any errors
            
        Example:
            >>> result = runner.run_file("train.py", params={"lr": 0.01, "epochs": 100})
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if not filepath.suffix == ".py":
            raise ValueError(f"Expected .py file, got: {filepath.suffix}")
        
        # Read file content
        code = filepath.read_text(encoding="utf-8")
        
        # Inject parameters if provided
        if params:
            param_code = self._generate_params_code(params)
            code = param_code + "\n" + code
        
        return self.run(code, timeout=timeout)
    
    def _generate_params_code(self, params: Dict[str, Any]) -> str:
        """Generate Python code to define parameters as variables.
        
        Args:
            params: Dictionary of parameter names and values
            
        Returns:
            Python code string that defines the parameters
        """
        lines = ["# Parameters injected by pyrun_jupyter"]
        for name, value in params.items():
            lines.append(f"{name} = {repr(value)}")
        return "\n".join(lines)
    
    # ==================== File Transfer Methods ====================
    
    def upload_file(
        self,
        local_path: Union[str, Path],
        remote_path: str,
        overwrite: bool = True
    ) -> Dict[str, Any]:
        """Upload a local file to the Jupyter server.
        
        Args:
            local_path: Path to local file to upload
            remote_path: Destination path on the server (e.g., "data/input.csv")
            overwrite: Whether to overwrite if file exists (default: True)
            
        Returns:
            Server response with file metadata
            
        Example:
            >>> runner.upload_file("local_data.csv", "input/data.csv")
            >>> runner.upload_file("model.py", "scripts/model.py")
        """
        return self._contents_manager.upload_file(str(local_path), remote_path, overwrite)
    
    def download_file(
        self,
        remote_path: str,
        local_path: Union[str, Path]
    ) -> Path:
        """Download a file from the Jupyter server.
        
        Args:
            remote_path: Path on the server (e.g., "output/model.pt")
            local_path: Local destination path
            
        Returns:
            Path to downloaded file
            
        Example:
            >>> runner.download_file("output/trained_model.pt", "local/model.pt")
            >>> runner.download_file("results.csv", "./results.csv")
        """
        return self._contents_manager.download_file(remote_path, str(local_path))
    
    def list_files(self, path: str = "") -> List[Dict[str, Any]]:
        """List files and directories on the server.
        
        Args:
            path: Directory path on server (empty for root)
            
        Returns:
            List of file/directory info dictionaries with name, path, type, size
            
        Example:
            >>> files = runner.list_files("output/")
            >>> for f in files:
            ...     print(f"{f['name']} ({f['type']})")
        """
        return self._contents_manager.list_contents(path)
    
    def delete_file(self, remote_path: str) -> None:
        """Delete a file on the server.
        
        Args:
            remote_path: Path to file on server
        """
        self._contents_manager.delete_file(remote_path)
    
    def file_exists(self, remote_path: str) -> bool:
        """Check if a file exists on the server.
        
        Args:
            remote_path: Path to check
            
        Returns:
            True if file exists, False otherwise
        """
        return self._contents_manager.file_exists(remote_path)
    
    # ==================== Context Manager ====================
    
    def __enter__(self) -> "JupyterRunner":
        """Context manager entry."""
        if self.auto_start_kernel and not self._kernel_id:
            self.start_kernel()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stop the kernel."""
        self.stop_kernel()
    
    def __repr__(self) -> str:
        status = "connected" if self.is_connected else "disconnected"
        return f"JupyterRunner(url='{self.url}', kernel_id={self._kernel_id}, status={status})"
