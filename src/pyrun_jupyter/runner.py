"""Main JupyterRunner class for executing Python code on remote Jupyter servers.

This module provides the core JupyterRunner class for:
- Executing Python code and .py files on remote Jupyter kernels
- Uploading/downloading files via Contents API or kernel execution
- Managing kernel lifecycle (start, stop, restart, connect)

Typical usage:
    from pyrun_jupyter import JupyterRunner
    
    with JupyterRunner("http://localhost:8888", token="xxx") as runner:
        # Run code
        result = runner.run("print('Hello!')")
        
        # Upload project
        runner.upload_directory_via_kernel("./project", "remote_project")
        
        # Run file and get results
        result = runner.run_file("train.py", params={"epochs": 10})
        
        # Download outputs
        runner.download_kernel_files(["model.pth"], "./results", "remote_project")
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, TYPE_CHECKING

from .kernel import KernelManager
from .contents import ContentsManager, FileTransferError
from .websocket import KernelWebSocket
from .result import ExecutionResult
from .exceptions import PyrunJupyterError, KernelError, ExecutionError


class JupyterRunner:
    """Execute Python code and files on a remote Jupyter server.
    
    This class provides a complete interface for remote Python execution including:
    - Code execution: run(), run_file()
    - File transfer via API: upload_file(), download_file(), upload_directory()
    - File transfer via kernel: upload_via_kernel(), download_kernel_files()
    - Kernel management: start_kernel(), stop_kernel(), restart_kernel()
    
    Attributes:
        url: Jupyter server URL
        token: Authentication token
        kernel_name: Name of kernel specification (e.g., "python3")
        kernel_id: ID of currently connected kernel (None if not connected)
        is_connected: Whether currently connected to a kernel
    
    Example:
        Basic usage::
        
            runner = JupyterRunner("http://localhost:8888", token="your_token")
            result = runner.run("print('Hello!')")
            print(result.stdout)  # Hello!
            runner.stop_kernel()
        
        Context manager (recommended)::
        
            with JupyterRunner("http://localhost:8888", token="xxx") as runner:
                result = runner.run_file("script.py")
                print(result.stdout)
            # Kernel automatically stopped
        
        File transfer for Kaggle/remote environments::
        
            with JupyterRunner(kaggle_url) as runner:
                # Upload entire project
                runner.upload_directory_via_kernel("./my_project", "project")
                
                # Run training
                runner.run("import os; os.chdir('project'); exec(open('train.py').read())")
                
                # Download results
                runner.download_kernel_files(
                    ["model.pth", "metrics.json"],
                    local_dir="./results",
                    working_dir="project"
                )
    """
    
    # Class-level type hints for better IDE support
    url: str
    token: Optional[str]
    kernel_name: str
    auto_start_kernel: bool
    reuse_kernel: bool
    _kernel_manager: KernelManager
    _contents_manager: ContentsManager
    _kernel_id: Optional[str]
    _websocket: Optional[KernelWebSocket]
    
    def __init__(
        self,
        url: str,
        token: Optional[str] = None,
        kernel_name: str = "python3",
        auto_start_kernel: bool = True,
        reuse_kernel: bool = True,
    ) -> None:
        """Initialize JupyterRunner.
        
        Args:
            url: Jupyter server URL (e.g., "http://localhost:8888" or Kaggle proxy URL)
            token: Authentication token for the server (None for token-less auth)
            kernel_name: Name of kernel to use (default: "python3")
            auto_start_kernel: Whether to start a kernel automatically on first run
            reuse_kernel: Whether to reuse existing kernel on reconnection (default: True)
        
        Raises:
            ConnectionError: If cannot connect to the Jupyter server
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
    
    def upload_directory(
        self,
        local_dir: Union[str, Path],
        remote_dir: str = "",
        pattern: str = "**/*",
        exclude_patterns: List[str] = None
    ) -> List[str]:
        """Upload a directory of files to the Jupyter server.
        
        Args:
            local_dir: Local directory path to upload
            remote_dir: Destination directory on server (empty for root)
            pattern: Glob pattern for files to include (default: all files)
            exclude_patterns: List of patterns to exclude (e.g., ["__pycache__", "*.pyc"])
            
        Returns:
            List of uploaded remote file paths
            
        Example:
            >>> runner.upload_directory("./my_project", "project")
            >>> runner.upload_directory("./src", "src", pattern="**/*.py")
            >>> runner.upload_directory("./data", "data", exclude_patterns=["*.tmp", "__pycache__"])
        """
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            raise ValueError(f"Not a directory: {local_dir}")
        
        exclude_patterns = exclude_patterns or ["__pycache__", "*.pyc", ".git", ".venv", "*.egg-info"]
        uploaded = []
        
        for local_path in local_dir.glob(pattern):
            if not local_path.is_file():
                continue
            
            # Check exclusions
            rel_path = local_path.relative_to(local_dir)
            skip = False
            for exc in exclude_patterns:
                if any(part.startswith(exc.replace("*", "")) or 
                       Path(part).match(exc) for part in rel_path.parts):
                    skip = True
                    break
                if rel_path.match(exc):
                    skip = True
                    break
            
            if skip:
                continue
            
            # Build remote path
            remote_path = f"{remote_dir}/{rel_path}".lstrip("/") if remote_dir else str(rel_path)
            remote_path = remote_path.replace("\\", "/")  # Normalize for Unix
            
            try:
                self.upload_file(local_path, remote_path)
                uploaded.append(remote_path)
            except Exception as e:
                print(f"Warning: Failed to upload {local_path}: {e}")
        
        return uploaded
    
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
    
    def download_files(
        self,
        remote_paths: List[str],
        local_dir: Union[str, Path],
        flatten: bool = False
    ) -> List[Path]:
        """Download multiple files from the Jupyter server.
        
        Args:
            remote_paths: List of paths on the server to download
            local_dir: Local directory to save files
            flatten: If True, save all files directly in local_dir without subdirs
            
        Returns:
            List of paths to downloaded files
            
        Example:
            >>> runner.download_files(
            ...     ["output/model.pt", "output/metrics.json"],
            ...     "./results"
            ... )
        """
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded = []
        for remote_path in remote_paths:
            try:
                if flatten:
                    filename = Path(remote_path).name
                    local_path = local_dir / filename
                else:
                    local_path = local_dir / remote_path
                
                result = self.download_file(remote_path, local_path)
                downloaded.append(result)
            except Exception as e:
                print(f"Warning: Failed to download {remote_path}: {e}")
        
        return downloaded
    
    def download_kernel_files(
        self,
        remote_paths: List[str],
        local_dir: Union[str, Path],
        working_dir: str = ""
    ) -> List[Path]:
        """Download files created by kernel execution (e.g., on Kaggle).
        
        Some Jupyter environments (like Kaggle) store kernel outputs in a different
        location than the Contents API. This method uses kernel execution to read
        and transfer files.
        
        Args:
            remote_paths: List of file paths relative to kernel working directory
            local_dir: Local directory to save files
            working_dir: Working directory on server (kernel's cwd)
            
        Returns:
            List of paths to downloaded files
            
        Example:
            >>> # After running training that created model.pth
            >>> runner.download_kernel_files(
            ...     ["model.pth", "predictions.png"],
            ...     "./results",
            ...     working_dir="my_project"
            ... )
        """
        import base64
        
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded = []
        
        for filename in remote_paths:
            if working_dir:
                full_path = f"{working_dir}/{filename}"
            else:
                full_path = filename
            
            # Use kernel to read file and encode as base64
            result = self.run(f'''
import os
import base64

filepath = {repr(full_path)}
if os.path.exists(filepath):
    with open(filepath, 'rb') as f:
        content = base64.b64encode(f.read()).decode('ascii')
    # Print in chunks to avoid output limits
    chunk_size = 50000
    for i in range(0, len(content), chunk_size):
        print(content[i:i+chunk_size], end='')
    print()  # Final newline
    print("__FILE_OK__")
else:
    print("__FILE_NOT_FOUND__")
''', timeout=120.0)
            
            output = result.stdout.strip()
            
            if "__FILE_NOT_FOUND__" in output:
                print(f"Warning: {filename} not found on server")
                continue
            
            if "__FILE_OK__" in output:
                # Extract base64 content (everything before __FILE_OK__)
                b64_content = output.replace("__FILE_OK__", "").strip()
                try:
                    file_bytes = base64.b64decode(b64_content)
                    local_path = local_dir / Path(filename).name
                    local_path.write_bytes(file_bytes)
                    downloaded.append(local_path)
                except Exception as e:
                    print(f"Warning: Failed to decode {filename}: {e}")
            else:
                print(f"Warning: Unexpected output for {filename}")
        
        return downloaded
    
    def upload_via_kernel(
        self,
        local_path: Union[str, Path],
        remote_path: str
    ) -> bool:
        """Upload a file via kernel execution (for environments like Kaggle).
        
        Some Jupyter environments don't allow direct file uploads via Contents API.
        This method transfers files by executing Python code in the kernel.
        
        Args:
            local_path: Path to local file to upload
            remote_path: Destination path on server (relative to kernel working dir)
            
        Returns:
            True if upload succeeded
            
        Example:
            >>> runner.upload_via_kernel("model.py", "project/model.py")
        """
        import base64
        
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        content_bytes = local_path.read_bytes()
        content_b64 = base64.b64encode(content_bytes).decode('ascii')
        
        # Create directory and write file
        remote_dir = str(Path(remote_path).parent)
        
        result = self.run(f'''
import os
import base64

remote_path = {repr(remote_path)}
remote_dir = os.path.dirname(remote_path)
if remote_dir:
    os.makedirs(remote_dir, exist_ok=True)

content = base64.b64decode({repr(content_b64)})
with open(remote_path, 'wb') as f:
    f.write(content)
print("__UPLOAD_OK__")
''')
        
        return "__UPLOAD_OK__" in result.stdout
    
    def upload_directory_via_kernel(
        self,
        local_dir: Union[str, Path],
        remote_dir: str = "",
        pattern: str = "**/*",
        exclude_patterns: List[str] = None
    ) -> List[str]:
        """Upload a directory via kernel execution (for environments like Kaggle).
        
        Args:
            local_dir: Local directory path to upload
            remote_dir: Destination directory on server
            pattern: Glob pattern for files to include (default: all files)
            exclude_patterns: List of patterns to exclude
            
        Returns:
            List of uploaded remote file paths
            
        Example:
            >>> runner.upload_directory_via_kernel("./my_project", "project")
        """
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            raise ValueError(f"Not a directory: {local_dir}")
        
        exclude_patterns = exclude_patterns or ["__pycache__", "*.pyc", ".git", ".venv", "*.egg-info"]
        uploaded = []
        
        for local_path in local_dir.glob(pattern):
            if not local_path.is_file():
                continue
            
            # Check exclusions
            rel_path = local_path.relative_to(local_dir)
            skip = False
            for exc in exclude_patterns:
                if any(part.startswith(exc.replace("*", "")) or 
                       Path(part).match(exc) for part in rel_path.parts):
                    skip = True
                    break
                if rel_path.match(exc):
                    skip = True
                    break
            
            if skip:
                continue
            
            # Build remote path
            remote_path = f"{remote_dir}/{rel_path}".lstrip("/") if remote_dir else str(rel_path)
            remote_path = remote_path.replace("\\", "/")
            
            try:
                if self.upload_via_kernel(local_path, remote_path):
                    uploaded.append(remote_path)
            except Exception as e:
                print(f"Warning: Failed to upload {local_path}: {e}")
        
        return uploaded
    
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
