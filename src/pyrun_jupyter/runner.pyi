"""Type stubs for JupyterRunner - remote Jupyter execution.

This module provides type hints for the JupyterRunner class which enables:
- Remote Python code execution on Jupyter servers
- File upload/download (via Contents API or kernel execution)
- Kernel lifecycle management

Key Methods:
    Code Execution:
        - run(code) -> ExecutionResult: Execute Python code string
        - run_file(path, params) -> ExecutionResult: Execute .py file with optional params
    
    File Transfer (Contents API):
        - upload_file(local, remote) -> dict: Upload single file
        - download_file(remote, local) -> Path: Download single file
        - upload_directory(local_dir, remote_dir) -> list[str]: Upload directory
        - download_files(remote_paths, local_dir) -> list[Path]: Download multiple files
        - list_files(path) -> list[dict]: List remote directory contents
        - file_exists(path) -> bool: Check if remote file exists
        - delete_file(path) -> None: Delete remote file
    
    File Transfer (Kernel-based, for Kaggle etc.):
        - upload_via_kernel(local, remote) -> bool: Upload via kernel execution
        - upload_directory_via_kernel(local_dir, remote_dir) -> list[str]: Upload dir via kernel
        - download_kernel_files(paths, local_dir, working_dir) -> list[Path]: Download kernel outputs
    
    Kernel Management:
        - start_kernel(name) -> str: Start new kernel, returns kernel_id
        - stop_kernel() -> None: Stop current kernel
        - restart_kernel() -> None: Restart current kernel
        - connect_to_kernel(kernel_id) -> None: Connect to existing kernel
        - list_kernels() -> list[dict]: List all running kernels

Example:
    with JupyterRunner("http://localhost:8888", token="xxx") as runner:
        # Upload project
        runner.upload_directory_via_kernel("./src", "project")
        
        # Run training with parameters
        result = runner.run_file("train.py", params={"epochs": 100, "lr": 0.001})
        print(result.stdout)
        
        # Download results
        runner.download_kernel_files(["model.pth"], "./output", "project")
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .result import ExecutionResult


class JupyterRunner:
    """Remote Jupyter code executor with file transfer capabilities."""
    
    # Connection settings
    url: str
    token: Optional[str]
    kernel_name: str
    auto_start_kernel: bool
    reuse_kernel: bool
    
    # Runtime state
    @property
    def kernel_id(self) -> Optional[str]: ...
    @property
    def is_connected(self) -> bool: ...
    
    def __init__(
        self,
        url: str,
        token: Optional[str] = None,
        kernel_name: str = "python3",
        auto_start_kernel: bool = True,
        reuse_kernel: bool = True,
    ) -> None: ...
    
    # ==================== Code Execution ====================
    
    def run(self, code: str, timeout: float = 60.0) -> ExecutionResult:
        """Execute Python code on the remote kernel.
        
        Args:
            code: Python code string to execute
            timeout: Maximum execution time in seconds (default: 60)
        
        Returns:
            ExecutionResult with stdout, stderr, errors, and rich outputs
        
        Example:
            result = runner.run("x = 42; print(f'Answer: {x}')")
            print(result.stdout)  # Answer: 42
        """
        ...
    
    def run_file(
        self,
        filepath: Union[str, Path],
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 60.0,
    ) -> ExecutionResult:
        """Execute a Python .py file on the remote kernel.
        
        Args:
            filepath: Path to the .py file to execute
            params: Optional dict of parameters injected as variables before execution
            timeout: Maximum execution time in seconds
        
        Returns:
            ExecutionResult with stdout, stderr, errors, and rich outputs
        
        Example:
            result = runner.run_file("train.py", params={"lr": 0.01, "epochs": 100})
        """
        ...
    
    # ==================== Kernel Management ====================
    
    def start_kernel(self, name: Optional[str] = None) -> str:
        """Start a new kernel.
        
        Args:
            name: Kernel name (uses default if not specified)
        
        Returns:
            Kernel ID string
        """
        ...
    
    def stop_kernel(self) -> None:
        """Stop the current kernel."""
        ...
    
    def restart_kernel(self) -> None:
        """Restart the current kernel (clears all state)."""
        ...
    
    def list_kernels(self) -> List[Dict[str, Any]]:
        """List all running kernels on the server.
        
        Returns:
            List of kernel info dicts with 'id', 'name', 'state' keys
        """
        ...
    
    def connect_to_kernel(self, kernel_id: str) -> None:
        """Connect to an existing kernel by ID.
        
        Args:
            kernel_id: ID of the kernel to connect to
        """
        ...
    
    # ==================== File Transfer (Contents API) ====================
    
    def upload_file(
        self,
        local_path: Union[str, Path],
        remote_path: str,
        overwrite: bool = True,
    ) -> Dict[str, Any]:
        """Upload a local file to the Jupyter server via Contents API.
        
        Args:
            local_path: Path to local file
            remote_path: Destination path on server (e.g., "data/input.csv")
            overwrite: Whether to overwrite existing file (default: True)
        
        Returns:
            Server response dict with file metadata
        
        Note:
            May not work on all platforms (e.g., Kaggle). Use upload_via_kernel() instead.
        """
        ...
    
    def upload_directory(
        self,
        local_dir: Union[str, Path],
        remote_dir: str = "",
        pattern: str = "**/*",
        exclude_patterns: Optional[List[str]] = None,
    ) -> List[str]:
        """Upload a directory to the server via Contents API.
        
        Args:
            local_dir: Local directory path
            remote_dir: Destination directory on server
            pattern: Glob pattern for files to include (default: all)
            exclude_patterns: Patterns to exclude (default: __pycache__, *.pyc, .git, etc.)
        
        Returns:
            List of uploaded remote file paths
        """
        ...
    
    def download_file(
        self,
        remote_path: str,
        local_path: Union[str, Path],
    ) -> Path:
        """Download a file from the server via Contents API.
        
        Args:
            remote_path: Path on server (e.g., "output/model.pt")
            local_path: Local destination path
        
        Returns:
            Path to downloaded file
        """
        ...
    
    def download_files(
        self,
        remote_paths: List[str],
        local_dir: Union[str, Path],
        flatten: bool = False,
    ) -> List[Path]:
        """Download multiple files from the server.
        
        Args:
            remote_paths: List of paths on server
            local_dir: Local directory to save files
            flatten: If True, save all files directly in local_dir without subdirs
        
        Returns:
            List of paths to downloaded files
        """
        ...
    
    def list_files(self, path: str = "") -> List[Dict[str, Any]]:
        """List files and directories on the server.
        
        Args:
            path: Directory path on server (empty for root)
        
        Returns:
            List of dicts with 'name', 'path', 'type', 'size' keys
        """
        ...
    
    def file_exists(self, remote_path: str) -> bool:
        """Check if a file exists on the server.
        
        Args:
            remote_path: Path to check
        
        Returns:
            True if file exists
        """
        ...
    
    def delete_file(self, remote_path: str) -> None:
        """Delete a file on the server.
        
        Args:
            remote_path: Path to file on server
        """
        ...
    
    # ==================== File Transfer (Kernel-based) ====================
    
    def upload_via_kernel(
        self,
        local_path: Union[str, Path],
        remote_path: str,
    ) -> bool:
        """Upload a file via kernel execution (for Kaggle and similar platforms).
        
        This method transfers files by executing Python code in the kernel,
        bypassing the Contents API which may not be available.
        
        Args:
            local_path: Path to local file
            remote_path: Destination path relative to kernel working directory
        
        Returns:
            True if upload succeeded
        
        Example:
            runner.upload_via_kernel("model.py", "project/models/model.py")
        """
        ...
    
    def upload_directory_via_kernel(
        self,
        local_dir: Union[str, Path],
        remote_dir: str = "",
        pattern: str = "**/*",
        exclude_patterns: Optional[List[str]] = None,
    ) -> List[str]:
        """Upload a directory via kernel execution (for Kaggle and similar platforms).
        
        Args:
            local_dir: Local directory path
            remote_dir: Destination directory on server
            pattern: Glob pattern for files to include
            exclude_patterns: Patterns to exclude (default: __pycache__, *.pyc, .git, etc.)
        
        Returns:
            List of uploaded remote file paths
        
        Example:
            uploaded = runner.upload_directory_via_kernel(
                "./my_project",
                "project",
                pattern="**/*.py",
                exclude_patterns=["tests/", "__pycache__"]
            )
            print(f"Uploaded {len(uploaded)} files")
        """
        ...
    
    def download_kernel_files(
        self,
        remote_paths: List[str],
        local_dir: Union[str, Path],
        working_dir: str = "",
    ) -> List[Path]:
        """Download files created by kernel execution (for Kaggle and similar platforms).
        
        This method reads files via kernel execution and transfers them back,
        useful for downloading training outputs, saved models, plots, etc.
        
        Args:
            remote_paths: List of file paths relative to working_dir
            local_dir: Local directory to save files
            working_dir: Working directory on server where files are located
        
        Returns:
            List of paths to downloaded files
        
        Example:
            # After training that saved model.pth and metrics.json
            downloaded = runner.download_kernel_files(
                ["model.pth", "metrics.json", "loss_plot.png"],
                local_dir="./results",
                working_dir="my_project"
            )
        """
        ...
    
    # ==================== Context Manager ====================
    
    def __enter__(self) -> "JupyterRunner": ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...
    def __repr__(self) -> str: ...
