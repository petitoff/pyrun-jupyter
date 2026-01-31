"""Contents API module for file operations."""

import base64
import mimetypes
from pathlib import Path
from typing import Dict, Any, Optional, List

import requests

from .exceptions import ConnectionError, PyrunJupyterError


class FileTransferError(PyrunJupyterError):
    """Raised when file upload/download fails."""
    pass


class ContentsManager:
    """Manages file operations on Jupyter server via Contents API.
    
    Provides methods to:
    - Upload files to the server
    - Download files from the server
    - List directory contents
    - Delete files
    """
    
    def __init__(self, base_url: str, token: str = None, headers: Dict[str, str] = None):
        """Initialize contents manager.
        
        Args:
            base_url: Jupyter server URL (e.g., http://localhost:8888)
            token: Authentication token
            headers: Additional headers to include in requests
        """
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.headers = headers or {}
        
        if token:
            self.headers["Authorization"] = f"token {token}"
    
    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request to Jupyter API."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.request(
                method,
                url,
                headers=self.headers,
                timeout=60,  # Longer timeout for file transfers
                **kwargs
            )
            response.raise_for_status()
            return response
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Failed to connect to Jupyter server: {e}")
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                raise FileTransferError(f"Path not found on server")
            raise FileTransferError(f"HTTP error: {e}")
        except requests.exceptions.Timeout:
            raise ConnectionError("Connection timed out during file transfer")
    
    def upload_file(
        self,
        local_path: str,
        remote_path: str,
        overwrite: bool = True
    ) -> Dict[str, Any]:
        """Upload a local file to the Jupyter server.
        
        Args:
            local_path: Path to local file to upload
            remote_path: Destination path on the server (e.g., "data/file.csv")
            overwrite: Whether to overwrite if file exists (default: True)
            
        Returns:
            Server response with file metadata
            
        Raises:
            FileNotFoundError: If local file doesn't exist
            FileTransferError: If upload fails
        """
        local_path = Path(local_path)
        
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        # Read file content and encode as base64
        content = local_path.read_bytes()
        content_b64 = base64.b64encode(content).decode("ascii")
        
        # Guess mimetype
        mimetype, _ = mimetypes.guess_type(str(local_path))
        if mimetype is None:
            mimetype = "application/octet-stream"
        
        # Prepare request body
        body = {
            "type": "file",
            "format": "base64",
            "content": content_b64,
            "name": local_path.name,
        }
        
        # Clean remote path (remove leading slash if present)
        remote_path = remote_path.lstrip("/")
        
        # Upload file using PUT
        response = self._request(
            "PUT",
            f"/api/contents/{remote_path}",
            json=body
        )
        
        return response.json()
    
    def download_file(
        self,
        remote_path: str,
        local_path: str
    ) -> Path:
        """Download a file from the Jupyter server.
        
        Args:
            remote_path: Path on the server (e.g., "output/model.pt")
            local_path: Local destination path
            
        Returns:
            Path to downloaded file
            
        Raises:
            FileTransferError: If download fails
        """
        # Clean remote path
        remote_path = remote_path.lstrip("/")
        
        # Get file content
        response = self._request(
            "GET",
            f"/api/contents/{remote_path}?content=1"
        )
        
        data = response.json()
        
        if data.get("type") == "directory":
            raise FileTransferError(f"Cannot download directory: {remote_path}")
        
        # Decode content
        content_format = data.get("format", "text")
        content = data.get("content", "")
        
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        if content_format == "base64":
            # Binary file
            file_bytes = base64.b64decode(content)
            local_path.write_bytes(file_bytes)
        else:
            # Text file
            local_path.write_text(content, encoding="utf-8")
        
        return local_path
    
    def list_contents(self, path: str = "") -> List[Dict[str, Any]]:
        """List contents of a directory on the server.
        
        Args:
            path: Directory path on server (empty for root)
            
        Returns:
            List of file/directory info dictionaries
        """
        path = path.lstrip("/")
        response = self._request("GET", f"/api/contents/{path}")
        data = response.json()
        
        if data.get("type") == "directory":
            return data.get("content", [])
        return [data]
    
    def delete_file(self, remote_path: str) -> None:
        """Delete a file on the server.
        
        Args:
            remote_path: Path to file on server
        """
        remote_path = remote_path.lstrip("/")
        self._request("DELETE", f"/api/contents/{remote_path}")
    
    def file_exists(self, remote_path: str) -> bool:
        """Check if a file exists on the server.
        
        Args:
            remote_path: Path to check
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            remote_path = remote_path.lstrip("/")
            self._request("GET", f"/api/contents/{remote_path}?content=0")
            return True
        except FileTransferError:
            return False
