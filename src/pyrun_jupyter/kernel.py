"""Kernel management module."""

import requests
from typing import Dict, List, Optional, Any

from .exceptions import ConnectionError, KernelError


class KernelManager:
    """Manages Jupyter kernel lifecycle via REST API.
    
    Provides methods to:
    - Start new kernels
    - Stop kernels
    - List available kernels
    - Get kernel specifications
    """
    
    def __init__(self, base_url: str, token: str = None, headers: Dict[str, str] = None):
        """Initialize kernel manager.
        
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
        """Make HTTP request to Jupyter API.
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint path
            **kwargs: Additional arguments for requests
            
        Returns:
            Response object
            
        Raises:
            ConnectionError: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.request(
                method,
                url,
                headers=self.headers,
                timeout=30,
                **kwargs
            )
            response.raise_for_status()
            return response
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Failed to connect to Jupyter server at {self.base_url}: {e}")
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                raise ConnectionError("Authentication failed. Check your token.")
            elif response.status_code == 403:
                raise ConnectionError("Access forbidden. Check your permissions.")
            raise ConnectionError(f"HTTP error: {e}")
        except requests.exceptions.Timeout:
            raise ConnectionError("Connection timed out")
    
    def list_kernels(self) -> List[Dict[str, Any]]:
        """List all running kernels.
        
        Returns:
            List of kernel info dictionaries with id, name, execution_state, etc.
        """
        response = self._request("GET", "/api/kernels")
        return response.json()
    
    def start_kernel(self, name: str = "python3") -> Dict[str, Any]:
        """Start a new kernel.
        
        Args:
            name: Kernel spec name (default: python3)
            
        Returns:
            Kernel info dictionary with id and other details
        """
        response = self._request("POST", "/api/kernels", json={"name": name})
        return response.json()
    
    def stop_kernel(self, kernel_id: str) -> None:
        """Stop a running kernel.
        
        Args:
            kernel_id: ID of kernel to stop
            
        Raises:
            KernelError: If kernel cannot be stopped
        """
        try:
            self._request("DELETE", f"/api/kernels/{kernel_id}")
        except ConnectionError as e:
            raise KernelError(f"Failed to stop kernel {kernel_id}: {e}")
    
    def restart_kernel(self, kernel_id: str) -> None:
        """Restart a kernel.
        
        Args:
            kernel_id: ID of kernel to restart
        """
        self._request("POST", f"/api/kernels/{kernel_id}/restart")
    
    def interrupt_kernel(self, kernel_id: str) -> None:
        """Interrupt a kernel's execution.
        
        Args:
            kernel_id: ID of kernel to interrupt
        """
        self._request("POST", f"/api/kernels/{kernel_id}/interrupt")
    
    def get_kernel_info(self, kernel_id: str) -> Dict[str, Any]:
        """Get information about a specific kernel.
        
        Args:
            kernel_id: ID of the kernel
            
        Returns:
            Kernel info dictionary
        """
        response = self._request("GET", f"/api/kernels/{kernel_id}")
        return response.json()
    
    def list_kernelspecs(self) -> Dict[str, Any]:
        """List available kernel specifications.
        
        Returns:
            Dictionary of available kernelspecs
        """
        response = self._request("GET", "/api/kernelspecs")
        return response.json()
    
    def get_websocket_url(self, kernel_id: str) -> str:
        """Get WebSocket URL for a kernel.
        
        Args:
            kernel_id: ID of the kernel
            
        Returns:
            WebSocket URL for kernel channels
        """
        # Convert http(s) to ws(s)
        ws_base = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_base}/api/kernels/{kernel_id}/channels"
        
        # Add token as query parameter for WebSocket auth
        if self.token:
            ws_url += f"?token={self.token}"
        
        return ws_url
