"""WebSocket handler for Jupyter kernel communication."""

import json
import uuid
import threading
from typing import Optional, Callable, Dict, Any, List

import websocket

from .result import ExecutionResult
from .exceptions import ExecutionError


class KernelWebSocket:
    """WebSocket connection to a Jupyter kernel for code execution.
    
    Handles the Jupyter messaging protocol over WebSocket for:
    - Sending execute_request messages
    - Receiving stream (stdout/stderr), execute_result, and error messages
    - Collecting all outputs into ExecutionResult
    """
    
    def __init__(self, ws_url: str, headers: Dict[str, str] = None):
        """Initialize WebSocket connection.
        
        Args:
            ws_url: WebSocket URL for the kernel (ws://server/api/kernels/{id}/channels)
            headers: Optional headers including auth token
        """
        self.ws_url = ws_url
        self.headers = headers or {}
        self.ws: Optional[websocket.WebSocket] = None
        self._msg_id: Optional[str] = None
        
    def connect(self) -> None:
        """Establish WebSocket connection."""
        self.ws = websocket.create_connection(
            self.ws_url,
            header=[f"{k}: {v}" for k, v in self.headers.items()],
            timeout=30
        )
    
    def close(self) -> None:
        """Close WebSocket connection."""
        if self.ws:
            self.ws.close()
            self.ws = None
    
    def execute(self, code: str, timeout: float = 60.0) -> ExecutionResult:
        """Execute code and collect results.
        
        Args:
            code: Python code to execute
            timeout: Maximum time to wait for execution (seconds)
            
        Returns:
            ExecutionResult with stdout, stderr, and any errors
        """
        if not self.ws:
            self.connect()
        
        # Create execute_request message
        self._msg_id = str(uuid.uuid4())
        msg = self._create_execute_request(code)
        
        # Send message
        self.ws.send(json.dumps(msg))
        
        # Collect responses
        result = ExecutionResult()
        stdout_parts: List[str] = []
        stderr_parts: List[str] = []
        
        self.ws.settimeout(timeout)
        
        while True:
            try:
                response = self.ws.recv()
                reply = json.loads(response)
                
                # Only process messages for our request
                parent_msg_id = reply.get("parent_header", {}).get("msg_id")
                if parent_msg_id != self._msg_id:
                    continue
                
                msg_type = reply.get("msg_type", "")
                content = reply.get("content", {})
                
                if msg_type == "stream":
                    # stdout or stderr output
                    name = content.get("name", "stdout")
                    text = content.get("text", "")
                    if name == "stdout":
                        stdout_parts.append(text)
                    else:
                        stderr_parts.append(text)
                
                elif msg_type == "execute_result":
                    # Rich output (e.g., display of variables)
                    result.data = content.get("data", {})
                    result.execution_count = content.get("execution_count")
                
                elif msg_type == "display_data":
                    # Additional display outputs
                    result.display_data.append(content.get("data", {}))
                
                elif msg_type == "error":
                    # Execution error
                    result.success = False
                    result.error_name = content.get("ename", "Error")
                    result.error = content.get("evalue", "")
                    result.error_traceback = content.get("traceback", [])
                
                elif msg_type == "execute_reply":
                    # Execution complete
                    status = content.get("status")
                    if status == "error":
                        result.success = False
                        if not result.error:
                            result.error_name = content.get("ename", "Error")
                            result.error = content.get("evalue", "")
                    result.execution_count = content.get("execution_count")
                    break
                
                elif msg_type == "status":
                    # Kernel status updates (busy/idle)
                    execution_state = content.get("execution_state")
                    if execution_state == "idle" and result.execution_count is not None:
                        break
                        
            except websocket.WebSocketTimeoutException:
                result.success = False
                result.error = "Execution timed out"
                result.error_name = "TimeoutError"
                break
        
        result.stdout = "".join(stdout_parts)
        result.stderr = "".join(stderr_parts)
        
        return result
    
    def _create_execute_request(self, code: str) -> Dict[str, Any]:
        """Create a Jupyter execute_request message.
        
        Args:
            code: Python code to execute
            
        Returns:
            Message dictionary following Jupyter protocol
        """
        return {
            "header": {
                "msg_id": self._msg_id,
                "msg_type": "execute_request",
                "username": "py2jupyter",
                "session": str(uuid.uuid4()),
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            "channel": "shell",
        }
