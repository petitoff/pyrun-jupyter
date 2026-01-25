"""Execution result dataclass."""

from dataclasses import dataclass, field
from typing import Optional, Any, List, Dict


@dataclass
class ExecutionResult:
    """Result of code execution on remote Jupyter kernel.
    
    Attributes:
        stdout: Standard output from execution
        stderr: Standard error from execution
        success: Whether execution completed without errors
        execution_count: Jupyter execution counter
        error: Error message if execution failed
        error_name: Exception class name (e.g., 'ValueError')
        error_traceback: Full traceback as list of strings
        data: Rich output data (text/plain, text/html, image/png, etc.)
        display_data: Additional display outputs
    """
    stdout: str = ""
    stderr: str = ""
    success: bool = True
    execution_count: Optional[int] = None
    error: Optional[str] = None
    error_name: Optional[str] = None
    error_traceback: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    display_data: List[Dict[str, Any]] = field(default_factory=list)
    
    def __str__(self) -> str:
        if self.success:
            return self.stdout or "(no output)"
        return f"Error: {self.error_name}: {self.error}"
    
    @property
    def text(self) -> str:
        """Get plain text result if available."""
        return self.data.get("text/plain", "")
    
    @property
    def html(self) -> Optional[str]:
        """Get HTML result if available."""
        return self.data.get("text/html")
    
    @property
    def has_error(self) -> bool:
        """Check if execution resulted in an error."""
        return not self.success
