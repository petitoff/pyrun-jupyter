"""Type stubs for ExecutionResult - Jupyter code execution results.

ExecutionResult is returned by JupyterRunner.run() and JupyterRunner.run_file().
It contains all output and error information from code execution.

Key Attributes:
    stdout: str - Standard output text
    stderr: str - Standard error text  
    success: bool - True if no errors occurred
    has_error: bool - True if errors occurred (opposite of success)
    error: str | None - Error message if failed
    error_name: str | None - Exception type name (e.g., "ValueError")
    error_traceback: list[str] - Full traceback lines
    data: dict - Rich outputs (text/plain, text/html, image/png, etc.)
    text: str - Shortcut for data["text/plain"]
    html: str | None - Shortcut for data["text/html"]

Example:
    result = runner.run("print('hello'); x = 1/0")
    
    if result.has_error:
        print(f"Failed: {result.error_name}: {result.error}")
        # Failed: ZeroDivisionError: division by zero
    else:
        print(result.stdout)
    
    # Access rich output
    if result.html:
        display_html(result.html)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionResult:
    """Result of code execution on remote Jupyter kernel."""
    
    # Output streams
    stdout: str
    """Standard output from print() and similar."""
    
    stderr: str
    """Standard error output."""
    
    # Execution status
    success: bool
    """True if execution completed without errors."""
    
    execution_count: Optional[int]
    """Jupyter cell execution counter (e.g., In[5] -> 5)."""
    
    # Error information (only if success=False)
    error: Optional[str]
    """Error message if execution failed."""
    
    error_name: Optional[str]
    """Exception class name (e.g., 'ValueError', 'TypeError')."""
    
    error_traceback: List[str]
    """Full traceback as list of strings."""
    
    # Rich output data
    data: Dict[str, Any]
    """Rich output dict with mime types as keys.
    
    Common keys:
        - 'text/plain': Plain text representation
        - 'text/html': HTML output (tables, styled text)
        - 'image/png': Base64-encoded PNG image
        - 'application/json': JSON data
    """
    
    display_data: List[Dict[str, Any]]
    """Additional display outputs from display() calls."""
    
    # Convenience properties
    @property
    def text(self) -> str:
        """Get plain text result from data['text/plain']."""
        ...
    
    @property
    def html(self) -> Optional[str]:
        """Get HTML result from data['text/html'] if available."""
        ...
    
    @property
    def has_error(self) -> bool:
        """Check if execution resulted in an error (opposite of success)."""
        ...
    
    def __str__(self) -> str: ...
