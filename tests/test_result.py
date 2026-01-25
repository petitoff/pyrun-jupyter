"""Tests for ExecutionResult dataclass."""

import pytest
from pyrun_jupyter.result import ExecutionResult


class TestExecutionResult:
    """Test ExecutionResult functionality."""
    
    def test_default_values(self):
        """Test default initialization."""
        result = ExecutionResult()
        
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.success is True
        assert result.error is None
        assert result.data == {}
    
    def test_success_result(self):
        """Test successful execution result."""
        result = ExecutionResult(
            stdout="Hello, World!\n",
            success=True,
            execution_count=1
        )
        
        assert str(result) == "Hello, World!\n"
        assert not result.has_error
    
    def test_error_result(self):
        """Test error execution result."""
        result = ExecutionResult(
            success=False,
            error="division by zero",
            error_name="ZeroDivisionError",
            error_traceback=["Traceback...", "ZeroDivisionError: division by zero"]
        )
        
        assert result.has_error
        assert "ZeroDivisionError" in str(result)
    
    def test_rich_output(self):
        """Test rich output access."""
        result = ExecutionResult(
            data={
                "text/plain": "42",
                "text/html": "<b>42</b>"
            }
        )
        
        assert result.text == "42"
        assert result.html == "<b>42</b>"
    
    def test_no_output(self):
        """Test result with no output."""
        result = ExecutionResult()
        assert str(result) == "(no output)"
