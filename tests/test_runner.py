"""Tests for JupyterRunner."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

from pyrun_jupyter.runner import JupyterRunner
from pyrun_jupyter.result import ExecutionResult


class TestJupyterRunner:
    """Test JupyterRunner class."""
    
    @patch.object(JupyterRunner, '_validate_connection')
    def test_initialization(self, mock_validate):
        """Test runner initialization."""
        runner = JupyterRunner(
            "http://localhost:8888",
            token="test_token",
            kernel_name="python3"
        )
        
        assert runner.url == "http://localhost:8888"
        assert runner.token == "test_token"
        assert runner.kernel_name == "python3"
        assert not runner.is_connected
    
    @patch.object(JupyterRunner, '_validate_connection')
    def test_url_trailing_slash_removed(self, mock_validate):
        """Test that trailing slash is removed from URL."""
        runner = JupyterRunner("http://localhost:8888/", token="xxx")
        assert runner.url == "http://localhost:8888"
    
    @patch.object(JupyterRunner, '_validate_connection')
    def test_generate_params_code(self, mock_validate):
        """Test parameter code generation."""
        runner = JupyterRunner("http://localhost:8888")
        
        params = {
            "learning_rate": 0.001,
            "epochs": 100,
            "model_name": "resnet50",
            "use_cuda": True
        }
        
        code = runner._generate_params_code(params)
        
        assert "learning_rate = 0.001" in code
        assert "epochs = 100" in code
        assert "model_name = 'resnet50'" in code
        assert "use_cuda = True" in code
    
    @patch.object(JupyterRunner, '_validate_connection')
    def test_run_file_not_found(self, mock_validate):
        """Test FileNotFoundError for missing file."""
        runner = JupyterRunner("http://localhost:8888")
        
        with pytest.raises(FileNotFoundError):
            runner.run_file("nonexistent_file.py")
    
    @patch.object(JupyterRunner, '_validate_connection')
    def test_run_file_wrong_extension(self, mock_validate):
        """Test ValueError for non-.py file."""
        runner = JupyterRunner("http://localhost:8888")
        
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"print('hello')")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match=r"Expected .py file"):
                runner.run_file(temp_path)
        finally:
            Path(temp_path).unlink()
    
    @patch.object(JupyterRunner, '_validate_connection')
    def test_run_file_with_params(self, mock_validate):
        """Test running file with parameters."""
        runner = JupyterRunner("http://localhost:8888", auto_start_kernel=False)
        runner._kernel_id = "test-kernel-id"  # Simulate connected state
        runner._websocket = Mock()
        runner._websocket.execute.return_value = ExecutionResult(stdout="OK")
        
        with tempfile.NamedTemporaryFile(
            suffix=".py", 
            delete=False, 
            mode='w',
            encoding='utf-8'
        ) as f:
            f.write("print(f'LR: {lr}')")
            temp_path = f.name
        
        try:
            result = runner.run_file(temp_path, params={"lr": 0.01})
            
            # Check that execute was called with injected params
            call_args = runner._websocket.execute.call_args[0][0]
            assert "lr = 0.01" in call_args
            assert "print(f'LR: {lr}')" in call_args
        finally:
            Path(temp_path).unlink()
    
    @patch.object(JupyterRunner, '_validate_connection')
    def test_repr(self, mock_validate):
        """Test string representation."""
        runner = JupyterRunner("http://localhost:8888")
        repr_str = repr(runner)
        
        assert "JupyterRunner" in repr_str
        assert "localhost:8888" in repr_str
        assert "disconnected" in repr_str


class TestJupyterRunnerContextManager:
    """Test context manager functionality."""
    
    @patch.object(JupyterRunner, '_validate_connection')
    @patch.object(JupyterRunner, 'start_kernel')
    @patch.object(JupyterRunner, 'stop_kernel')
    def test_context_manager(self, mock_stop, mock_start, mock_validate):
        """Test context manager starts and stops kernel."""
        with JupyterRunner("http://localhost:8888") as runner:
            mock_start.assert_called_once()
        
        mock_stop.assert_called_once()
