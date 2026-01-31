"""Tests for file transfer methods (upload/download via kernel)."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import base64
import os

from pyrun_jupyter.runner import JupyterRunner
from pyrun_jupyter.result import ExecutionResult


class TestUploadViaKernel:
    """Test upload_via_kernel method."""
    
    @patch.object(JupyterRunner, '_validate_connection')
    def test_upload_via_kernel_success(self, mock_validate):
        """Test successful file upload via kernel."""
        runner = JupyterRunner("http://localhost:8888", auto_start_kernel=False)
        runner._kernel_id = "test-kernel"
        runner._websocket = Mock()
        runner._websocket.execute.return_value = ExecutionResult(stdout="__UPLOAD_OK__")
        
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
            f.write("print('hello')")
            temp_path = f.name
        
        try:
            result = runner.upload_via_kernel(temp_path, "remote/test.py")
            assert result is True
            
            # Verify kernel execution was called
            assert runner._websocket.execute.called
            call_code = runner._websocket.execute.call_args[0][0]
            assert "base64" in call_code
            assert "remote/test.py" in call_code
        finally:
            os.unlink(temp_path)
    
    @patch.object(JupyterRunner, '_validate_connection')
    def test_upload_via_kernel_file_not_found(self, mock_validate):
        """Test upload_via_kernel with non-existent file."""
        runner = JupyterRunner("http://localhost:8888", auto_start_kernel=False)
        
        with pytest.raises(FileNotFoundError):
            runner.upload_via_kernel("nonexistent.py", "remote.py")


class TestUploadDirectoryViaKernel:
    """Test upload_directory_via_kernel method."""
    
    @patch.object(JupyterRunner, '_validate_connection')
    @patch.object(JupyterRunner, 'upload_via_kernel')
    def test_upload_directory_via_kernel(self, mock_upload, mock_validate):
        """Test uploading directory via kernel."""
        mock_upload.return_value = True
        runner = JupyterRunner("http://localhost:8888", auto_start_kernel=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test structure
            (Path(tmpdir) / "main.py").write_text("print('main')")
            (Path(tmpdir) / "utils").mkdir()
            (Path(tmpdir) / "utils" / "helper.py").write_text("def help(): pass")
            (Path(tmpdir) / "__pycache__").mkdir()
            (Path(tmpdir) / "__pycache__" / "main.cpython-39.pyc").write_bytes(b"cache")
            
            uploaded = runner.upload_directory_via_kernel(
                tmpdir,
                remote_dir="project",
                pattern="**/*.py"
            )
            
            # Should upload .py files but not __pycache__
            assert len(uploaded) == 2
            assert "project/main.py" in uploaded
            assert "project/utils/helper.py" in uploaded
    
    @patch.object(JupyterRunner, '_validate_connection')
    def test_upload_directory_not_a_directory(self, mock_validate):
        """Test error when path is not a directory."""
        runner = JupyterRunner("http://localhost:8888", auto_start_kernel=False)
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Not a directory"):
                runner.upload_directory_via_kernel(temp_path, "remote")
        finally:
            os.unlink(temp_path)
    
    @patch.object(JupyterRunner, '_validate_connection')
    @patch.object(JupyterRunner, 'upload_via_kernel')
    def test_upload_directory_with_exclude_patterns(self, mock_upload, mock_validate):
        """Test exclude patterns work correctly."""
        mock_upload.return_value = True
        runner = JupyterRunner("http://localhost:8888", auto_start_kernel=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "main.py").write_text("main")
            (Path(tmpdir) / "test_main.py").write_text("test")
            (Path(tmpdir) / "config.py").write_text("config")
            
            uploaded = runner.upload_directory_via_kernel(
                tmpdir,
                remote_dir="project",
                exclude_patterns=["test_*", "__pycache__"]
            )
            
            # Should not upload test files
            assert "project/main.py" in uploaded
            assert "project/config.py" in uploaded
            assert not any("test_" in f for f in uploaded)


class TestDownloadKernelFiles:
    """Test download_kernel_files method."""
    
    @patch.object(JupyterRunner, '_validate_connection')
    def test_download_kernel_files_success(self, mock_validate):
        """Test successful file download via kernel."""
        runner = JupyterRunner("http://localhost:8888", auto_start_kernel=False)
        runner._kernel_id = "test-kernel"
        runner._websocket = Mock()
        
        # Simulate file content returned by kernel
        test_content = b"model weights data"
        encoded = base64.b64encode(test_content).decode('ascii')
        runner._websocket.execute.return_value = ExecutionResult(
            stdout=f"{encoded}\n__FILE_OK__"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            downloaded = runner.download_kernel_files(
                ["model.pth"],
                local_dir=tmpdir,
                working_dir="project"
            )
            
            assert len(downloaded) == 1
            assert downloaded[0].name == "model.pth"
            assert downloaded[0].read_bytes() == test_content
    
    @patch.object(JupyterRunner, '_validate_connection')
    def test_download_kernel_files_not_found(self, mock_validate):
        """Test handling of missing files."""
        runner = JupyterRunner("http://localhost:8888", auto_start_kernel=False)
        runner._kernel_id = "test-kernel"
        runner._websocket = Mock()
        runner._websocket.execute.return_value = ExecutionResult(
            stdout="__FILE_NOT_FOUND__"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            downloaded = runner.download_kernel_files(
                ["missing.pth"],
                local_dir=tmpdir
            )
            
            assert len(downloaded) == 0
    
    @patch.object(JupyterRunner, '_validate_connection')
    def test_download_kernel_files_multiple(self, mock_validate):
        """Test downloading multiple files."""
        runner = JupyterRunner("http://localhost:8888", auto_start_kernel=False)
        runner._kernel_id = "test-kernel"
        runner._websocket = Mock()
        
        # Different content for each file
        contents = {
            "model.pth": b"model data",
            "config.json": b'{"lr": 0.01}',
        }
        
        def mock_execute(code, **kwargs):
            for filename, content in contents.items():
                if filename in code:
                    encoded = base64.b64encode(content).decode('ascii')
                    return ExecutionResult(stdout=f"{encoded}\n__FILE_OK__")
            return ExecutionResult(stdout="__FILE_NOT_FOUND__")
        
        runner._websocket.execute.side_effect = mock_execute
        
        with tempfile.TemporaryDirectory() as tmpdir:
            downloaded = runner.download_kernel_files(
                list(contents.keys()),
                local_dir=tmpdir,
                working_dir="project"
            )
            
            assert len(downloaded) == 2
            for path in downloaded:
                assert path.read_bytes() == contents[path.name]


class TestUploadDirectory:
    """Test upload_directory method (Contents API)."""
    
    @patch.object(JupyterRunner, '_validate_connection')
    @patch.object(JupyterRunner, 'upload_file')
    def test_upload_directory(self, mock_upload, mock_validate):
        """Test uploading directory via Contents API."""
        mock_upload.return_value = {"path": "test"}
        runner = JupyterRunner("http://localhost:8888", auto_start_kernel=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "script.py").write_text("print(1)")
            (Path(tmpdir) / "data.csv").write_text("a,b,c")
            
            uploaded = runner.upload_directory(
                tmpdir,
                remote_dir="project"
            )
            
            assert len(uploaded) == 2


class TestDownloadFiles:
    """Test download_files method."""
    
    @patch.object(JupyterRunner, '_validate_connection')
    @patch.object(JupyterRunner, 'download_file')
    def test_download_files(self, mock_download, mock_validate):
        """Test downloading multiple files."""
        runner = JupyterRunner("http://localhost:8888", auto_start_kernel=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_download.side_effect = lambda remote, local: Path(local)
            
            downloaded = runner.download_files(
                ["file1.txt", "file2.txt"],
                local_dir=tmpdir
            )
            
            assert mock_download.call_count == 2
    
    @patch.object(JupyterRunner, '_validate_connection')
    @patch.object(JupyterRunner, 'download_file')
    def test_download_files_flatten(self, mock_download, mock_validate):
        """Test flatten option removes subdirectories."""
        runner = JupyterRunner("http://localhost:8888", auto_start_kernel=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            def capture_local_path(remote, local):
                return Path(local)
            
            mock_download.side_effect = capture_local_path
            
            runner.download_files(
                ["subdir/file1.txt", "another/file2.txt"],
                local_dir=tmpdir,
                flatten=True
            )
            
            # With flatten, files should be saved directly in tmpdir
            calls = mock_download.call_args_list
            for call in calls:
                local_path = call[0][1]
                # Should not have subdirectory structure
                assert Path(local_path).parent == Path(tmpdir)
