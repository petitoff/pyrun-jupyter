"""Tests for ContentsManager (Jupyter Contents API)."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import base64
import json

from pyrun_jupyter.contents import ContentsManager, FileTransferError


class TestContentsManager:
    """Test ContentsManager class."""
    
    def test_initialization(self):
        """Test ContentsManager initialization."""
        manager = ContentsManager("http://localhost:8888", token="test_token")
        
        assert manager.base_url == "http://localhost:8888"
        assert manager.token == "test_token"
        assert "Authorization" in manager.headers
        assert manager.headers["Authorization"] == "token test_token"
    
    def test_initialization_no_token(self):
        """Test initialization without token."""
        manager = ContentsManager("http://localhost:8888")
        
        assert manager.token is None
        assert "Authorization" not in manager.headers
    
    def test_url_trailing_slash_removed(self):
        """Test that trailing slash is removed from URL."""
        manager = ContentsManager("http://localhost:8888/")
        assert manager.base_url == "http://localhost:8888"


class TestContentsManagerUpload:
    """Test upload functionality."""
    
    @patch('pyrun_jupyter.contents.requests.request')
    def test_upload_file_text(self, mock_request):
        """Test uploading a text file."""
        mock_response = Mock()
        mock_response.json.return_value = {"path": "test.py", "type": "file"}
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response
        
        manager = ContentsManager("http://localhost:8888", token="xxx")
        
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as f:
            f.write("print('hello')")
            temp_path = f.name
        
        try:
            result = manager.upload_file(temp_path, "scripts/test.py")
            
            assert result["path"] == "test.py"
            mock_request.assert_called_once()
            
            # Check correct endpoint - requests.request(method, url, ...)
            call_args = mock_request.call_args
            url = call_args[0][1]  # Second positional arg is url
            assert "/api/contents/scripts/test.py" in url
            
            # Check body has base64 content
            body = call_args[1]["json"]
            assert body["type"] == "file"
            assert body["format"] == "base64"
        finally:
            Path(temp_path).unlink()
    
    @patch('pyrun_jupyter.contents.requests.request')
    def test_upload_file_binary(self, mock_request):
        """Test uploading a binary file."""
        mock_response = Mock()
        mock_response.json.return_value = {"path": "model.pth"}
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response
        
        manager = ContentsManager("http://localhost:8888")
        
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            f.write(b"\x00\x01\x02\x03binary data")
            temp_path = f.name
        
        try:
            result = manager.upload_file(temp_path, "model.pth")
            assert result["path"] == "model.pth"
        finally:
            Path(temp_path).unlink()
    
    def test_upload_file_not_found(self):
        """Test uploading non-existent file."""
        manager = ContentsManager("http://localhost:8888")
        
        with pytest.raises(FileNotFoundError):
            manager.upload_file("nonexistent.py", "remote.py")
    
    @patch('pyrun_jupyter.contents.requests.request')
    def test_upload_file_strips_leading_slash(self, mock_request):
        """Test that leading slash is stripped from remote path."""
        mock_response = Mock()
        mock_response.json.return_value = {"path": "test.py"}
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response
        
        manager = ContentsManager("http://localhost:8888")
        
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as f:
            f.write("x = 1")
            temp_path = f.name
        
        try:
            manager.upload_file(temp_path, "/scripts/test.py")
            
            # requests.request(method, url, ...) - url is second positional arg
            call_url = mock_request.call_args[0][1]
            assert "/api/contents/scripts/test.py" in call_url
            assert "/api/contents//scripts" not in call_url
        finally:
            Path(temp_path).unlink()


class TestContentsManagerDownload:
    """Test download functionality."""
    
    @patch('pyrun_jupyter.contents.requests.request')
    def test_download_file_text(self, mock_request):
        """Test downloading a text file."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "type": "file",
            "format": "text",
            "content": "print('hello world')"
        }
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response
        
        manager = ContentsManager("http://localhost:8888")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "downloaded.py"
            result = manager.download_file("scripts/test.py", str(local_path))
            
            assert result == local_path
            assert local_path.read_text() == "print('hello world')"
    
    @patch('pyrun_jupyter.contents.requests.request')
    def test_download_file_binary(self, mock_request):
        """Test downloading a binary file."""
        binary_content = b"\x00\x01\x02model weights"
        mock_response = Mock()
        mock_response.json.return_value = {
            "type": "file",
            "format": "base64",
            "content": base64.b64encode(binary_content).decode('ascii')
        }
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response
        
        manager = ContentsManager("http://localhost:8888")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "model.pth"
            manager.download_file("model.pth", str(local_path))
            
            assert local_path.read_bytes() == binary_content
    
    @patch('pyrun_jupyter.contents.requests.request')
    def test_download_creates_parent_dirs(self, mock_request):
        """Test that download creates parent directories."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "type": "file",
            "format": "text",
            "content": "data"
        }
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response
        
        manager = ContentsManager("http://localhost:8888")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "nested" / "deep" / "file.txt"
            manager.download_file("file.txt", str(local_path))
            
            assert local_path.exists()
            assert local_path.read_text() == "data"
    
    @patch('pyrun_jupyter.contents.requests.request')
    def test_download_directory_error(self, mock_request):
        """Test error when trying to download a directory."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "type": "directory",
            "content": []
        }
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response
        
        manager = ContentsManager("http://localhost:8888")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileTransferError, match="Cannot download directory"):
                manager.download_file("some_dir", str(Path(tmpdir) / "local"))


class TestContentsManagerList:
    """Test list_contents functionality."""
    
    @patch('pyrun_jupyter.contents.requests.request')
    def test_list_contents_directory(self, mock_request):
        """Test listing directory contents."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "type": "directory",
            "content": [
                {"name": "file1.py", "type": "file", "size": 100},
                {"name": "subdir", "type": "directory"},
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response
        
        manager = ContentsManager("http://localhost:8888")
        result = manager.list_contents("project/")
        
        assert len(result) == 2
        assert result[0]["name"] == "file1.py"
        assert result[1]["name"] == "subdir"
    
    @patch('pyrun_jupyter.contents.requests.request')
    def test_list_contents_single_file(self, mock_request):
        """Test listing a single file returns list with one item."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "type": "file",
            "name": "script.py",
            "size": 50
        }
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response
        
        manager = ContentsManager("http://localhost:8888")
        result = manager.list_contents("script.py")
        
        assert len(result) == 1
        assert result[0]["name"] == "script.py"


class TestContentsManagerFileExists:
    """Test file_exists functionality."""
    
    @patch('pyrun_jupyter.contents.requests.request')
    def test_file_exists_true(self, mock_request):
        """Test file_exists returns True for existing file."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response
        
        manager = ContentsManager("http://localhost:8888")
        assert manager.file_exists("existing.py") is True
    
    @patch('pyrun_jupyter.contents.requests.request')
    def test_file_exists_false(self, mock_request):
        """Test file_exists returns False for missing file."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("404")
        mock_request.return_value = mock_response
        
        manager = ContentsManager("http://localhost:8888")
        
        # file_exists should catch exception and return False
        with patch.object(manager, '_request', side_effect=FileTransferError("not found")):
            assert manager.file_exists("missing.py") is False


class TestContentsManagerDelete:
    """Test delete functionality."""
    
    @patch('pyrun_jupyter.contents.requests.request')
    def test_delete_file(self, mock_request):
        """Test deleting a file."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response
        
        manager = ContentsManager("http://localhost:8888")
        manager.delete_file("old_file.py")
        
        mock_request.assert_called_once()
        assert mock_request.call_args[0][0] == "DELETE"
