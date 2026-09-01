"""Tests for file transfer methods (upload/download via kernel)."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pyrun_jupyter.contents import FileTransferError, RemoteFileNotFoundError
from pyrun_jupyter.runner import JupyterRunner


class TestUploadViaKernel:
    """Test upload_via_kernel method."""

    @patch.object(JupyterRunner, "_validate_connection")
    def test_upload_via_kernel_success(self, mock_validate):
        """Test successful file upload via kernel."""
        runner = JupyterRunner("http://localhost:8888", auto_start_kernel=False)
        runner._kernel_id = "test-kernel"
        runner._websocket = Mock()
        runner._websocket.upload_file.return_value = True

        with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
            f.write("print('hello')")
            temp_path = f.name

        try:
            result = runner.upload_via_kernel(temp_path, "remote/test.py")
            assert result is True

            runner._websocket.upload_file.assert_called_once_with(
                Path(temp_path),
                "remote/test.py",
                chunk_size=4 * 1024 * 1024,
                timeout=120.0,
            )
        finally:
            os.unlink(temp_path)

    @patch.object(JupyterRunner, "_validate_connection")
    def test_upload_via_kernel_file_not_found(self, mock_validate):
        """Test upload_via_kernel with non-existent file."""
        runner = JupyterRunner("http://localhost:8888", auto_start_kernel=False)

        with pytest.raises(FileNotFoundError):
            runner.upload_via_kernel("nonexistent.py", "remote.py")


class TestUploadDirectoryViaKernel:
    """Test upload_directory_via_kernel method."""

    @patch.object(JupyterRunner, "_validate_connection")
    @patch.object(JupyterRunner, "upload_via_kernel")
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
                tmpdir, remote_dir="project", pattern="**/*.py"
            )

            # Should upload .py files but not __pycache__
            assert len(uploaded) == 2
            assert "project/main.py" in uploaded
            assert "project/utils/helper.py" in uploaded

    @patch.object(JupyterRunner, "_validate_connection")
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

    @patch.object(JupyterRunner, "_validate_connection")
    @patch.object(JupyterRunner, "upload_via_kernel")
    def test_upload_directory_with_exclude_patterns(self, mock_upload, mock_validate):
        """Test exclude patterns work correctly."""
        mock_upload.return_value = True
        runner = JupyterRunner("http://localhost:8888", auto_start_kernel=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "main.py").write_text("main")
            (Path(tmpdir) / "test_main.py").write_text("test")
            (Path(tmpdir) / "config.py").write_text("config")

            uploaded = runner.upload_directory_via_kernel(
                tmpdir, remote_dir="project", exclude_patterns=["test_*", "__pycache__"]
            )

            # Should not upload test files
            assert "project/main.py" in uploaded
            assert "project/config.py" in uploaded
            assert not any("test_" in f for f in uploaded)


class TestDownloadKernelFiles:
    """Test download_kernel_files method."""

    @staticmethod
    def _runner_with_remote_files(remote_files):
        """Build a runner whose kernel serves ``remote_files`` over binary transfer."""
        runner = JupyterRunner("http://localhost:8888", auto_start_kernel=False)
        runner._kernel_id = "test-kernel"
        runner._websocket = Mock()

        def fake_download(remote_path, local_path, **kwargs):
            if remote_path not in remote_files:
                raise RemoteFileNotFoundError(f"Remote file not found: {remote_path}")
            content = remote_files[remote_path]
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(content)
            return len(content)

        runner._websocket.download_file.side_effect = fake_download
        return runner

    @patch.object(JupyterRunner, "_validate_connection")
    def test_download_kernel_files_success(self, mock_validate):
        """Test successful file download via kernel."""
        test_content = b"model weights data"
        runner = self._runner_with_remote_files({"project/model.pth": test_content})

        with tempfile.TemporaryDirectory() as tmpdir:
            downloaded = runner.download_kernel_files(
                ["model.pth"], local_dir=tmpdir, working_dir="project"
            )

            assert len(downloaded) == 1
            assert downloaded[0].name == "model.pth"
            assert downloaded[0].read_bytes() == test_content

    @patch.object(JupyterRunner, "_validate_connection")
    def test_download_kernel_files_not_found(self, mock_validate):
        """Missing remote files are skipped rather than raised."""
        runner = self._runner_with_remote_files({})

        with tempfile.TemporaryDirectory() as tmpdir:
            downloaded = runner.download_kernel_files(["missing.pth"], local_dir=tmpdir)

            assert len(downloaded) == 0

    @patch.object(JupyterRunner, "_validate_connection")
    def test_download_kernel_files_skips_failed_transfer(self, mock_validate):
        """A verification failure on one file must not abort the remaining files."""
        runner = self._runner_with_remote_files({"good.bin": b"intact"})
        working = runner._websocket.download_file.side_effect

        def flaky_download(remote_path, local_path, **kwargs):
            if remote_path == "corrupt.bin":
                raise FileTransferError("Downloaded file failed verification: corrupt.bin")
            return working(remote_path, local_path, **kwargs)

        runner._websocket.download_file.side_effect = flaky_download

        with tempfile.TemporaryDirectory() as tmpdir:
            downloaded = runner.download_kernel_files(["corrupt.bin", "good.bin"], local_dir=tmpdir)

            assert [path.name for path in downloaded] == ["good.bin"]

    @patch.object(JupyterRunner, "_validate_connection")
    def test_download_kernel_files_multiple(self, mock_validate):
        """Test downloading multiple files."""
        contents = {
            "model.pth": b"model data",
            "config.json": b'{"lr": 0.01}',
        }
        runner = self._runner_with_remote_files(
            {f"project/{name}": content for name, content in contents.items()}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            downloaded = runner.download_kernel_files(
                list(contents.keys()), local_dir=tmpdir, working_dir="project"
            )

            assert len(downloaded) == 2
            for path in downloaded:
                assert path.read_bytes() == contents[path.name]

    @patch.object(JupyterRunner, "_validate_connection")
    def test_download_kernel_files_forwards_transfer_options(self, mock_validate):
        """Chunk size and timeout must reach the binary transport."""
        runner = self._runner_with_remote_files({"model.pth": b"weights"})

        with tempfile.TemporaryDirectory() as tmpdir:
            runner.download_kernel_files(
                ["model.pth"], local_dir=tmpdir, chunk_size=1024, timeout=5.0
            )

        runner._websocket.download_file.assert_called_once()
        _args, kwargs = runner._websocket.download_file.call_args
        assert kwargs["chunk_size"] == 1024
        assert kwargs["timeout"] == 5.0

    @patch.object(JupyterRunner, "_validate_connection")
    def test_download_kernel_files_preserves_relative_paths_when_flatten_disabled(
        self, mock_validate
    ):
        """Nested artifact paths should be recreated locally when flatten=False."""
        content = b"checkpoint"
        runner = self._runner_with_remote_files({"project/models/best/model.pth": content})

        with tempfile.TemporaryDirectory() as tmpdir:
            downloaded = runner.download_kernel_files(
                ["models/best/model.pth"],
                local_dir=tmpdir,
                working_dir="project",
                flatten=False,
            )

            assert len(downloaded) == 1
            assert downloaded[0] == Path(tmpdir) / "models" / "best" / "model.pth"
            assert downloaded[0].read_bytes() == content

    @patch.object(JupyterRunner, "_validate_connection")
    def test_download_kernel_files_rejects_local_path_traversal(self, mock_validate):
        """A remote artifact name cannot escape the requested local directory."""
        runner = self._runner_with_remote_files({})

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unsafe remote file path"):
                runner.download_kernel_files(
                    ["../outside.bin"],
                    local_dir=tmpdir,
                    flatten=False,
                )

        runner._websocket.download_file.assert_not_called()


class TestUploadDirectory:
    """Test upload_directory method (Contents API)."""

    @patch.object(JupyterRunner, "_validate_connection")
    @patch.object(JupyterRunner, "upload_file")
    def test_upload_directory(self, mock_upload, mock_validate):
        """Test uploading directory via Contents API."""
        mock_upload.return_value = {"path": "test"}
        runner = JupyterRunner("http://localhost:8888", auto_start_kernel=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "script.py").write_text("print(1)")
            (Path(tmpdir) / "data.csv").write_text("a,b,c")

            uploaded = runner.upload_directory(tmpdir, remote_dir="project")

            assert len(uploaded) == 2


class TestDownloadFiles:
    """Test download_files method."""

    @patch.object(JupyterRunner, "_validate_connection")
    @patch.object(JupyterRunner, "download_file")
    def test_download_files(self, mock_download, mock_validate):
        """Test downloading multiple files."""
        runner = JupyterRunner("http://localhost:8888", auto_start_kernel=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_download.side_effect = lambda remote, local: Path(local)

            runner.download_files(["file1.txt", "file2.txt"], local_dir=tmpdir)

            assert mock_download.call_count == 2

    @patch.object(JupyterRunner, "_validate_connection")
    @patch.object(JupyterRunner, "download_file")
    def test_download_files_flatten(self, mock_download, mock_validate):
        """Test flatten option removes subdirectories."""
        runner = JupyterRunner("http://localhost:8888", auto_start_kernel=False)

        with tempfile.TemporaryDirectory() as tmpdir:

            def capture_local_path(remote, local):
                return Path(local)

            mock_download.side_effect = capture_local_path

            runner.download_files(
                ["subdir/file1.txt", "another/file2.txt"], local_dir=tmpdir, flatten=True
            )

            # With flatten, files should be saved directly in tmpdir
            calls = mock_download.call_args_list
            for call in calls:
                local_path = call[0][1]
                # Should not have subdirectory structure
                assert Path(local_path).parent == Path(tmpdir)
