"""Tests for JupyterRunner."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

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

    @patch.object(JupyterRunner, '_validate_connection')
    def test_normalize_entrypoint_rejects_path_outside_project(self, mock_validate):
        """Entrypoint must stay inside the synced project directory."""
        runner = JupyterRunner("http://localhost:8888")

        with tempfile.TemporaryDirectory() as project_dir, tempfile.TemporaryDirectory() as external_dir:
            external_path = Path(external_dir) / "external.py"
            external_path.write_text("print('external')", encoding="utf-8")
            try:
                with pytest.raises(ValueError, match="must be inside project directory"):
                    runner._normalize_entrypoint(project_dir, external_path)
            finally:
                if external_path.exists():
                    external_path.unlink()

    @patch.object(JupyterRunner, '_validate_connection')
    @patch.object(JupyterRunner, 'download_kernel_files')
    @patch.object(JupyterRunner, '_resolve_kernel_artifacts')
    @patch.object(JupyterRunner, '_sync_project_via_kernel')
    @patch.object(JupyterRunner, '_prepare_remote_project_dir')
    def test_run_project_syncs_executes_and_downloads_artifacts(
        self,
        mock_prepare,
        mock_sync,
        mock_resolve_artifacts,
        mock_download_artifacts,
        mock_validate,
    ):
        """run_project should sync the project, run the entrypoint, and fetch artifacts."""
        mock_resolve_artifacts.return_value = ["outputs/model.bin"]
        downloaded_artifact = Path("/tmp/artifacts/outputs/model.bin")
        mock_download_artifacts.return_value = [downloaded_artifact]

        runner = JupyterRunner("http://localhost:8888", auto_start_kernel=False)
        runner._kernel_id = "test-kernel-id"
        runner._websocket = Mock()
        runner._websocket.execute.return_value = ExecutionResult(stdout="training done\n")

        with tempfile.TemporaryDirectory() as project_dir:
            project_root = Path(project_dir)
            (project_root / "train.py").write_text("print('train')", encoding="utf-8")
            (project_root / "pkg").mkdir()
            (project_root / "pkg" / "model.py").write_text("class Model: pass", encoding="utf-8")

            result = runner.run_project(
                project_dir=project_root,
                entrypoint="train.py",
                artifact_paths=["outputs/*.bin"],
                local_artifact_dir="/tmp/artifacts",
                timeout=120.0,
            )

        mock_prepare.assert_called_once_with(runner.DEFAULT_REMOTE_PROJECT_DIR)
        mock_sync.assert_called_once()
        executed_code = runner._websocket.execute.call_args[0][0]
        assert "runpy.run_path" in executed_code
        assert runner.DEFAULT_REMOTE_PROJECT_DIR in executed_code
        mock_resolve_artifacts.assert_called_once_with(
            ["outputs/*.bin"],
            working_dir=runner.DEFAULT_REMOTE_PROJECT_DIR,
        )
        mock_download_artifacts.assert_called_once_with(
            ["outputs/model.bin"],
            local_dir="/tmp/artifacts",
            working_dir=runner.DEFAULT_REMOTE_PROJECT_DIR,
            flatten=False,
        )
        assert result.data["artifacts"] == [str(downloaded_artifact)]

    @patch.object(JupyterRunner, '_validate_connection')
    @patch.object(JupyterRunner, 'download_kernel_files')
    @patch.object(JupyterRunner, '_resolve_kernel_artifacts')
    @patch.object(JupyterRunner, '_sync_project_via_kernel')
    @patch.object(JupyterRunner, '_prepare_remote_project_dir')
    def test_run_project_attempts_artifact_download_after_failure(
        self,
        mock_prepare,
        mock_sync,
        mock_resolve_artifacts,
        mock_download_artifacts,
        mock_validate,
    ):
        """Artifact collection should still run after execution errors."""
        mock_resolve_artifacts.return_value = ["outputs/partial.txt"]
        mock_download_artifacts.return_value = [Path("/tmp/artifacts/outputs/partial.txt")]

        runner = JupyterRunner("http://localhost:8888", auto_start_kernel=False)
        runner._kernel_id = "test-kernel-id"
        runner._websocket = Mock()
        runner._websocket.execute.return_value = ExecutionResult(
            stdout="partial logs\n",
            success=False,
            error="boom",
            error_name="RuntimeError",
        )

        with tempfile.TemporaryDirectory() as project_dir:
            project_root = Path(project_dir)
            (project_root / "train.py").write_text("raise RuntimeError('boom')", encoding="utf-8")

            result = runner.run_project(
                project_dir=project_root,
                entrypoint="train.py",
                artifact_paths=["outputs/*.txt"],
            )

        assert result.has_error
        mock_download_artifacts.assert_called_once()

    @patch.object(JupyterRunner, '_validate_connection')
    def test_resolve_kernel_artifacts_supports_globs(self, mock_validate):
        """Artifact resolution should parse JSON metadata returned by the kernel."""
        runner = JupyterRunner("http://localhost:8888", auto_start_kernel=False)
        runner._kernel_id = "test-kernel-id"
        runner._websocket = Mock()
        runner._websocket.execute.return_value = ExecutionResult(
            stdout="__PYRUN_ARTIFACTS__\n" + json.dumps(["outputs/model.bin", "logs/train.txt"])
        )

        resolved = runner._resolve_kernel_artifacts(
            ["outputs/*.bin", "logs/*.txt"],
            working_dir="remote/project",
        )

        assert resolved == ["outputs/model.bin", "logs/train.txt"]
        executed_code = runner._websocket.execute.call_args[0][0]
        assert "glob.glob" in executed_code
        assert "remote/project" in executed_code


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
