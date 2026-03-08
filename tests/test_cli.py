"""Tests for the pyrun-jupyter CLI."""

from argparse import Namespace
from pathlib import Path
import tempfile
from unittest.mock import patch

from pyrun_jupyter.cli import create_parser, handle_run_project
from pyrun_jupyter.result import ExecutionResult


class TestCLIParser:
    """Test CLI parser configuration."""

    def test_run_project_parser_accepts_project_arguments(self):
        """run-project should expose project sync and artifact flags."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "run-project",
                "examples",
                "basic_usage.py",
                "--url",
                "http://localhost:8888",
                "--artifact",
                "outputs/*.bin",
                "--exclude",
                ".git",
                "--artifact-dir",
                "results",
            ]
        )

        assert args.command == "run-project"
        assert args.project_dir == "examples"
        assert args.entrypoint == "basic_usage.py"
        assert args.artifact == ["outputs/*.bin"]
        assert args.exclude == [".git"]
        assert args.artifact_dir == "results"


class TestHandleRunProject:
    """Test run-project command handling."""

    @patch("pyrun_jupyter.cli.print_result")
    @patch("pyrun_jupyter.cli.JupyterRunner")
    def test_handle_run_project_invokes_runner(self, mock_runner_cls, mock_print_result):
        """run-project should call JupyterRunner.run_project with parsed arguments."""
        mock_runner = mock_runner_cls.return_value.__enter__.return_value
        mock_runner.run_project.return_value = ExecutionResult(
            stdout="ok\n",
            data={"artifacts": ["artifacts/model.bin"]},
        )
        mock_print_result.return_value = 0

        with tempfile.TemporaryDirectory() as project_dir:
            Path(project_dir, "train.py").write_text("print('hi')", encoding="utf-8")

            args = Namespace(
                project_dir=project_dir,
                entrypoint="train.py",
                url="http://localhost:8888",
                token="secret",
                kernel="python3",
                timeout=120.0,
                params='{"epochs": 5}',
                artifact=["outputs/*.bin"],
                artifact_dir="downloaded",
                exclude=[".git", "__pycache__"],
                remote_dir="remote/project",
            )

            exit_code = handle_run_project(args)

        assert exit_code == 0
        mock_runner.run_project.assert_called_once_with(
            project_dir=Path(project_dir),
            entrypoint="train.py",
            artifact_paths=["outputs/*.bin"],
            local_artifact_dir="downloaded",
            remote_dir="remote/project",
            exclude_patterns=[".git", "__pycache__"],
            params={"epochs": 5},
            timeout=120.0,
        )
        mock_print_result.assert_called_once()

    def test_handle_run_project_rejects_missing_project_dir(self):
        """run-project should fail before constructing the runner when the directory is missing."""
        args = Namespace(
            project_dir="/tmp/does-not-exist",
            entrypoint="train.py",
            url="http://localhost:8888",
            token=None,
            kernel="python3",
            timeout=60.0,
            params=None,
            artifact=[],
            artifact_dir="artifacts",
            exclude=[],
            remote_dir=None,
        )

        assert handle_run_project(args) == 1
