"""Command-line interface for pyrun-jupyter.

This module provides a CLI for executing Python code and files on remote Jupyter servers.

Usage:
    pyrun-jupyter run-file script.py --url http://localhost:8888 --token xxx
    pyrun-jupyter run "print('hello')" --url http://localhost:8888 --token xxx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from .runner import JupyterRunner
from .exceptions import PyrunJupyterError


def parse_params(params_str: Optional[str]) -> Dict[str, Any]:
    """Parse parameters string into a dictionary.
    
    Args:
        params_str: JSON string or key=value pairs separated by commas.
                   Examples: '{"lr": 0.01, "epochs": 100}' or "lr=0.01,epochs=100"
    
    Returns:
        Dictionary of parameters
    
    Raises:
        ValueError: If the params string cannot be parsed
    """
    if not params_str:
        return {}
    
    params_str = params_str.strip()
    
    # Try JSON first
    if params_str.startswith("{") and params_str.endswith("}"):
        try:
            return json.loads(params_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
    
    # Parse key=value pairs
    params: Dict[str, Any] = {}
    for pair in params_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"Invalid parameter format: '{pair}'. Expected 'key=value' or JSON.")
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        
        # Try to convert value to appropriate type
        params[key] = convert_value(value)
    
    return params


def convert_value(value: str) -> Any:
    """Convert a string value to its appropriate Python type.
    
    Args:
        value: String value to convert
        
    Returns:
        Converted value (int, float, bool, str, or None)
    """
    # Try None
    if value.lower() == "none" or value.lower() == "null":
        return None
    
    # Try bool
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    
    # Try int
    try:
        return int(value)
    except ValueError:
        pass
    
    # Try float
    try:
        return float(value)
    except ValueError:
        pass
    
    # Return as string (strip quotes if present)
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    
    return value


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="pyrun-jupyter",
        description="Execute Python code on remote Jupyter servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pyrun-jupyter run-file script.py --url http://localhost:8888 --token xxx
  pyrun-jupyter run "print('hello')" --url http://localhost:8888 --token xxx
  pyrun-jupyter run-file train.py --url http://localhost:8888 --token xxx \\
      --params '{"lr": 0.01, "epochs": 100}'
  pyrun-jupyter run-file train.py --url http://localhost:8888 --token xxx \\
      --params "lr=0.01,epochs=100"
        """
    )
    
    # Global arguments
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.4.0"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # run-file command
    run_file_parser = subparsers.add_parser(
        "run-file",
        help="Execute a Python file on a remote Jupyter server",
        description="Execute a Python file on a remote Jupyter server."
    )
    run_file_parser.add_argument(
        "filepath",
        type=str,
        help="Path to the Python file to execute"
    )
    run_file_parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Jupyter server URL (e.g., http://localhost:8888)"
    )
    run_file_parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Authentication token for the Jupyter server"
    )
    run_file_parser.add_argument(
        "--kernel",
        type=str,
        default="python3",
        help="Kernel name to use (default: python3)"
    )
    run_file_parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Parameters to inject as variables. Can be JSON (e.g., '{\"key\": \"value\"}') "
             "or key=value pairs separated by commas (e.g., 'key=value,key2=123')"
    )
    run_file_parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Maximum execution time in seconds (default: 60)"
    )
    
    # run command
    run_parser = subparsers.add_parser(
        "run",
        help="Execute Python code on a remote Jupyter server",
        description="Execute Python code on a remote Jupyter server."
    )
    run_parser.add_argument(
        "code",
        type=str,
        help="Python code to execute"
    )
    run_parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Jupyter server URL (e.g., http://localhost:8888)"
    )
    run_parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Authentication token for the Jupyter server"
    )
    run_parser.add_argument(
        "--kernel",
        type=str,
        default="python3",
        help="Kernel name to use (default: python3)"
    )
    run_parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Maximum execution time in seconds (default: 60)"
    )
    
    return parser


def handle_run_file(args: argparse.Namespace) -> int:
    """Handle the run-file command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    filepath = Path(args.filepath)
    
    if not filepath.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        return 1
    
    if not filepath.suffix == ".py":
        print(f"Error: Expected .py file, got: {filepath.suffix}", file=sys.stderr)
        return 1
    
    try:
        params = parse_params(args.params)
    except ValueError as e:
        print(f"Error: Invalid params format: {e}", file=sys.stderr)
        return 1
    
    try:
        with JupyterRunner(
            url=args.url,
            token=args.token,
            kernel_name=args.kernel
        ) as runner:
            result = runner.run_file(
                filepath=filepath,
                params=params if params else None,
                timeout=args.timeout
            )
            
            # Print output
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr)
            
            # Print error info if execution failed
            if result.has_error:
                print(f"\nError: {result.error}", file=sys.stderr)
                if result.error_traceback:
                    for line in result.error_traceback:
                        print(line, file=sys.stderr)
                return 1
            
            return 0
            
    except PyrunJupyterError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def handle_run(args: argparse.Namespace) -> int:
    """Handle the run command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        with JupyterRunner(
            url=args.url,
            token=args.token,
            kernel_name=args.kernel
        ) as runner:
            result = runner.run(
                code=args.code,
                timeout=args.timeout
            )
            
            # Print output
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr)
            
            # Print error info if execution failed
            if result.has_error:
                print(f"\nError: {result.error}", file=sys.stderr)
                if result.error_traceback:
                    for line in result.error_traceback:
                        print(line, file=sys.stderr)
                return 1
            
            return 0
            
    except PyrunJupyterError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def main(args: Optional[list] = None) -> int:
    """Main entry point for the CLI.
    
    Args:
        args: Command-line arguments (defaults to sys.argv[1:])
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    if parsed_args.command is None:
        parser.print_help()
        return 1
    
    if parsed_args.command == "run-file":
        return handle_run_file(parsed_args)
    elif parsed_args.command == "run":
        return handle_run(parsed_args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
