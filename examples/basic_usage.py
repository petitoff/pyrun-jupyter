"""Example: Basic usage of py2jupyter."""

from py2jupyter import JupyterRunner

# Configuration - update these values for your Jupyter server
JUPYTER_URL = "http://localhost:8888"
JUPYTER_TOKEN = "your_token_here"  # Get from Jupyter startup output


def main():
    """Demonstrate basic py2jupyter usage."""
    
    # Create a runner and connect to server
    runner = JupyterRunner(JUPYTER_URL, token=JUPYTER_TOKEN)
    
    try:
        # Start a kernel
        kernel_id = runner.start_kernel()
        print(f"Started kernel: {kernel_id}")
        
        # Run inline code
        print("\n--- Running inline code ---")
        result = runner.run("""
import sys
print(f"Python version: {sys.version}")
print(f"Hello from Jupyter!")
x = 42
print(f"The answer is {x}")
""")
        print("Output:")
        print(result.stdout)
        
        # Variables persist across runs in the same kernel
        print("\n--- Variables persist ---")
        result = runner.run("print(f'x is still {x}')")
        print("Output:")
        print(result.stdout)
        
        # Handle errors gracefully
        print("\n--- Error handling ---")
        result = runner.run("undefined_variable")
        if result.has_error:
            print(f"Caught error: {result.error_name}: {result.error}")
        
    finally:
        # Always clean up
        runner.stop_kernel()
        print("\nKernel stopped.")


if __name__ == "__main__":
    main()
