"""Example: Running .py files with parameters."""

from pathlib import Path
from py2jupyter import JupyterRunner

JUPYTER_URL = "http://localhost:8888"
JUPYTER_TOKEN = "your_token_here"


def main():
    """Demonstrate running .py files with parameter injection."""
    
    # Using context manager (recommended)
    with JupyterRunner(JUPYTER_URL, token=JUPYTER_TOKEN) as runner:
        
        # Create a sample script to run
        script_path = Path("temp_training_script.py")
        script_path.write_text("""
# This script uses parameters injected by py2jupyter
print(f"Training Configuration:")
print(f"  Learning Rate: {learning_rate}")
print(f"  Epochs: {epochs}")
print(f"  Batch Size: {batch_size}")
print(f"  Model: {model_name}")

# Simulate training
for epoch in range(min(epochs, 3)):
    loss = 1.0 / (epoch + 1) * learning_rate * 1000
    print(f"Epoch {epoch+1}: loss = {loss:.4f}")

print("\\nTraining complete!")
""")
        
        try:
            # Run the script with parameters
            result = runner.run_file(
                script_path,
                params={
                    "learning_rate": 0.001,
                    "epochs": 10,
                    "batch_size": 32,
                    "model_name": "ResNet50"
                }
            )
            
            print("Script output:")
            print("-" * 40)
            print(result.stdout)
            print("-" * 40)
            print(f"\nExecution successful: {result.success}")
            
        finally:
            # Clean up the temp script
            script_path.unlink()


if __name__ == "__main__":
    main()
