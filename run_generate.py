"""Wrapper to run dataset generation, suppressing warnings."""
import warnings
warnings.filterwarnings("ignore")

import sys
import os

# Suppress all stderr warnings
os.environ["PYTHONWARNINGS"] = "ignore"

from training.generate_dataset import generate_dataset

if __name__ == "__main__":
    try:
        count = generate_dataset()
        print(f"\nTotal pairs generated: {count}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
