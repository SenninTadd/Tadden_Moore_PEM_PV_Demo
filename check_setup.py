#!/usr/bin/env python3
"""
Quick diagnostic script to check if test dependencies are working
"""

import sys

print("Python version:", sys.version)
print("-" * 50)

# Check imports
try:
    import torch
    print("✓ torch:", torch.__version__)
except ImportError as e:
    print("✗ torch: NOT INSTALLED")
    print("  Install with: pip install torch")

try:
    import numpy
    print("✓ numpy:", numpy.__version__)
except ImportError:
    print("✗ numpy: NOT INSTALLED")
    print("  Install with: pip install numpy")

try:
    import pytest
    print("✓ pytest:", pytest.__version__)
except ImportError:
    print("✗ pytest: NOT INSTALLED (optional)")
    print("  Install with: pip install pytest")

print("-" * 50)

# Check if main file exists and can be imported
try:
    sys.path.insert(0, '.')
    from Tadden_Moore_PEM_PV_Demo import MCSteerer, cosine_similarity
    print("✓ Main module imports successfully")
except Exception as e:
    print(f"✗ Main module import failed: {e}")

print("-" * 50)
print("\nTo run tests, you need:")
print("  pip install -r requirements-test.txt")
print("\nOr minimal install:")
print("  pip install torch numpy pytest")
