# MC Framework Testing Guide

This document describes the comprehensive test suite for the Metacognitive Core (MC) Framework.

## Overview

The test suite validates the core logic of the MC Framework **without requiring GPU hardware or large model downloads**. This makes it perfect for CI/CD pipelines and rapid development iteration.

## Test Files

### `test_framework.py`

Comprehensive unit tests covering:

- **SAE Feature Extraction** (`_encode_feats`, `_decode_feats`)
  - Different output formats (attribute, dict, fallback)
  - Encode/decode roundtrip consistency
  - Shape preservation

- **Steering Mechanism** (`MCSteerer` class)
  - Delta calculation (proportional control)
  - Max norm clamping (safety mechanism)
  - Statistics tracking
  - Last-token-only modification
  - Tuple/tensor output handling

- **Hook Registration** (`layer_hook` context manager)
  - Proper registration and cleanup
  - Exception-safe cleanup
  - Correct layer indexing

- **Mathematical Functions**
  - Cosine similarity edge cases
  - Normalization correctness

- **Integration Tests**
  - Full steering pipeline
  - Multi-step generation simulation

- **Edge Cases**
  - Zero/negative strength
  - Large sequences
  - Batch size assumptions

### `demo_notebook.ipynb`

Interactive demonstration with:
- Toy SAE implementation
- Visual steering analysis
- Comparative strength experiments
- Validation tests
- **Requires only**: `torch`, `numpy`, `matplotlib`, `scikit-learn`

## Running Tests

### Quick Test (No Installation)

```bash
# Just run the test file directly
python test_framework.py
```

### With pytest (Recommended)

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run tests with coverage
pytest test_framework.py -v --cov=. --cov-report=term

# Run tests with detailed output
pytest test_framework.py -vv
```

### Test a Specific Component

```bash
# Test only SAE feature extraction
pytest test_framework.py::TestSAEFeatureExtraction -v

# Test only steering mechanism
pytest test_framework.py::TestSteeringMechanism -v

# Test only hook registration
pytest test_framework.py::TestHookRegistration -v
```

## CI/CD Integration

### GitHub Actions

The repository includes a comprehensive GitHub Actions workflow (`.github/workflows/test.yml`) that:

1. **Tests on Multiple Platforms**
   - Ubuntu, macOS, Windows
   - Python 3.8, 3.9, 3.10, 3.11

2. **Runs Multiple Checks**
   - Unit tests with coverage
   - Code linting (flake8)
   - Import verification
   - Style checking (black, isort)

3. **CPU-Only Testing**
   - No GPU required
   - Fast execution (<5 minutes typical)
   - Minimal resource usage

### Triggering CI

CI runs automatically on:
- Push to `main`, `master`, `develop`, or `claude/**` branches
- Pull requests to main branches
- Manual workflow dispatch

## Test Coverage

Current coverage includes:

- ✅ SAE encoding with multiple output formats
- ✅ SAE decoding with fallback methods
- ✅ Steering delta calculation
- ✅ Max norm safety clamping
- ✅ Hook registration/removal lifecycle
- ✅ Statistics tracking
- ✅ Cosine similarity edge cases
- ✅ Integration pipeline
- ✅ Zero/negative steering
- ✅ Long sequence handling

## Minimal Dependencies

The test suite only requires:

```
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
torch>=2.0.0
numpy>=1.24.0
```

**NOT required for testing:**
- `transformers` (mocked)
- `sae-lens` (mocked)
- `accelerate` (not used in tests)
- GPU/CUDA (CPU-only tests)

This keeps CI/CD fast and reduces infrastructure costs.

## Writing New Tests

### Test Structure

```python
import unittest
from unittest.mock import Mock
from Tadden_Moore_PEM_PV_Demo import <component>

class TestNewComponent(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        pass

    def test_specific_behavior(self):
        """Test a specific behavior"""
        # Arrange
        mock_obj = Mock()

        # Act
        result = function_under_test(mock_obj)

        # Assert
        self.assertEqual(result, expected_value)
```

### Mocking Guidelines

1. **Mock External Dependencies**: Always mock `transformers`, `sae-lens` models
2. **Use Realistic Shapes**: Use actual tensor dimensions from Gemma-2B (hidden_dim=256+)
3. **Test Edge Cases**: Include zero, negative, very large values
4. **Verify Safety**: Test that safety mechanisms (max_norm) work correctly

### Example: Testing a New Feature

```python
def test_new_steering_feature(self):
    """Test new proportional-integral control"""
    # Mock SAE
    mock_sae = Mock()
    mock_sae.return_value = {"feature_acts": torch.randn(1, 1, 512)}

    # Create steerer with new feature
    steerer = MCSteerer(
        mock_sae,
        concept_feats=torch.randn(1, 1, 512),
        strength=3.0,
        integral_gain=0.1  # New parameter
    )

    # Test that integral term accumulates
    test_output = torch.randn(1, 5, 256)
    steerer.hook(None, None, test_output)
    steerer.hook(None, None, test_output)

    # Verify integral term > 0
    self.assertGreater(steerer.integral_term, 0.0)
```

## Performance Benchmarks

Typical test execution times on Ubuntu with Python 3.10:

- **Unit tests only**: ~2 seconds
- **Full test suite**: ~5 seconds
- **With coverage report**: ~7 seconds

## Debugging Failed Tests

### Common Issues

1. **Shape Mismatches**
   ```python
   # Check tensor shapes
   print(f"Expected: {expected.shape}, Got: {actual.shape}")
   ```

2. **Mock Not Called**
   ```python
   # Verify mock was used
   mock_obj.method.assert_called_once()
   ```

3. **Numerical Precision**
   ```python
   # Use appropriate tolerance
   self.assertAlmostEqual(a, b, places=5)  # Not assertEqual for floats
   ```

### Verbose Output

```bash
# Run single test with maximum verbosity
pytest test_framework.py::TestClass::test_method -vv -s

# Show print statements
pytest test_framework.py -s
```

## Contributing Tests

When contributing new features:

1. ✅ Add unit tests for new functions/classes
2. ✅ Update integration tests if behavior changes
3. ✅ Ensure all tests pass on CPU
4. ✅ Maintain >80% code coverage
5. ✅ Document test purpose in docstrings

## Additional Resources

- **Main Implementation**: `Tadden_Moore_PEM_PV_Demo.py`
- **Demo Notebook**: `demo_notebook.ipynb`
- **Paper**: `Tadden_Moore-Photon_Empress_Moore-v3.6.9.pdf`
- **DOI**: [10.5281/zenodo.17623226](https://doi.org/10.5281/zenodo.17623226)

## Questions?

For questions about testing or the MC Framework:
- Check the paper for theoretical background
- Review `demo_notebook.ipynb` for practical examples
- Examine existing tests for patterns
- Open an issue on GitHub

---

**Author:** Tadden Moore
**Date:** 2025-11-16
**License:** MIT
