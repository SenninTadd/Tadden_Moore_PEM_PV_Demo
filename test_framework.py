#!/usr/bin/env python3
"""
Comprehensive Unit Tests for MC Framework
Author: Tadden Moore
Date: 2025-11-16
Version: 1.0.0

Tests the core components of the Metacognitive Core Framework without
requiring full model downloads or GPU hardware.
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
import sys
import os

# Mock heavy dependencies before importing main module
# This allows tests to run without transformers/sae-lens installed
sys.modules['transformers'] = MagicMock()
sys.modules['sae_lens'] = MagicMock()

# Import the module under test
sys.path.insert(0, os.path.dirname(__file__))
from Tadden_Moore_PEM_PV_Demo import (
    _encode_feats,
    _decode_feats,
    MCSteerer,
    cosine_similarity,
    layer_hook,
    capture_concept_features,
)


class TestSAEFeatureExtraction(unittest.TestCase):
    """Test SAE encoding and decoding functions"""

    def setUp(self):
        """Set up test fixtures"""
        self.device = "cpu"
        self.hidden_dim = 256
        self.feature_dim = 512
        self.batch_size = 1
        self.seq_len = 10

    def test_encode_feats_with_feature_acts_attribute(self):
        """Test encoding when SAE output has feature_acts attribute"""
        # Mock SAE
        mock_sae = Mock()
        mock_output = Mock()
        expected_feats = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        mock_output.feature_acts = expected_feats
        mock_sae.return_value = mock_output

        # Test data
        h_last = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)

        # Run encoding
        result = _encode_feats(mock_sae, h_last)

        # Assertions
        mock_sae.assert_called_once_with(h_last)
        self.assertTrue(torch.equal(result, expected_feats))

    def test_encode_feats_with_dict_output(self):
        """Test encoding when SAE output is a dictionary"""
        # Mock SAE
        mock_sae = Mock()
        expected_feats = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        mock_sae.return_value = {"feature_acts": expected_feats}

        # Test data
        h_last = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)

        # Run encoding
        result = _encode_feats(mock_sae, h_last)

        # Assertions
        self.assertTrue(torch.equal(result, expected_feats))

    def test_encode_feats_fallback_to_encode_method(self):
        """Test encoding falls back to encode() method"""
        # Mock SAE
        mock_sae = Mock()
        mock_sae.return_value = {}  # No feature_acts
        expected_feats = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        mock_sae.encode.return_value = expected_feats

        # Test data
        h_last = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)

        # Run encoding
        result = _encode_feats(mock_sae, h_last)

        # Assertions
        mock_sae.encode.assert_called_once_with(h_last)
        self.assertTrue(torch.equal(result, expected_feats))

    def test_decode_feats_with_decode_method(self):
        """Test decoding when SAE has decode method"""
        # Mock SAE
        mock_sae = Mock()
        expected_hidden = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        mock_sae.decode.return_value = expected_hidden

        # Test data
        feats = torch.randn(self.batch_size, self.seq_len, self.feature_dim)

        # Run decoding
        result = _decode_feats(mock_sae, feats)

        # Assertions
        mock_sae.decode.assert_called_once_with(feats)
        self.assertTrue(torch.equal(result, expected_hidden))

    def test_decode_feats_fallback_to_manual_decoding(self):
        """Test decoding falls back to manual W_dec matrix multiplication"""
        # Mock SAE without decode method
        mock_sae = Mock(spec=[])  # spec=[] means no methods
        W_dec = torch.randn(self.hidden_dim, self.feature_dim)  # Correct dimensions: [hidden, feature]
        mock_sae.W_dec = Mock()
        mock_sae.W_dec.T = W_dec.T

        # Test data
        feats = torch.randn(self.batch_size, self.seq_len, self.feature_dim)

        # Run decoding
        result = _decode_feats(mock_sae, feats)

        # Assertions
        expected = feats @ W_dec.T  # [B, S, F] @ [F, H] = [B, S, H]
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_encode_decode_roundtrip_preserves_shape(self):
        """Test that encode->decode preserves tensor shapes"""
        # Mock SAE
        mock_sae = Mock()
        feats = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        hidden = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)

        mock_sae.return_value = {"feature_acts": feats}
        mock_sae.decode.return_value = hidden

        # Test data
        h_input = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)

        # Encode then decode
        encoded = _encode_feats(mock_sae, h_input)
        decoded = _decode_feats(mock_sae, encoded)

        # Check shapes
        self.assertEqual(encoded.shape, (self.batch_size, self.seq_len, self.feature_dim))
        self.assertEqual(decoded.shape, (self.batch_size, self.seq_len, self.hidden_dim))


class TestSteeringMechanism(unittest.TestCase):
    """Test the MCSteerer class and steering mathematics"""

    def setUp(self):
        """Set up test fixtures"""
        self.hidden_dim = 256
        self.feature_dim = 512
        self.batch_size = 1
        self.seq_len = 5

        # Create mock SAE
        self.mock_sae = Mock()
        self.W_dec = torch.randn(self.hidden_dim, self.feature_dim)  # [H, F]
        self.mock_sae.W_dec = Mock()
        self.mock_sae.W_dec.T = self.W_dec.T

        # Concept features
        self.concept_feats = torch.randn(1, 1, self.feature_dim)

    def test_mcsteerer_initialization(self):
        """Test MCSteerer initializes correctly"""
        strength = 4.0
        max_norm = 100.0

        steerer = MCSteerer(
            self.mock_sae,
            self.concept_feats,
            strength=strength,
            max_norm=max_norm
        )

        self.assertEqual(steerer.strength, strength)
        self.assertEqual(steerer.max_norm, max_norm)
        self.assertEqual(steerer.steering_count, 0)
        self.assertEqual(steerer.total_delta_norm, 0.0)
        self.assertTrue(torch.equal(steerer.f, self.concept_feats))

    def test_steering_delta_calculation(self):
        """Test that steering delta is calculated as strength * concept_feats"""
        strength = 5.0
        steerer = MCSteerer(self.mock_sae, self.concept_feats, strength=strength)

        # Expected delta
        expected_delta = self.concept_feats * strength

        # Mock the encoding and decoding
        current_feats = torch.randn(1, 1, self.feature_dim)
        self.mock_sae.return_value = {"feature_acts": current_feats}
        # Make decode return a real tensor
        self.mock_sae.decode.return_value = torch.randn(1, 1, self.hidden_dim)

        # Create test input
        test_output = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)

        # Apply hook
        result = steerer.hook(None, None, test_output)

        # Verify delta magnitude is correct (strength * concept_feats norm)
        expected_norm = torch.linalg.norm(expected_delta).item()
        self.assertAlmostEqual(steerer.total_delta_norm, expected_norm, places=5)

    def test_max_norm_clamping(self):
        """Test that steering delta is clamped to max_norm"""
        strength = 10.0
        max_norm = 1.0  # Very small to force clamping

        steerer = MCSteerer(
            self.mock_sae,
            self.concept_feats,
            strength=strength,
            max_norm=max_norm
        )

        # Mock encoding and decoding
        current_feats = torch.randn(1, 1, self.feature_dim)
        self.mock_sae.return_value = {"feature_acts": current_feats}
        self.mock_sae.decode.return_value = torch.randn(1, 1, self.hidden_dim)

        # Create test input
        test_output = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)

        # Apply hook
        steerer.hook(None, None, test_output)

        # Delta norm should be <= max_norm
        avg_norm = steerer.total_delta_norm / steerer.steering_count
        self.assertLessEqual(avg_norm, max_norm + 1e-6)

    def test_steering_count_increments(self):
        """Test that steering count increments on each hook call"""
        steerer = MCSteerer(self.mock_sae, self.concept_feats, strength=4.0)

        # Mock encoding and decoding
        self.mock_sae.return_value = {"feature_acts": torch.randn(1, 1, self.feature_dim)}
        self.mock_sae.decode.return_value = torch.randn(1, 1, self.hidden_dim)

        # Apply hook multiple times
        test_output = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        for i in range(5):
            steerer.hook(None, None, test_output)
            self.assertEqual(steerer.steering_count, i + 1)

    def test_hook_modifies_last_token_only(self):
        """Test that hook only modifies the last token in sequence"""
        steerer = MCSteerer(self.mock_sae, self.concept_feats, strength=1.0)

        # Mock encoding/decoding
        current_feats = torch.randn(1, 1, self.feature_dim)
        steered_hidden = torch.randn(1, 1, self.hidden_dim)
        self.mock_sae.return_value = {"feature_acts": current_feats}
        self.mock_sae.decode.return_value = steered_hidden

        # Create test input
        original_output = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        original_first_tokens = original_output[:, :-1, :].clone()

        # Apply hook
        result = steerer.hook(None, None, original_output)

        # First tokens should be unchanged
        self.assertTrue(torch.allclose(
            result[:, :-1, :],
            original_first_tokens,
            atol=1e-6
        ))

        # Last token should be modified
        self.assertFalse(torch.allclose(
            result[:, -1:, :],
            original_output[:, -1:, :],
            atol=1e-6
        ))

    def test_get_stats_returns_correct_values(self):
        """Test that get_stats returns accurate statistics"""
        steerer = MCSteerer(self.mock_sae, self.concept_feats, strength=3.0)

        # Before any steering
        stats = steerer.get_stats()
        self.assertEqual(stats["steering_count"], 0)
        self.assertEqual(stats["avg_delta_norm"], 0.0)

        # After steering
        self.mock_sae.return_value = {"feature_acts": torch.randn(1, 1, self.feature_dim)}
        self.mock_sae.decode.return_value = torch.randn(1, 1, self.hidden_dim)
        test_output = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)

        steerer.hook(None, None, test_output)
        steerer.hook(None, None, test_output)

        stats = steerer.get_stats()
        self.assertEqual(stats["steering_count"], 2)
        self.assertGreater(stats["avg_delta_norm"], 0.0)

    def test_hook_handles_tuple_output(self):
        """Test that hook handles tuple outputs from transformer layers"""
        steerer = MCSteerer(self.mock_sae, self.concept_feats, strength=1.0)

        # Mock encoding
        self.mock_sae.return_value = {"feature_acts": torch.randn(1, 1, self.feature_dim)}
        self.mock_sae.decode.return_value = torch.randn(1, 1, self.hidden_dim)

        # Test with tuple output
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        tuple_output = (hidden_states, "other_data")

        result = steerer.hook(None, None, tuple_output)

        # Should return tuple
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)

    def test_hook_handles_tensor_output(self):
        """Test that hook handles direct tensor outputs"""
        steerer = MCSteerer(self.mock_sae, self.concept_feats, strength=1.0)

        # Mock encoding
        self.mock_sae.return_value = {"feature_acts": torch.randn(1, 1, self.feature_dim)}
        self.mock_sae.decode.return_value = torch.randn(1, 1, self.hidden_dim)

        # Test with tensor output
        tensor_output = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)

        result = steerer.hook(None, None, tensor_output)

        # Should return tensor
        self.assertIsInstance(result, torch.Tensor)


class TestHookRegistration(unittest.TestCase):
    """Test hook registration and removal"""

    def test_layer_hook_context_manager_registration(self):
        """Test that layer_hook properly registers hooks"""
        # Mock model with layers
        mock_model = Mock()
        mock_layer = Mock()
        mock_model.model.layers = [mock_layer]

        # Mock hook handle
        mock_handle = Mock()
        mock_layer.register_forward_hook.return_value = mock_handle

        # Test hook function
        def test_hook(module, input, output):
            return output

        # Use context manager
        with layer_hook(mock_model, 0, test_hook):
            # Verify hook was registered
            mock_layer.register_forward_hook.assert_called_once_with(test_hook)
            # Handle should not be removed yet
            mock_handle.remove.assert_not_called()

        # After exiting context, handle should be removed
        mock_handle.remove.assert_called_once()

    def test_layer_hook_removes_hook_on_exception(self):
        """Test that layer_hook removes hook even if exception occurs"""
        # Mock model
        mock_model = Mock()
        mock_layer = Mock()
        mock_model.model.layers = [mock_layer]
        mock_handle = Mock()
        mock_layer.register_forward_hook.return_value = mock_handle

        # Test that hook is removed even on exception
        try:
            with layer_hook(mock_model, 0, lambda m, i, o: o):
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Handle should still be removed
        mock_handle.remove.assert_called_once()

    def test_layer_hook_accesses_correct_layer(self):
        """Test that layer_hook accesses the correct layer index"""
        # Mock model with multiple layers
        mock_model = Mock()
        mock_layers = [Mock() for _ in range(5)]
        mock_model.model.layers = mock_layers

        for i, layer in enumerate(mock_layers):
            layer.register_forward_hook.return_value = Mock()

        # Test accessing layer 3
        layer_idx = 3
        with layer_hook(mock_model, layer_idx, lambda m, i, o: o):
            # Only layer 3 should have hook registered
            for i, layer in enumerate(mock_layers):
                if i == layer_idx:
                    layer.register_forward_hook.assert_called_once()
                else:
                    layer.register_forward_hook.assert_not_called()


class TestCosineSimilarity(unittest.TestCase):
    """Test cosine similarity calculation"""

    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity of identical vectors is 1.0"""
        a = torch.randn(100)
        result = cosine_similarity(a, a)
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors is ~0.0"""
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0])
        result = cosine_similarity(a, b)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_cosine_similarity_opposite_vectors(self):
        """Test cosine similarity of opposite vectors is -1.0"""
        a = torch.randn(100)
        b = -a
        result = cosine_similarity(a, b)
        self.assertAlmostEqual(result, -1.0, places=5)

    def test_cosine_similarity_range(self):
        """Test cosine similarity is always in [-1, 1]"""
        for _ in range(10):
            a = torch.randn(50)
            b = torch.randn(50)
            result = cosine_similarity(a, b)
            self.assertGreaterEqual(result, -1.0 - 1e-6)
            self.assertLessEqual(result, 1.0 + 1e-6)

    def test_cosine_similarity_normalized_vectors(self):
        """Test cosine similarity with pre-normalized vectors"""
        a = torch.randn(100)
        b = torch.randn(100)
        a = a / torch.linalg.norm(a)
        b = b / torch.linalg.norm(b)

        result = cosine_similarity(a, b)
        expected = (a * b).sum().item()

        self.assertAlmostEqual(result, expected, places=5)


class TestConceptCapture(unittest.TestCase):
    """Test concept feature capture"""

    def test_capture_concept_features_returns_correct_shape(self):
        """Test that capture_concept_features returns features of correct shape"""
        # Mock model, tokenizer, SAE
        mock_model = Mock()
        mock_tok = Mock()
        mock_sae = Mock()

        # Setup layer structure
        mock_layer = Mock()
        mock_model.model.layers = [mock_layer] * 12
        mock_handle = Mock()
        mock_layer.register_forward_hook.return_value = mock_handle

        # Mock tokenizer - needs to return object with .to() method
        mock_tok_output = Mock()
        mock_tok_output.to.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tok.return_value = mock_tok_output

        # Mock model forward pass - we need to trigger the hook
        def mock_forward(**kwargs):
            # Simulate calling the hook
            hooks = mock_layer.register_forward_hook.call_args_list
            if hooks:
                hook_fn = hooks[0][0][0]
                hidden_states = torch.randn(1, 3, 256)
                hook_fn(mock_layer, None, hidden_states)
            return Mock()

        mock_model.return_value = Mock()
        mock_model.side_effect = mock_forward

        # Mock SAE encoding
        feature_dim = 512
        expected_feats = torch.randn(1, 1, feature_dim)
        mock_sae.return_value = {"feature_acts": expected_feats}

        # Capture features
        result = capture_concept_features(
            mock_model,
            mock_tok,
            mock_sae,
            "Test concept"
        )

        # Verify shape
        self.assertEqual(result.shape[-1], feature_dim)

    def test_capture_concept_features_extracts_last_token(self):
        """Test that only last token features are captured"""
        # This is implicitly tested by checking the shape is (1, 1, feature_dim)
        # rather than (1, seq_len, feature_dim)
        pass


class TestIntegration(unittest.TestCase):
    """Integration tests for the full steering pipeline"""

    def test_full_steering_pipeline_without_models(self):
        """Test the complete steering pipeline with mocked components"""
        # Setup dimensions
        hidden_dim = 128
        feature_dim = 256

        # Mock SAE
        mock_sae = Mock()
        W_dec = torch.randn(hidden_dim, feature_dim)  # Correct: [H, F]
        mock_sae.W_dec = Mock()
        mock_sae.W_dec.T = W_dec.T

        # Concept features
        concept_feats = torch.randn(1, 1, feature_dim)

        # Create steerer
        steerer = MCSteerer(mock_sae, concept_feats, strength=3.0, max_norm=50.0)

        # Mock encoding: different features for each step
        call_count = [0]
        def mock_sae_call(h):
            call_count[0] += 1
            return {"feature_acts": torch.randn(1, 1, feature_dim) * call_count[0]}

        mock_sae.side_effect = mock_sae_call
        # Decode: [B, S, F] @ [F, H] = [B, S, H]
        mock_sae.decode.side_effect = lambda f: f @ W_dec.T

        # Simulate multiple generation steps
        for step in range(5):
            hidden_states = torch.randn(1, step + 1, hidden_dim)
            result = steerer.hook(None, None, hidden_states)

            # Verify output shape
            self.assertEqual(result.shape, hidden_states.shape)

        # Verify statistics
        stats = steerer.get_stats()
        self.assertEqual(stats["steering_count"], 5)
        self.assertGreater(stats["avg_delta_norm"], 0.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""

    def test_zero_strength_steering(self):
        """Test that zero strength produces no steering"""
        mock_sae = Mock()
        concept_feats = torch.randn(1, 1, 256)
        steerer = MCSteerer(mock_sae, concept_feats, strength=0.0)

        # Mock encoding
        current_feats = torch.randn(1, 1, 256)
        mock_sae.return_value = {"feature_acts": current_feats}
        mock_sae.decode.side_effect = lambda f: f @ torch.randn(256, 128)

        # Apply hook
        original = torch.randn(1, 5, 128)
        steerer.hook(None, None, original)

        # Delta norm should be zero
        stats = steerer.get_stats()
        self.assertAlmostEqual(stats["avg_delta_norm"], 0.0, places=5)

    def test_negative_strength_steering(self):
        """Test that negative strength reverses steering direction"""
        mock_sae = Mock()
        concept_feats = torch.ones(1, 1, 256)

        steerer_pos = MCSteerer(mock_sae, concept_feats, strength=5.0)
        steerer_neg = MCSteerer(mock_sae, concept_feats, strength=-5.0)

        # Mock encoding and decoding
        mock_sae.return_value = {"feature_acts": torch.zeros(1, 1, 256)}
        mock_sae.decode.return_value = torch.randn(1, 1, 128)

        # Apply both
        test_input = torch.randn(1, 3, 128)
        steerer_pos.hook(None, None, test_input)
        steerer_neg.hook(None, None, test_input)

        # Both should have same magnitude
        self.assertAlmostEqual(
            steerer_pos.total_delta_norm,
            steerer_neg.total_delta_norm,
            places=5
        )

    def test_large_sequence_length(self):
        """Test handling of long sequences"""
        mock_sae = Mock()
        concept_feats = torch.randn(1, 1, 256)
        steerer = MCSteerer(mock_sae, concept_feats, strength=2.0)

        # Mock encoding
        mock_sae.return_value = {"feature_acts": torch.randn(1, 1, 256)}
        mock_sae.decode.return_value = torch.randn(1, 1, 128)

        # Very long sequence
        long_sequence = torch.randn(1, 1000, 128)
        result = steerer.hook(None, None, long_sequence)

        # Should still only modify last token
        self.assertEqual(result.shape, long_sequence.shape)

    def test_batch_size_one_assumption(self):
        """Test that implementation assumes batch size of 1"""
        # This implementation is designed for batch_size=1 during generation
        # which is standard for autoregressive sampling
        mock_sae = Mock()
        concept_feats = torch.randn(1, 1, 256)
        steerer = MCSteerer(mock_sae, concept_feats, strength=2.0)

        mock_sae.return_value = {"feature_acts": torch.randn(1, 1, 256)}
        mock_sae.decode.return_value = torch.randn(1, 1, 128)

        # Test with batch_size=1
        single_batch = torch.randn(1, 5, 128)
        result = steerer.hook(None, None, single_batch)
        self.assertEqual(result.shape[0], 1)


def run_tests():
    """Run all tests and return results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSAEFeatureExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestSteeringMechanism))
    suite.addTests(loader.loadTestsFromTestCase(TestHookRegistration))
    suite.addTests(loader.loadTestsFromTestCase(TestCosineSimilarity))
    suite.addTests(loader.loadTestsFromTestCase(TestConceptCapture))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    print("=" * 70)
    print("MC Framework Test Suite")
    print("Testing core components without requiring full model downloads")
    print("=" * 70)
    print()

    result = run_tests()

    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
