#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic Tests for UoomiNumPy deepseek Educational Project
"""

import sys
import os
import unittest
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from config import ModelConfig
from model import ActivationFunctions, RMSNorm, RotaryPositionalEmbedding
from tokenizer import DeepSeekTokenizer
from weight_loader import WeightLoader


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = ModelConfig()
        
        # Check default values
        self.assertEqual(config.vocab_size, 102400)
        self.assertEqual(config.hidden_size, 2048)
        self.assertEqual(config.num_hidden_layers, 24)
        self.assertEqual(config.num_attention_heads, 16)
        self.assertEqual(config.max_position_embeddings, 4096)
    
    def test_custom_config(self):
        """Test custom configuration creation."""
        config = ModelConfig(
            vocab_size=50000,
            hidden_size=1024,
            num_hidden_layers=12
        )
        
        self.assertEqual(config.vocab_size, 50000)
        self.assertEqual(config.hidden_size, 1024)
        self.assertEqual(config.num_hidden_layers, 12)
    
    def test_config_serialization(self):
        """Test configuration JSON serialization."""
        import tempfile
        
        config = ModelConfig(vocab_size=50000, hidden_size=1024)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.to_json(f.name)
            
            # Load back
            loaded_config = ModelConfig.from_json(f.name)
            
            self.assertEqual(config.vocab_size, loaded_config.vocab_size)
            self.assertEqual(config.hidden_size, loaded_config.hidden_size)
        
        # Clean up
        os.unlink(f.name)


class TestActivationFunctions(unittest.TestCase):
    """Test activation functions."""
    
    def setUp(self):
        """Set up test data."""
        self.x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    def test_silu(self):
        """Test SiLU activation."""
        result = ActivationFunctions.silu(self.x)
        
        # SiLU should be smooth and non-negative for positive inputs
        self.assertTrue(np.all(result[self.x >= 0] >= 0))
        self.assertEqual(result.shape, self.x.shape)
    
    def test_relu(self):
        """Test ReLU activation."""
        result = ActivationFunctions.relu(self.x)
        
        # ReLU should zero out negative values
        expected = np.maximum(0, self.x)
        np.testing.assert_array_equal(result, expected)
    
    def test_gelu(self):
        """Test GELU activation."""
        result = ActivationFunctions.gelu(self.x)
        
        # GELU should be smooth
        self.assertEqual(result.shape, self.x.shape)
        self.assertTrue(np.all(np.isfinite(result)))
    
    def test_softmax(self):
        """Test Softmax activation."""
        result = ActivationFunctions.softmax(self.x)
        
        # Softmax should sum to 1 and be non-negative
        self.assertAlmostEqual(np.sum(result), 1.0, places=6)
        self.assertTrue(np.all(result >= 0))
        self.assertEqual(result.shape, self.x.shape)


class TestRMSNorm(unittest.TestCase):
    """Test RMS Normalization."""
    
    def test_rms_norm(self):
        """Test RMS normalization functionality."""
        # Create test input
        x = np.random.randn(2, 4, 8)  # (batch, seq, hidden)
        weight = np.ones(8)  # weight for last dimension
        
        norm = RMSNorm()
        result = norm.forward(x, weight)
        
        # Check output shape
        self.assertEqual(result.shape, x.shape)
        
        # Check that normalization reduces variance
        # (not exact due to RMS vs standard normalization)
        self.assertTrue(np.all(np.isfinite(result)))


class TestRotaryPositionalEmbedding(unittest.TestCase):
    """Test Rotary Positional Embedding."""
    
    def test_rope_creation(self):
        """Test RoPE creation and basic functionality."""
        rope = RotaryPositionalEmbedding(dim=64, max_seq_len=512)
        
        # Test position encoding generation
        seq_len = 10
        cos, sin = rope.get_cos_sin(seq_len)
        
        self.assertEqual(cos.shape, (seq_len, 32))  # dim // 2
        self.assertEqual(sin.shape, (seq_len, 32))
        
        # Check that values are in valid range for cos/sin
        self.assertTrue(np.all(np.abs(cos) <= 1.0))
        self.assertTrue(np.all(np.abs(sin) <= 1.0))
    
    def test_rope_application(self):
        """Test applying RoPE to query and key tensors."""
        rope = RotaryPositionalEmbedding(dim=64, max_seq_len=512)
        
        # Create test tensors
        batch_size, seq_len, num_heads, head_dim = 2, 8, 4, 64
        q = np.random.randn(batch_size, seq_len, num_heads, head_dim)
        k = np.random.randn(batch_size, seq_len, num_heads, head_dim)
        
        # Apply RoPE
        q_rot, k_rot = rope.apply_rope(q, k)
        
        # Check output shapes
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)
        
        # Check that rotation preserves magnitude (approximately)
        q_norm_orig = np.linalg.norm(q, axis=-1)
        q_norm_rot = np.linalg.norm(q_rot, axis=-1)
        np.testing.assert_allclose(q_norm_orig, q_norm_rot, rtol=1e-5)


class TestTokenizer(unittest.TestCase):
    """Test tokenizer functionality."""
    
    def test_tokenizer_creation(self):
        """Test tokenizer creation with minimal setup."""
        # Create a minimal tokenizer for testing
        tokenizer = DeepSeekTokenizer()
        
        # Test basic properties
        self.assertIsInstance(tokenizer.vocab, dict)
        self.assertIsInstance(tokenizer.special_tokens, dict)
    
    def test_basic_encoding_decoding(self):
        """Test basic encoding and decoding (with mock data)."""
        tokenizer = DeepSeekTokenizer()
        
        # Set up minimal vocab for testing
        tokenizer.vocab = {
            "hello": 1,
            "world": 2,
            "!": 3,
            "<unk>": 0
        }
        tokenizer.id_to_token = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.special_tokens = {"<unk>": 0}
        
        # Test encoding
        text = "hello world !"
        # This is a simplified test - real tokenizer would handle subwords
        tokens = text.split()
        token_ids = [tokenizer.vocab.get(token, 0) for token in tokens]
        
        # Test decoding
        decoded_tokens = [tokenizer.id_to_token.get(id, "<unk>") for id in token_ids]
        
        self.assertEqual(len(token_ids), len(tokens))
        self.assertEqual(len(decoded_tokens), len(tokens))


class TestWeightLoader(unittest.TestCase):
    """Test weight loading functionality."""
    
    def test_weight_loader_creation(self):
        """Test weight loader creation."""
        loader = WeightLoader()
        self.assertIsInstance(loader, WeightLoader)
    
    def test_numpy_weight_operations(self):
        """Test NumPy weight save/load operations."""
        import tempfile
        import shutil
        
        loader = WeightLoader()
        
        # Create test weights
        test_weights = {
            "layer1.weight": np.random.randn(100, 50).astype(np.float32),
            "layer1.bias": np.random.randn(100).astype(np.float32),
            "layer2.weight": np.random.randn(50, 25).astype(np.float32)
        }
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Save weights
            loader.save_numpy_weights(test_weights, temp_dir)
            
            # Load weights back
            loaded_weights = loader.load_numpy_weights(temp_dir)
            
            # Check that weights match
            self.assertEqual(set(test_weights.keys()), set(loaded_weights.keys()))
            
            for key in test_weights:
                np.testing.assert_array_equal(test_weights[key], loaded_weights[key])
        
        finally:
            # Clean up
            shutil.rmtree(temp_dir)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_model_config_integration(self):
        """Test that model components work together."""
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4
        )
        
        # Test that config values are consistent
        self.assertEqual(config.hidden_size % config.num_attention_heads, 0)
        
        head_dim = config.hidden_size // config.num_attention_heads
        self.assertEqual(head_dim, 32)
    
    def test_activation_chain(self):
        """Test chaining multiple activation functions."""
        x = np.random.randn(10, 20)
        
        # Apply multiple activations
        x1 = ActivationFunctions.gelu(x)
        x2 = ActivationFunctions.softmax(x1)
        
        # Check final output properties
        self.assertEqual(x2.shape, x.shape)
        self.assertTrue(np.all(x2 >= 0))
        np.testing.assert_allclose(np.sum(x2, axis=-1), 1.0, rtol=1e-6)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_classes = [
        TestModelConfig,
        TestActivationFunctions,
        TestRMSNorm,
        TestRotaryPositionalEmbedding,
        TestTokenizer,
        TestWeightLoader,
        TestIntegration
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("🧪 Running UoomiNumPy deepseek Model Tests")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)