#!/usr/bin/env python3

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from config import ModelConfig
from model import (
    RMSNorm, RotaryPositionalEmbedding, LinearLayer, 
    MultiHeadAttention, MLP, TransformerLayer
)

def test_component_creation():
    """Test creating individual components to find the bottleneck."""
    print("🧪 Testing component creation...")
    
    # Load config
    print("📋 Loading config...")
    config = ModelConfig.from_json("./examples/weights/deepseek_numpy_weights/config.json")
    print(f"✅ Config loaded: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")
    
    # Test RMSNorm
    print("🔧 Testing RMSNorm creation...")
    start_time = time.time()
    norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    print(f"✅ RMSNorm created in {time.time() - start_time:.3f}s")
    
    # Test LinearLayer
    print("🔧 Testing LinearLayer creation...")
    start_time = time.time()
    linear = LinearLayer(config.hidden_size, config.vocab_size, bias=False)
    print(f"✅ LinearLayer created in {time.time() - start_time:.3f}s")
    
    # Test RotaryPositionalEmbedding
    print("🔧 Testing RotaryPositionalEmbedding creation...")
    start_time = time.time()
    rope = RotaryPositionalEmbedding(
        dim=config.hidden_size // config.num_attention_heads,
        max_position_embeddings=config.max_position_embeddings
    )
    print(f"✅ RotaryPositionalEmbedding created in {time.time() - start_time:.3f}s")
    
    # Test MLP
    print("🔧 Testing MLP creation...")
    start_time = time.time()
    mlp = MLP(config)
    print(f"✅ MLP created in {time.time() - start_time:.3f}s")
    
    # Test MultiHeadAttention
    print("🔧 Testing MultiHeadAttention creation...")
    start_time = time.time()
    attention = MultiHeadAttention(config)
    print(f"✅ MultiHeadAttention created in {time.time() - start_time:.3f}s")
    
    # Test TransformerLayer
    print("🔧 Testing TransformerLayer creation...")
    start_time = time.time()
    layer = TransformerLayer(config)
    print(f"✅ TransformerLayer created in {time.time() - start_time:.3f}s")
    
    # Test creating multiple layers
    print(f"🔧 Testing creation of {config.num_hidden_layers} TransformerLayers...")
    start_time = time.time()
    layers = []
    for i in range(config.num_hidden_layers):
        if i % 5 == 0:
            print(f"   Creating layer {i+1}/{config.num_hidden_layers}...")
        layers.append(TransformerLayer(config))
    print(f"✅ All {config.num_hidden_layers} TransformerLayers created in {time.time() - start_time:.3f}s")
    
    print("🎉 All components created successfully!")

if __name__ == "__main__":
    test_component_creation()