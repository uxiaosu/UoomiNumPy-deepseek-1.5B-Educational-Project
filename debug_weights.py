#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from config import ModelConfig
from model import DeepSeekModel
from weight_loader import WeightLoader

def debug_weight_loading():
    """Debug weight loading process."""
    print("🔍 Debugging weight loading...")
    
    # Load config
    config = ModelConfig.from_json("./examples/weights/deepseek_numpy_weights/config.json")
    print(f"✅ Config loaded: {config.num_hidden_layers} layers")
    
    # Create model
    print("🧠 Creating model...")
    model = DeepSeekModel(config)
    print("✅ Model created")
    
    # Load weights
    print("📦 Loading weights...")
    weight_loader = WeightLoader("./examples/weights/deepseek_numpy_weights")
    weights_dict = weight_loader.load_weights()
    print(f"✅ Loaded {len(weights_dict)} weight tensors")
    
    # Check some key weights
    print("\n🔍 Checking key weights in dictionary:")
    key_weights = [
        'model.embed_tokens.weight',
        'model.layers.0.self_attn.q_proj.weight',
        'model.layers.0.self_attn.q_proj.bias',
        'lm_head.weight'
    ]
    for key in key_weights:
        if key in weights_dict:
            print(f"  ✅ {key}: {weights_dict[key].shape}")
        else:
            print(f"  ❌ {key}: NOT FOUND")
    
    # Apply weights to model
    print("\n⚡ Applying weights to model...")
    model.load_weights(weights_dict)
    print("✅ Weights applied")
    
    # Check if weights are loaded in model components
    print("\n🔍 Checking if weights are loaded in model:")
    print(f"  embed_tokens.weight: {model.embed_tokens.weight is not None}")
    print(f"  layers[0].self_attn.q_proj.weight: {model.layers[0].self_attn.q_proj.weight is not None}")
    print(f"  layers[0].self_attn.q_proj.bias: {model.layers[0].self_attn.q_proj.bias is not None}")
    print(f"  lm_head.weight: {model.lm_head.weight is not None}")
    
    if model.layers[0].self_attn.q_proj.weight is not None:
        print(f"  layers[0].self_attn.q_proj.weight shape: {model.layers[0].self_attn.q_proj.weight.shape}")
    
    print("\n🎉 Debug completed!")

if __name__ == "__main__":
    debug_weight_loading()