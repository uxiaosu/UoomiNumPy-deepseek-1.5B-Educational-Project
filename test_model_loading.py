#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test model loading to identify the issue
"""

import sys
import time
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from config import ModelConfig
from model import DeepSeekModel
from tokenizer import DeepSeekTokenizer
from weight_loader import WeightLoader

def test_model_loading():
    """Test model loading step by step."""
    model_dir = "./examples/weights/deepseek_numpy_weights"
    
    print("🚀 Starting model loading test...")
    print(f"📁 Model directory: {model_dir}")
    
    try:
        # Step 1: Load configuration
        print("📋 Step 1: Loading configuration...")
        start_time = time.time()
        config = ModelConfig.from_json(f"{model_dir}/config.json")
        elapsed = time.time() - start_time
        print(f"✅ Configuration loaded in {elapsed:.2f}s")
        print(f"   Hidden size: {config.hidden_size}")
        print(f"   Vocab size: {config.vocab_size}")
        print(f"   Num layers: {config.num_hidden_layers}")
        
        # Step 2: Load tokenizer
        print("🔤 Step 2: Loading tokenizer...")
        start_time = time.time()
        tokenizer = DeepSeekTokenizer(model_dir)
        elapsed = time.time() - start_time
        print(f"✅ Tokenizer loaded in {elapsed:.2f}s")
        
        # Step 3: Create model
        print("🧠 Step 3: Creating model...")
        start_time = time.time()
        model = DeepSeekModel(config)
        elapsed = time.time() - start_time
        print(f"✅ Model created in {elapsed:.2f}s")
        
        # Step 4: Load weights
        print("⚖️ Step 4: Loading weights...")
        start_time = time.time()
        weight_loader = WeightLoader(model_dir)
        weights_dict = weight_loader.load_weights()
        elapsed = time.time() - start_time
        print(f"✅ Weights loaded in {elapsed:.2f}s")
        print(f"   Number of weight tensors: {len(weights_dict)}")
        
        # Step 5: Apply weights to model
        print("🔧 Step 5: Applying weights to model...")
        start_time = time.time()
        model.load_weights(weights_dict)
        elapsed = time.time() - start_time
        print(f"✅ Weights applied in {elapsed:.2f}s")
        
        # Step 6: Test a simple forward pass
        print("🔄 Step 6: Testing forward pass...")
        start_time = time.time()
        
        # Encode a simple prompt
        test_prompt = "Hello"
        input_ids = tokenizer.encode(test_prompt, add_special_tokens=True)
        input_ids = [[input_ids[0]]]  # Just use the first token to keep it simple
        
        import numpy as np
        input_ids = np.array(input_ids, dtype=np.int64)
        print(f"   Input shape: {input_ids.shape}")
        print(f"   Input IDs: {input_ids}")
        
        # Forward pass
        logits = model.forward(input_ids)
        elapsed = time.time() - start_time
        print(f"✅ Forward pass completed in {elapsed:.2f}s")
        print(f"   Output shape: {logits.shape}")
        print(f"   Output sample: {logits[0, 0, :5]}")
        
        print("🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_loading()