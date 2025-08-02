#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test tokenizer loading to identify the issue
"""

import sys
import time
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from tokenizer import DeepSeekTokenizer

def test_tokenizer_loading():
    """Test tokenizer loading step by step."""
    tokenizer_dir = "./examples/weights/deepseek_numpy_weights"
    
    print("🔤 Starting tokenizer loading test...")
    print(f"📁 Tokenizer directory: {tokenizer_dir}")
    
    start_time = time.time()
    
    try:
        print("⏳ Creating tokenizer instance...")
        tokenizer = DeepSeekTokenizer(tokenizer_dir)
        
        elapsed = time.time() - start_time
        print(f"✅ Tokenizer loaded successfully in {elapsed:.2f}s")
        
        # Test basic functionality
        print("🧪 Testing basic tokenizer functionality...")
        test_text = "Hello world"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        
        print(f"   Original: {test_text}")
        print(f"   Tokens: {tokens}")
        print(f"   Decoded: {decoded}")
        print(f"   Vocab size: {tokenizer.get_vocab_size()}")
        
        print("✅ Tokenizer test completed successfully!")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Tokenizer loading failed after {elapsed:.2f}s")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tokenizer_loading()