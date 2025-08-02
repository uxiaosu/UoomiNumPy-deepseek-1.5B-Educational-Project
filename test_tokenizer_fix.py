#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify tokenizer fixes and improved generation quality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from tokenizer import DeepSeekTokenizer
from api import DeepSeekNumPy

def test_tokenizer_spaces():
    """Test if tokenizer correctly handles spaces."""
    print("🧪 Testing tokenizer space handling...")
    
    # Initialize tokenizer
    model_dir = "../DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = DeepSeekTokenizer(model_dir)
    
    # Test cases
    test_texts = [
        "What is 2+2?",
        "Hello world",
        "The quick brown fox",
        "This is a test sentence with multiple words"
    ]
    
    for text in test_texts:
        print(f"\n📝 Original: '{text}'")
        
        # Encode and decode
        tokens = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        
        print(f"🔢 Tokens: {tokens[:10]}..." if len(tokens) > 10 else f"🔢 Tokens: {tokens}")
        print(f"📤 Decoded: '{decoded}'")
        print(f"✅ Match: {text == decoded}")

def test_improved_generation():
    """Test improved generation quality with better parameters."""
    print("\n🚀 Testing improved generation...")
    
    # Initialize model
    model_dir = "../DeepSeek-R1-Distill-Qwen-1.5B"
    model = DeepSeekNumPy(model_dir)
    
    # Test prompts
    test_prompts = [
        "What is 2+2?",
        "The capital of France is",
        "Once upon a time"
    ]
    
    # Improved generation parameters
    generation_params = {
        'max_new_tokens': 30,
        'temperature': 0.8,  # Slightly lower for more coherent output
        'top_p': 0.9,
        'top_k': 40,
        'repetition_penalty': 1.1,  # Reduce repetition
        'verbose': True
    }
    
    for prompt in test_prompts:
        print(f"\n💭 Prompt: '{prompt}'")
        print("=" * 50)
        
        try:
            response = model.generate(prompt, **generation_params)
            print(f"\n📄 Generated Response:")
            print(f"'{response}'")
            print("=" * 50)
        except Exception as e:
            print(f"❌ Error: {e}")

def test_chat_format():
    """Test chat format generation."""
    print("\n💬 Testing chat format...")
    
    # Initialize model
    model_dir = "../DeepSeek-R1-Distill-Qwen-1.5B"
    model = DeepSeekNumPy(model_dir)
    
    # Test chat messages
    messages = [
        {"role": "user", "content": "What is 2+2? Please explain step by step."}
    ]
    
    try:
        response = model.chat(
            messages,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            verbose=True
        )
        print(f"\n📄 Chat Response:")
        print(f"'{response}'")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🔧 DeepSeek Tokenizer and Generation Quality Test")
    print("=" * 60)
    
    # Test tokenizer fixes
    test_tokenizer_spaces()
    
    # Test improved generation
    test_improved_generation()
    
    # Test chat format
    test_chat_format()
    
    print("\n✅ All tests completed!")