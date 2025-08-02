#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic Usage Example for UoomiNumPy deepseek Educational Project

This example demonstrates how to use the UoomiNumPy deepseek model for basic text generation.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from api import DeepSeekNumPy, load_model, quick_generate


def basic_text_generation():
    """Demonstrate basic text generation."""
    print("🚀 Basic Text Generation Example")
    print("=" * 50)
    
    # Path to your model directory
    model_dir = "../../DeepSeek-R1-Distill-Qwen-1.5B"
    
    # Load the model
    print("📦 Loading model...")
    model = load_model(model_dir)
    
    # Generate text
    prompts = [
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly,",
        "The most important lesson I learned today is",
        "Once upon a time, in a distant galaxy,"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n📝 Example {i}: {prompt}")
        print("-" * 40)
        
        # Generate with different parameters
        result = model.generate(
            prompt=prompt,
            max_new_tokens=30,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            verbose=False
        )
        
        print(f"Generated: {result}")
    
    print("\n✅ Basic text generation completed!")


def chat_example():
    """Demonstrate chat functionality."""
    print("\n💬 Chat Example")
    print("=" * 50)
    
    model_dir = "../../DeepSeek-R1-Distill-Qwen-1.5B"
    model = load_model(model_dir)
    
    # Chat examples
    chat_examples = [
        [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        [
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ],
        [
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": "What is 15 * 23?"}
        ]
    ]
    
    for i, messages in enumerate(chat_examples, 1):
        print(f"\n🗨️  Chat Example {i}:")
        print("-" * 30)
        
        for msg in messages:
            print(f"{msg['role'].title()}: {msg['content']}")
        
        response = model.chat(
            messages=messages,
            max_new_tokens=50,
            temperature=0.7,
            verbose=False
        )
        
        print(f"Assistant: {response}")
    
    print("\n✅ Chat examples completed!")


def batch_generation_example():
    """Demonstrate batch text generation."""
    print("\n📦 Batch Generation Example")
    print("=" * 50)
    
    model_dir = "../../DeepSeek-R1-Distill-Qwen-1.5B"
    model = load_model(model_dir)
    
    # Multiple prompts
    prompts = [
        "The benefits of renewable energy include",
        "Machine learning algorithms can help",
        "The key to successful teamwork is",
        "Climate change affects our planet by"
    ]
    
    print(f"🔄 Generating text for {len(prompts)} prompts...")
    
    results = model.batch_generate(
        prompts=prompts,
        max_new_tokens=25,
        temperature=0.8,
        verbose=False
    )
    
    for prompt, result in zip(prompts, results):
        print(f"\n📝 Prompt: {prompt}")
        print(f"Generated: {result}")
    
    print("\n✅ Batch generation completed!")


def tokenization_example():
    """Demonstrate tokenization functionality."""
    print("\n🔤 Tokenization Example")
    print("=" * 50)
    
    model_dir = "../../DeepSeek-R1-Distill-Qwen-1.5B"
    model = load_model(model_dir)
    
    # Test texts
    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "人工智能正在改变世界。",
        "What is the meaning of life, the universe, and everything?"
    ]
    
    for text in test_texts:
        print(f"\n📄 Text: {text}")
        print("-" * 30)
        
        # Encode
        token_ids = model.encode(text)
        print(f"Token IDs: {token_ids[:10]}{'...' if len(token_ids) > 10 else ''}")
        print(f"Number of tokens: {len(token_ids)}")
        
        # Decode
        decoded_text = model.decode(token_ids)
        print(f"Decoded: {decoded_text}")
        
        # Check consistency
        is_consistent = text.strip() == decoded_text.strip()
        print(f"Consistent: {'✅' if is_consistent else '❌'}")
    
    print("\n✅ Tokenization examples completed!")


def model_info_example():
    """Display model information."""
    print("\n📊 Model Information")
    print("=" * 50)
    
    model_dir = "../../DeepSeek-R1-Distill-Qwen-1.5B"
    model = load_model(model_dir)
    
    # Get model info
    info = model.get_model_info()
    
    print(f"Model Type: {info['model_type']}")
    print(f"Implementation: {info['implementation']}")
    print(f"Device: {info['device']}")
    print(f"Total Parameters: {info['total_parameters']:,}")
    
    print("\n📋 Configuration:")
    config = info['config']
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n🔤 Tokenizer:")
    tokenizer_info = info['tokenizer']
    for key, value in tokenizer_info.items():
        if key == 'special_tokens':
            print(f"  {key}:")
            for token_name, token_id in value.items():
                print(f"    {token_name}: {token_id}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✅ Model information displayed!")


def quick_generation_example():
    """Demonstrate quick generation without keeping model in memory."""
    print("\n⚡ Quick Generation Example")
    print("=" * 50)
    
    model_dir = "../../DeepSeek-R1-Distill-Qwen-1.5B"
    
    # Quick generation (loads model, generates, then releases memory)
    prompt = "The most fascinating thing about space exploration is"
    
    print(f"📝 Prompt: {prompt}")
    print("🔄 Generating (quick mode)...")
    
    result = quick_generate(
        model_dir=model_dir,
        prompt=prompt,
        max_new_tokens=40,
        temperature=0.8
    )
    
    print(f"Generated: {result}")
    print("\n✅ Quick generation completed!")


def main():
    """Run all examples."""
    print("🎯 UoomiNumPy deepseek Model - Usage Examples")
    print("=" * 60)
    
    try:
        # Run examples
        basic_text_generation()
        chat_example()
        batch_generation_example()
        tokenization_example()
        model_info_example()
        quick_generation_example()
        
        print("\n🎉 All examples completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please make sure the model directory path is correct.")
        print("Update the model_dir variable in the examples to point to your model.")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()