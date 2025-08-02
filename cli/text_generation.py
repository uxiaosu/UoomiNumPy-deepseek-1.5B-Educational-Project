#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Generation Module

Handles text generation functionality with various parameters.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from api import load_model, quick_generate


def generate_text(args):
    """Generate text using the model.
    
    Args:
        args: Namespace object containing generation parameters:
            - model_dir: Path to model directory
            - prompt: Input prompt text
            - max_tokens: Maximum tokens to generate
            - temperature: Sampling temperature
            - top_p: Top-p sampling parameter
            - top_k: Top-k sampling parameter
            - repetition_penalty: Repetition penalty
            - quick: Whether to use quick generation mode
            - verbose: Whether to show verbose output
    """
    print(f"🚀 Generating text with UoomiNumPy deepseek Model")
    print(f"📁 Model: {args.model_dir}")
    print(f"📝 Prompt: {args.prompt}")
    print("=" * 60)
    
    try:
        if args.quick:
            # Quick generation (load model, generate, release)
            result = quick_generate(
                model_dir=args.model_dir,
                prompt=args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k
            )
        else:
            # Load model and generate
            model = load_model(args.model_dir)
            result = model.generate(
                prompt=args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                verbose=args.verbose
            )
        
        print(f"\n📄 Generated Text:")
        print("-" * 40)
        print(result)
        print("-" * 40)
        print("✅ Generation completed!")
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)