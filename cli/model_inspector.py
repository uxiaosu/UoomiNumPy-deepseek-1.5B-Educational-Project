#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Inspector Module

Handles model information display functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from api import load_model


def show_model_info(args):
    """Show model information.
    
    Args:
        args: Namespace object containing parameters:
            - model_dir: Path to model directory
            - verbose: Whether to show verbose output
    """
    print(f"📊 Model Information")
    print(f"📁 Model: {args.model_dir}")
    print("=" * 60)
    
    try:
        model = load_model(args.model_dir)
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
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)