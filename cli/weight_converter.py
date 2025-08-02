#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weight Converter Module

Handles model weight conversion to NumPy format.
"""

import sys
from pathlib import Path


def convert_weights(args):
    """Convert model weights to NumPy format.
    
    Args:
        args: Namespace object containing conversion parameters:
            - source_dir: Source directory path
            - target_dir: Target directory path
            - format: Output format type
            - verbose: Whether to show verbose output
    """
    print(f"🔄 Converting weights to NumPy format")
    print(f"📁 Source: {args.source_dir}")
    print(f"📁 Target: {args.target_dir}")
    print(f"📦 Format: {args.format}")
    print("=" * 60)
    
    try:
        # Import and run weight conversion
        sys.path.append(str(Path(__file__).parent.parent / "examples"))
        from weight_conversion import convert_model_weights
        
        convert_model_weights(
            source_dir=args.source_dir,
            target_dir=args.target_dir,
            format_type=args.format
        )
        
        print("✅ Weight conversion completed!")
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)