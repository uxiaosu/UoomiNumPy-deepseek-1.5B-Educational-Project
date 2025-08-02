#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UoomiNumPy deepseek Educational Project

A pure NumPy implementation of the deepseek language model for educational purposes.
This project demonstrates how modern language models work under the hood.

Main entry point - now using modular CLI components.
"""

import sys
import argparse
from pathlib import Path

# Add CLI modules to path
sys.path.append(str(Path(__file__).parent / "cli"))

# Import CLI modules
from text_generation import generate_text
from chat_interface import run_chat
from weight_converter import convert_weights
from model_inspector import show_model_info
from test_runner import run_tests


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="UoomiNumPy deepseek Educational Project - Pure NumPy implementation of deepseek",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate text
  python main.py generate --model-dir ./model --prompt "The future of AI is"
  
  # Quick generation
  python main.py generate --model-dir ./model --prompt "Hello" --quick
  
  # Interactive chat
  python main.py chat --model-dir ./model
  
  # Convert weights
  python main.py convert --source-dir ./original_model --target-dir ./numpy_weights
  
  # Show model info
  python main.py info --model-dir ./model
  
  # Run tests
  python main.py test
        """
    )
    
    # Global arguments
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose output"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate text")
    generate_parser.add_argument(
        "--model-dir", "-m", 
        required=True, 
        help="Path to model directory"
    )
    generate_parser.add_argument(
        "--prompt", "-p", 
        required=True, 
        help="Text prompt for generation"
    )
    generate_parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=50, 
        help="Maximum tokens to generate (default: 50)"
    )
    generate_parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.8, 
        help="Sampling temperature (default: 0.8)"
    )
    generate_parser.add_argument(
        "--top-p", 
        type=float, 
        default=0.9, 
        help="Top-p (nucleus) sampling (default: 0.9)"
    )
    generate_parser.add_argument(
        "--top-k", 
        type=int, 
        default=50, 
        help="Top-k sampling (default: 50)"
    )
    generate_parser.add_argument(
        "--repetition-penalty", 
        type=float, 
        default=1.0, 
        help="Repetition penalty (default: 1.0)"
    )
    generate_parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Use quick generation (load model, generate, release)"
    )
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat session")
    chat_parser.add_argument(
        "--model-dir", "-m", 
        required=True, 
        help="Path to model directory"
    )
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert model weights")
    convert_parser.add_argument(
        "--source-dir", "-s", 
        required=True, 
        help="Source model directory"
    )
    convert_parser.add_argument(
        "--target-dir", "-t", 
        required=True, 
        help="Target directory for NumPy weights"
    )
    convert_parser.add_argument(
        "--format", 
        choices=["auto", "safetensors", "pytorch"], 
        default="auto", 
        help="Source weight format (default: auto)"
    )
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument(
        "--model-dir", "-m", 
        required=True, 
        help="Path to model directory"
    )
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run model tests")
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == "generate":
            generate_text(args)
        elif args.command == "chat":
            run_chat(args)
        elif args.command == "convert":
            convert_weights(args)
        elif args.command == "info":
            show_model_info(args)
        elif args.command == "test":
            run_tests(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n👋 Operation interrupted. Goodbye!")
        sys.exit(0)
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()