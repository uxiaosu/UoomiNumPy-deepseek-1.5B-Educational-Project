#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for UoomiNumPy deepseek Educational Project

This script provides a command-line interface for the UoomiNumPy deepseek model,
allowing users to quickly generate text, run chat sessions, or convert weights.
"""

import sys
import argparse
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from api import load_model, quick_generate


def generate_text(args):
    """Generate text using the model."""
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


def run_chat(args):
    """Run interactive chat session."""
    print(f"💬 Starting chat session with UoomiNumPy deepseek Model")
    print(f"📁 Model: {args.model_dir}")
    print("=" * 60)
    
    try:
        # Import and run chat example
        sys.path.append(str(Path(__file__).parent / "examples"))
        from chat_example import ChatSession
        
        chat = ChatSession(args.model_dir)
        chat.run()
        
    except Exception as e:
        print(f"❌ Error during chat: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def convert_weights(args):
    """Convert model weights to NumPy format."""
    print(f"🔄 Converting weights to NumPy format")
    print(f"📁 Source: {args.source_dir}")
    print(f"📁 Target: {args.target_dir}")
    print(f"📦 Format: {args.format}")
    print("=" * 60)
    
    try:
        # Import and run weight conversion
        sys.path.append(str(Path(__file__).parent / "examples"))
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


def show_model_info(args):
    """Show model information."""
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


def run_tests(args):
    """Run model tests."""
    print(f"🧪 Running UoomiNumPy deepseek Model Tests")
    print("=" * 60)
    
    try:
        # Import and run tests
        sys.path.append(str(Path(__file__).parent / "tests"))
        from test_basic import run_tests
        
        success = run_tests()
        
        if success:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


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