#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat Interface Module

Handles interactive chat session functionality.
"""

import sys
from pathlib import Path


def run_chat(args):
    """Run interactive chat session.
    
    Args:
        args: Namespace object containing chat parameters:
            - model_dir: Path to model directory
            - verbose: Whether to show verbose output
    """
    print(f"💬 Starting chat session with UoomiNumPy deepseek Model")
    print(f"📁 Model: {args.model_dir}")
    print("=" * 60)
    
    try:
        # Import and run chat example
        sys.path.append(str(Path(__file__).parent.parent / "examples"))
        from chat_example import ChatSession
        
        chat = ChatSession(args.model_dir)
        chat.run()
        
    except Exception as e:
        print(f"❌ Error during chat: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)