#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Runner Module

Handles test execution functionality.
"""

import sys
from pathlib import Path


def run_tests(args):
    """Run model tests.
    
    Args:
        args: Namespace object containing parameters:
            - verbose: Whether to show verbose output
    """
    print(f"🧪 Running UoomiNumPy deepseek Model Tests")
    print("=" * 60)
    
    try:
        # Import and run tests
        sys.path.append(str(Path(__file__).parent.parent / "tests"))
        from test_basic import run_tests as execute_tests
        
        success = execute_tests()
        
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