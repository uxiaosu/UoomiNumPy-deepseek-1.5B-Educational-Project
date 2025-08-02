#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI Module - Command Line Interface Components

This module contains all command-line interface functionality,
separated from the main entry point for better modularity.
"""

from .text_generation import generate_text
from .chat_interface import run_chat
from .weight_converter import convert_weights
from .model_inspector import show_model_info
from .test_runner import run_tests

__all__ = [
    'generate_text',
    'run_chat', 
    'convert_weights',
    'show_model_info',
    'run_tests'
]