#!/usr/bin/env python3
"""
UoomiNumPy deepseek Educational Project Package

A pure NumPy implementation of the DeepSeek language model.
This package provides a complete, standalone implementation that can run
without PyTorch or other deep learning frameworks.
"""

from .model import DeepSeekModel
from .tokenizer import DeepSeekTokenizer
from .generator import TextGenerator
from .config import ModelConfig

__version__ = "1.0.0"
__author__ = "DeepSeek NumPy Implementation"

__all__ = [
    "DeepSeekModel",
    "DeepSeekTokenizer", 
    "TextGenerator",
    "ModelConfig"
]