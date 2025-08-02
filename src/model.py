#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek Model Implementation

A complete implementation of the DeepSeek language model using NumPy.
This module imports all the modularized components.
"""

# Import all model components from the modular structure
try:
    # Try relative import first (when used as a package)
    from .model_components import (
        ActivationFunctions,
        RMSNorm,
        RotaryPositionalEmbedding,
        MultiHeadAttention,
        LinearLayer,
        MLP,
        TransformerLayer,
        DeepSeekModel
    )
except ImportError:
    # Fall back to absolute import (when run directly)
    from model_components import (
        ActivationFunctions,
        RMSNorm,
        RotaryPositionalEmbedding,
        MultiHeadAttention,
        LinearLayer,
        MLP,
        TransformerLayer,
        DeepSeekModel
    )

# Re-export for backward compatibility
__all__ = [
    'ActivationFunctions',
    'RMSNorm', 
    'RotaryPositionalEmbedding',
    'MultiHeadAttention',
    'LinearLayer',
    'MLP',
    'TransformerLayer',
    'DeepSeekModel'
]