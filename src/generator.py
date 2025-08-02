#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek Text Generator

A text generation engine for the DeepSeek model using NumPy.
This module imports all the modularized generator components.
"""

# Import all generator components from the modular structure
try:
    # Try relative import first (when used as a package)
    from .generator_components import (
        TextGenerator,
        SamplingUtils,
        GenerationUtils
    )
except ImportError:
    # Fall back to absolute import (when run directly)
    from generator_components import (
        TextGenerator,
        SamplingUtils,
        GenerationUtils
    )

# Re-export for backward compatibility
__all__ = [
    'TextGenerator',
    'SamplingUtils',
    'GenerationUtils'
]