#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalization Module

Normalization layers for the DeepSeek model.
"""

import numpy as np


class RMSNorm:
    """RMS Normalization layer.
    
    Root Mean Square Layer Normalization as described in:
    https://arxiv.org/abs/1910.07467
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """Initialize RMSNorm layer.
        
        Args:
            hidden_size: Size of the hidden dimension
            eps: Small epsilon for numerical stability
        """
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = np.ones(hidden_size, dtype=np.float32)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (..., hidden_size)
            
        Returns:
            Normalized tensor
        """
        variance = np.mean(x**2, axis=-1, keepdims=True)
        x_normalized = x / np.sqrt(variance + self.eps)
        return self.weight * x_normalized
    
    def load_weight(self, weight: np.ndarray):
        """Load weight parameters.
        
        Args:
            weight: Weight tensor of shape (hidden_size,)
        """
        self.weight = weight.astype(np.float32)