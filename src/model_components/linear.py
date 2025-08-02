#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear Layer Module

Linear transformation layers for the DeepSeek model.
"""

import numpy as np
from typing import Optional


class LinearLayer:
    """Linear transformation layer.
    
    Implements a fully connected layer: y = xW^T + b
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """Initialize linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to include bias term
        """
        self.in_features = in_features
        self.out_features = out_features
        # Initialize with None to avoid creating large random matrices during model creation
        # Weights will be loaded later via load_weight method
        self.weight = None
        self.bias = None if not bias else None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
            
        Raises:
            RuntimeError: If weights haven't been loaded
        """
        if self.weight is None:
            raise RuntimeError(f"LinearLayer weights not loaded. Call load_weight() first.")
        output = np.dot(x, self.weight.T)
        if self.bias is not None:
            output += self.bias
        return output
    
    def load_weight(self, weight: np.ndarray, bias: Optional[np.ndarray] = None):
        """Load weight and bias parameters.
        
        Args:
            weight: Weight matrix of shape (out_features, in_features)
            bias: Optional bias vector of shape (out_features,)
        """
        self.weight = weight.astype(np.float32)
        if bias is not None:
            self.bias = bias.astype(np.float32)