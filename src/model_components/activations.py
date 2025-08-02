#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Activation Functions Module

Collection of activation functions used in the DeepSeek model.
"""

import numpy as np


class ActivationFunctions:
    """Collection of activation functions."""
    
    @staticmethod
    def silu(x: np.ndarray) -> np.ndarray:
        """SiLU activation function: x * sigmoid(x)
        
        Args:
            x: Input array
            
        Returns:
            Activated array
        """
        return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function
        
        Args:
            x: Input array
            
        Returns:
            Activated array
        """
        return np.maximum(0, x)
    
    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        """GELU activation function
        
        Args:
            x: Input array
            
        Returns:
            Activated array
        """
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    
    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax function with numerical stability
        
        Args:
            x: Input array
            axis: Axis along which to apply softmax
            
        Returns:
            Softmax probabilities
        """
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)