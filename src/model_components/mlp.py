#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP Module

Multi-Layer Perceptron implementation for the DeepSeek model.
"""

import numpy as np

# Import from parent directory
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import ModelConfig

# Import from current package
from .linear import LinearLayer
from .activations import ActivationFunctions


class MLP:
    """Multi-Layer Perceptron.
    
    Implements the feed-forward network used in transformer layers.
    Uses SwiGLU activation (gate mechanism with SiLU activation).
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize MLP.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = LinearLayer(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = LinearLayer(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = LinearLayer(self.intermediate_size, self.hidden_size, bias=False)
        
        # Activation function
        if config.hidden_act == "silu":
            self.act_fn = ActivationFunctions.silu
        elif config.hidden_act == "relu":
            self.act_fn = ActivationFunctions.relu
        elif config.hidden_act == "gelu":
            self.act_fn = ActivationFunctions.gelu
        else:
            raise ValueError(f"Unsupported activation function: {config.hidden_act}")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (..., hidden_size)
            
        Returns:
            Output tensor of shape (..., hidden_size)
        """
        gate = self.act_fn(self.gate_proj.forward(x))
        up = self.up_proj.forward(x)
        return self.down_proj.forward(gate * up)