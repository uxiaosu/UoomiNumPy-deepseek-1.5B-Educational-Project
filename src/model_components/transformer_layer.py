#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer Layer Module

Single transformer layer implementation for the DeepSeek model.
"""

import numpy as np
from typing import Optional

# Import from parent directory
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import ModelConfig

# Import from current package
from .attention import MultiHeadAttention
from .mlp import MLP
from .normalization import RMSNorm


class TransformerLayer:
    """Single Transformer layer.
    
    Implements a standard transformer layer with:
    - Multi-head self-attention
    - Feed-forward network (MLP)
    - Residual connections
    - Layer normalization
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize transformer layer.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.hidden_size = config.hidden_size
        
        self.self_attn = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(self, hidden_states: np.ndarray, attention_mask: Optional[np.ndarray] = None, position_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm.forward(hidden_states)
        hidden_states = self.self_attn.forward(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        hidden_states = self.mlp.forward(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states