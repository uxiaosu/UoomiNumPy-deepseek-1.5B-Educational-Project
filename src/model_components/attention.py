#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attention Module

Multi-head attention mechanism for the DeepSeek model.
"""

import math
import numpy as np
from typing import Optional

# Import from parent directory
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import ModelConfig

# Import from current package
from .linear import LinearLayer
from .positional_encoding import RotaryPositionalEmbedding
from .activations import ActivationFunctions


class MultiHeadAttention:
    """Multi-Head Attention mechanism.
    
    Implements the multi-head attention as described in "Attention Is All You Need".
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize multi-head attention.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        
        # Linear projections
        self.q_proj = LinearLayer(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = LinearLayer(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = LinearLayer(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = LinearLayer(self.num_heads * self.head_dim, self.hidden_size)
        
        # Rotary positional embedding
        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim, 
            max_position_embeddings=self.max_position_embeddings,
            base=config.rope_theta
        )
    
    def forward(self, hidden_states: np.ndarray, attention_mask: Optional[np.ndarray] = None, position_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Linear projections
        query_states = self.q_proj.forward(hidden_states)
        key_states = self.k_proj.forward(hidden_states)
        value_states = self.v_proj.forward(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Apply rotary positional embedding
        if position_ids is None:
            position_ids = np.arange(seq_len, dtype=np.int64)[None, :]
        
        query_states, key_states = self.rotary_emb.apply_rotary_pos_emb(query_states, key_states, position_ids)
        
        # Repeat key and value states for grouped query attention
        if self.num_key_value_heads != self.num_heads:
            key_states = self._repeat_kv(key_states, self.num_heads // self.num_key_value_heads)
            value_states = self._repeat_kv(value_states, self.num_heads // self.num_key_value_heads)
        
        # Compute attention scores
        attn_weights = np.matmul(query_states, key_states.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Apply softmax
        attn_weights = ActivationFunctions.softmax(attn_weights, axis=-1)
        
        # Apply attention to values
        attn_output = np.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj.forward(attn_output)
        
        return attn_output
    
    def _repeat_kv(self, hidden_states: np.ndarray, n_rep: int) -> np.ndarray:
        """Repeat key/value states for grouped query attention.
        
        Args:
            hidden_states: Input tensor
            n_rep: Number of repetitions
            
        Returns:
            Repeated tensor
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = np.expand_dims(hidden_states, 2)
        hidden_states = np.repeat(hidden_states, n_rep, axis=2)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)