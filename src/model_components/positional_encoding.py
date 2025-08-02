#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Positional Encoding Module

Positional encoding implementations for the DeepSeek model.
"""

import numpy as np
from typing import Tuple


class RotaryPositionalEmbedding:
    """Rotary Positional Embedding (RoPE).
    
    Implementation of Rotary Position Embedding as described in:
    https://arxiv.org/abs/2104.09864
    """
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        """Initialize RoPE.
        
        Args:
            dim: Dimension of the embedding
            max_position_embeddings: Maximum sequence length
            base: Base for frequency computation
        """
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute frequency matrix
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        self.inv_freq = inv_freq
        
        # Cache for cos and sin values
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
    
    def _update_cos_sin_cache(self, seq_len: int):
        """Update cached cos and sin values.
        
        Args:
            seq_len: Sequence length to cache for
        """
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = np.arange(seq_len, dtype=np.float32)
            freqs = np.outer(t, self.inv_freq)
            emb = np.concatenate([freqs, freqs], axis=-1)
            self._cos_cached = np.cos(emb)
            self._sin_cached = np.sin(emb)
    
    def apply_rotary_pos_emb(self, q: np.ndarray, k: np.ndarray, position_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply rotary positional embedding to query and key tensors.
        
        Args:
            q: Query tensor
            k: Key tensor
            position_ids: Position indices
            
        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        seq_len = q.shape[-2]
        self._update_cos_sin_cache(seq_len)
        
        cos = self._cos_cached[position_ids]
        sin = self._sin_cached[position_ids]
        
        # Apply rotation
        q_embed = self._rotate_half(q, cos, sin)
        k_embed = self._rotate_half(k, cos, sin)
        
        return q_embed, k_embed
    
    def _rotate_half(self, x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
        """Rotate half of the hidden dims of the input.
        
        Args:
            x: Input tensor
            cos: Cosine values
            sin: Sine values
            
        Returns:
            Rotated tensor
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return np.concatenate([-x2, x1], axis=-1) * sin + x * cos