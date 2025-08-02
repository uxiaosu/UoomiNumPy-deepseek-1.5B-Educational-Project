#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek Model - Pure NumPy Implementation

A complete implementation of the DeepSeek language model using only NumPy.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import math

from config import ModelConfig


class ActivationFunctions:
    """Collection of activation functions."""
    
    @staticmethod
    def silu(x: np.ndarray) -> np.ndarray:
        """SiLU activation function: x * sigmoid(x)"""
        return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        """GELU activation function"""
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    
    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax function with numerical stability"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class RMSNorm:
    """RMS Normalization layer."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = np.ones(hidden_size, dtype=np.float32)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        variance = np.mean(x**2, axis=-1, keepdims=True)
        x_normalized = x / np.sqrt(variance + self.eps)
        return self.weight * x_normalized
    
    def load_weight(self, weight: np.ndarray):
        """Load weight parameters."""
        self.weight = weight.astype(np.float32)


class RotaryPositionalEmbedding:
    """Rotary Positional Embedding (RoPE)."""
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
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
        """Update cached cos and sin values."""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = np.arange(seq_len, dtype=np.float32)
            freqs = np.outer(t, self.inv_freq)
            emb = np.concatenate([freqs, freqs], axis=-1)
            self._cos_cached = np.cos(emb)
            self._sin_cached = np.sin(emb)
    
    def apply_rotary_pos_emb(self, q: np.ndarray, k: np.ndarray, position_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply rotary positional embedding to query and key tensors."""
        seq_len = q.shape[-2]
        self._update_cos_sin_cache(seq_len)
        
        cos = self._cos_cached[position_ids]
        sin = self._sin_cached[position_ids]
        
        # Apply rotation
        q_embed = self._rotate_half(q, cos, sin)
        k_embed = self._rotate_half(k, cos, sin)
        
        return q_embed, k_embed
    
    def _rotate_half(self, x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
        """Rotate half of the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return np.concatenate([-x2, x1], axis=-1) * sin + x * cos


class MultiHeadAttention:
    """Multi-Head Attention mechanism."""
    
    def __init__(self, config: ModelConfig):
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
        """Forward pass."""
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
        """Repeat key/value states for grouped query attention."""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = np.expand_dims(hidden_states, 2)
        hidden_states = np.repeat(hidden_states, n_rep, axis=2)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LinearLayer:
    """Linear transformation layer."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        # Initialize with None to avoid creating large random matrices during model creation
        # Weights will be loaded later via load_weight method
        self.weight = None
        self.bias = None if not bias else None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        if self.weight is None:
            raise RuntimeError(f"LinearLayer weights not loaded. Call load_weight() first.")
        output = np.dot(x, self.weight.T)
        if self.bias is not None:
            output += self.bias
        return output
    
    def load_weight(self, weight: np.ndarray, bias: Optional[np.ndarray] = None):
        """Load weight and bias parameters."""
        self.weight = weight.astype(np.float32)
        if bias is not None:
            self.bias = bias.astype(np.float32)


class MLP:
    """Multi-Layer Perceptron."""
    
    def __init__(self, config: ModelConfig):
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
        """Forward pass."""
        gate = self.act_fn(self.gate_proj.forward(x))
        up = self.up_proj.forward(x)
        return self.down_proj.forward(gate * up)


class TransformerLayer:
    """Single Transformer layer."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.hidden_size = config.hidden_size
        
        self.self_attn = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(self, hidden_states: np.ndarray, attention_mask: Optional[np.ndarray] = None, position_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass."""
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


class DeepSeekModel:
    """Complete DeepSeek model implementation."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
        # Embedding layer
        self.embed_tokens = LinearLayer(config.vocab_size, config.hidden_size, bias=False)
        
        # Transformer layers
        self.layers = [TransformerLayer(config) for _ in range(config.num_hidden_layers)]
        
        # Final normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Output layer
        self.lm_head = LinearLayer(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, input_ids: np.ndarray, attention_mask: Optional[np.ndarray] = None, position_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass."""
        batch_size, seq_len = input_ids.shape
        
        # Direct embedding lookup (much more efficient than one-hot)
        hidden_states = self.embed_tokens.weight[input_ids]
        
        # Create causal attention mask if not provided
        if attention_mask is None:
            attention_mask = self._create_causal_mask(seq_len)
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = np.arange(seq_len, dtype=np.int64)[None, :]
        
        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, attention_mask, position_ids)
        
        # Final normalization
        hidden_states = self.norm.forward(hidden_states)
        
        # Output projection
        logits = self.lm_head.forward(hidden_states)
        
        return logits
    
    def _create_causal_mask(self, seq_len: int) -> np.ndarray:
        """Create causal attention mask."""
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        mask = mask * -1e9  # Use large negative number instead of -inf
        return mask[None, None, :, :]
    
    def load_weights(self, weights_dict: Dict[str, np.ndarray]):
        """Load model weights from dictionary."""
        # Load embedding weights
        if 'model.embed_tokens.weight' in weights_dict:
            self.embed_tokens.load_weight(weights_dict['model.embed_tokens.weight'])
        
        # Load transformer layer weights
        for i, layer in enumerate(self.layers):
            layer_prefix = f'model.layers.{i}'
            
            # Attention weights
            if f'{layer_prefix}.self_attn.q_proj.weight' in weights_dict:
                bias = weights_dict.get(f'{layer_prefix}.self_attn.q_proj.bias')
                layer.self_attn.q_proj.load_weight(weights_dict[f'{layer_prefix}.self_attn.q_proj.weight'], bias)
            if f'{layer_prefix}.self_attn.k_proj.weight' in weights_dict:
                bias = weights_dict.get(f'{layer_prefix}.self_attn.k_proj.bias')
                layer.self_attn.k_proj.load_weight(weights_dict[f'{layer_prefix}.self_attn.k_proj.weight'], bias)
            if f'{layer_prefix}.self_attn.v_proj.weight' in weights_dict:
                bias = weights_dict.get(f'{layer_prefix}.self_attn.v_proj.bias')
                layer.self_attn.v_proj.load_weight(weights_dict[f'{layer_prefix}.self_attn.v_proj.weight'], bias)
            if f'{layer_prefix}.self_attn.o_proj.weight' in weights_dict:
                bias = weights_dict.get(f'{layer_prefix}.self_attn.o_proj.bias')
                layer.self_attn.o_proj.load_weight(weights_dict[f'{layer_prefix}.self_attn.o_proj.weight'], bias)
            
            # MLP weights
            if f'{layer_prefix}.mlp.gate_proj.weight' in weights_dict:
                layer.mlp.gate_proj.load_weight(weights_dict[f'{layer_prefix}.mlp.gate_proj.weight'])
            if f'{layer_prefix}.mlp.up_proj.weight' in weights_dict:
                layer.mlp.up_proj.load_weight(weights_dict[f'{layer_prefix}.mlp.up_proj.weight'])
            if f'{layer_prefix}.mlp.down_proj.weight' in weights_dict:
                layer.mlp.down_proj.load_weight(weights_dict[f'{layer_prefix}.mlp.down_proj.weight'])
            
            # Normalization weights
            if f'{layer_prefix}.input_layernorm.weight' in weights_dict:
                layer.input_layernorm.load_weight(weights_dict[f'{layer_prefix}.input_layernorm.weight'])
            if f'{layer_prefix}.post_attention_layernorm.weight' in weights_dict:
                layer.post_attention_layernorm.load_weight(weights_dict[f'{layer_prefix}.post_attention_layernorm.weight'])
        
        # Load final norm and output weights
        if 'model.norm.weight' in weights_dict:
            self.norm.load_weight(weights_dict['model.norm.weight'])
        if 'lm_head.weight' in weights_dict:
            self.lm_head.load_weight(weights_dict['lm_head.weight'])
    
    def get_total_parameters(self) -> int:
        """Calculate total number of parameters."""
        total = 0
        
        # Embedding parameters
        if self.embed_tokens.weight is not None:
            total += self.embed_tokens.weight.size
        
        # Transformer layer parameters
        for layer in self.layers:
            # Attention parameters
            if layer.self_attn.q_proj.weight is not None:
                total += layer.self_attn.q_proj.weight.size
            if layer.self_attn.k_proj.weight is not None:
                total += layer.self_attn.k_proj.weight.size
            if layer.self_attn.v_proj.weight is not None:
                total += layer.self_attn.v_proj.weight.size
            if layer.self_attn.o_proj.weight is not None:
                total += layer.self_attn.o_proj.weight.size
            
            # MLP parameters
            if layer.mlp.gate_proj.weight is not None:
                total += layer.mlp.gate_proj.weight.size
            if layer.mlp.up_proj.weight is not None:
                total += layer.mlp.up_proj.weight.size
            if layer.mlp.down_proj.weight is not None:
                total += layer.mlp.down_proj.weight.size
            
            # Normalization parameters
            if layer.input_layernorm.weight is not None:
                total += layer.input_layernorm.weight.size
            if layer.post_attention_layernorm.weight is not None:
                total += layer.post_attention_layernorm.weight.size
        
        # Final norm and output parameters
        if self.norm.weight is not None:
            total += self.norm.weight.size
        if self.lm_head.weight is not None:
            total += self.lm_head.weight.size
        
        return total