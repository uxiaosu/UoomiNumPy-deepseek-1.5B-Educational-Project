#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek Model Module

Main DeepSeek model implementation using modular components.
"""

import numpy as np
from typing import Dict, Optional

# Import from parent directory
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import ModelConfig

# Import from current package
from .linear import LinearLayer
from .transformer_layer import TransformerLayer
from .normalization import RMSNorm


class DeepSeekModel:
    """Complete DeepSeek model implementation.
    
    This is the main model class that combines all components:
    - Token embeddings
    - Multiple transformer layers
    - Final normalization
    - Language modeling head
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize DeepSeek model.
        
        Args:
            config: Model configuration
        """
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
        """Forward pass.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            
        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
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
        """Create causal attention mask.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Causal mask tensor
        """
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        mask = mask * -1e9  # Use large negative number instead of -inf
        return mask[None, None, :, :]
    
    def load_weights(self, weights_dict: Dict[str, np.ndarray]):
        """Load model weights from dictionary.
        
        Args:
            weights_dict: Dictionary mapping weight names to arrays
        """
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
        """Calculate total number of parameters.
        
        Returns:
            Total parameter count
        """
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