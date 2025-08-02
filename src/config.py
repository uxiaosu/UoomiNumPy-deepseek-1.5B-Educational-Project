#!/usr/bin/env python3
"""
Model Configuration

Defines the configuration class for the DeepSeek model.
"""

import json
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration class for DeepSeek model."""
    
    # Model architecture
    vocab_size: int = 151936
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    max_position_embeddings: int = 131072
    
    # Activation and normalization
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    
    # Attention
    attention_dropout: float = 0.0
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    
    # Generation
    bos_token_id: int = 100000
    eos_token_id: int = 100001
    pad_token_id: int = 100001
    
    # Model type
    model_type: str = "deepseek"
    
    @classmethod
    def from_json(cls, config_path: str) -> 'ModelConfig':
        """Load configuration from JSON file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # Filter out unsupported parameters
        supported_params = {
            'vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
            'num_attention_heads', 'num_key_value_heads', 'max_position_embeddings',
            'hidden_act', 'rms_norm_eps', 'attention_dropout', 'rope_theta',
            'rope_scaling', 'bos_token_id', 'eos_token_id', 'pad_token_id', 'model_type'
        }
        
        filtered_config = {k: v for k, v in config_dict.items() if k in supported_params}
        return cls(**filtered_config)
    
    def to_json(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=2, ensure_ascii=False)
    
    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_size // self.num_attention_heads