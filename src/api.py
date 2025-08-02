#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek NumPy API

Provides a simple, high-level API for using the DeepSeek model.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import time

from config import ModelConfig
from model import DeepSeekModel
from tokenizer import DeepSeekTokenizer
from generator import TextGenerator
from weight_loader import WeightLoader


class DeepSeekNumPy:
    """High-level API for DeepSeek model."""
    
    def __init__(
        self, 
        model_dir: str, 
        device: str = "cpu",
        load_weights: bool = True,
        optimize_weights: bool = True
    ):
        """Initialize the DeepSeek model.
        
        Args:
            model_dir: Directory containing model files.
            device: Device to run on (only 'cpu' supported).
            load_weights: Whether to load model weights immediately.
            optimize_weights: Whether to optimize weights for inference.
        """
        self.model_dir = Path(model_dir)
        self.device = device
        
        if device != "cpu":
            print(f"Warning: Only CPU device is supported. Using CPU instead of {device}")
            self.device = "cpu"
        
        print(f"🚀 Initializing UoomiNumPy deepseek Model")
        print(f"   Model directory: {self.model_dir}")
        print(f"   Device: {self.device}")
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.tokenizer = self._load_tokenizer()
        self.model = DeepSeekModel(self.config)
        self.generator = TextGenerator(self.model, self.tokenizer, self.config)
        
        # Load weights if requested
        if load_weights:
            self.load_model_weights(optimize=optimize_weights)
        
        print(f"✅ UoomiNumPy deepseek Model initialized successfully")
    
    def _load_config(self) -> ModelConfig:
        """Load model configuration."""
        config_path = self.model_dir / "config.json"
        
        if config_path.exists():
            print(f"📋 Loading configuration from {config_path}")
            return ModelConfig.from_json(str(config_path))
        else:
            print(f"⚠️  Configuration file not found, using default config")
            return ModelConfig()
    
    def _load_tokenizer(self) -> DeepSeekTokenizer:
        """Load tokenizer."""
        print(f"🔤 Loading tokenizer from {self.model_dir}")
        return DeepSeekTokenizer(str(self.model_dir))
    
    def load_model_weights(self, optimize: bool = True):
        """Load model weights.
        
        Args:
            optimize: Whether to optimize weights for inference.
        """
        print(f"⚖️  Loading model weights...")
        
        weight_loader = WeightLoader(str(self.model_dir))
        weights = weight_loader.load_weights()
        
        # Validate weights
        if not weight_loader.validate_weights(weights):
            raise ValueError("Weight validation failed")
        
        # Optimize weights if requested
        if optimize:
            weights = weight_loader.optimize_weights(weights)
        
        # Load weights into model
        self.model.load_weights(weights)
        
        # Get model info
        weight_info = weight_loader.get_weight_info(weights)
        total_params = self.model.get_total_parameters()
        
        print(f"✅ Model weights loaded successfully")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Memory usage: {weight_info['total_memory_mb']:.1f} MB")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        verbose: bool = False
    ) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_p: Nucleus sampling probability threshold.
            top_k: Top-k sampling parameter.
            do_sample: Whether to use sampling or greedy decoding.
            repetition_penalty: Penalty for repeating tokens.
            verbose: Whether to print generation progress.
            
        Returns:
            Generated text string.
        """
        return self.generator.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            verbose=verbose
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        verbose: bool = False
    ) -> str:
        """Generate a chat response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability threshold.
            top_k: Top-k sampling parameter.
            verbose: Whether to print generation progress.
            
        Returns:
            Generated response string.
        """
        return self.generator.chat(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            verbose=verbose
        )
    
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        verbose: bool = False
    ) -> List[str]:
        """Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability threshold.
            top_k: Top-k sampling parameter.
            verbose: Whether to print generation progress.
            
        Returns:
            List of generated text strings.
        """
        return self.generator.batch_generate(
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            verbose=verbose
        )
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Input text to encode.
            add_special_tokens: Whether to add special tokens.
            
        Returns:
            List of token IDs.
        """
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs to decode.
            skip_special_tokens: Whether to skip special tokens.
            
        Returns:
            Decoded text string.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def get_perplexity(self, text: str) -> float:
        """Calculate perplexity of the given text.
        
        Args:
            text: Input text to evaluate.
            
        Returns:
            Perplexity score.
        """
        return self.generator.get_perplexity(text)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dictionary containing model information.
        """
        return {
            'model_type': 'DeepSeek',
            'implementation': 'Pure NumPy',
            'config': {
                'vocab_size': self.config.vocab_size,
                'hidden_size': self.config.hidden_size,
                'num_layers': self.config.num_hidden_layers,
                'num_attention_heads': self.config.num_attention_heads,
                'max_position_embeddings': self.config.max_position_embeddings
            },
            'tokenizer': {
                'vocab_size': self.tokenizer.get_vocab_size(),
                'special_tokens': self.tokenizer.get_special_tokens_dict()
            },
            'total_parameters': self.model.get_total_parameters(),
            'device': self.device
        }
    
    def save_model(self, output_dir: str, save_weights: bool = True):
        """Save the model to a directory.
        
        Args:
            output_dir: Output directory to save the model.
            save_weights: Whether to save model weights.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving model to {output_path}")
        
        # Save configuration
        config_path = output_path / "config.json"
        self.config.to_json(str(config_path))
        print(f"   Saved config: {config_path}")
        
        # Save model info
        info_path = output_path / "model_info.json"
        with open(info_path, 'w') as f:
            json.dump(self.get_model_info(), f, indent=2)
        print(f"   Saved model info: {info_path}")
        
        # Save weights if requested
        if save_weights:
            weights_dir = output_path / "weights"
            weight_loader = WeightLoader(str(self.model_dir))
            weights = weight_loader.load_weights()
            weight_loader.save_weights_numpy(weights, str(weights_dir))
            print(f"   Saved weights: {weights_dir}")
        
        print(f"✅ Model saved successfully")
    
    @classmethod
    def from_pretrained(
        cls, 
        model_dir: str, 
        **kwargs
    ) -> 'DeepSeekNumPy':
        """Load a pretrained model.
        
        Args:
            model_dir: Directory containing model files.
            **kwargs: Additional arguments for initialization.
            
        Returns:
            Initialized DeepSeekNumPy instance.
        """
        return cls(model_dir=model_dir, **kwargs)
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Make the model callable for text generation.
        
        Args:
            prompt: Input text prompt.
            **kwargs: Additional arguments for generation.
            
        Returns:
            Generated text string.
        """
        return self.generate(prompt, **kwargs)


def load_model(model_dir: str, **kwargs) -> DeepSeekNumPy:
    """Convenience function to load a DeepSeek model.
    
    Args:
        model_dir: Directory containing model files.
        **kwargs: Additional arguments for initialization.
        
    Returns:
        Initialized DeepSeekNumPy instance.
    """
    return DeepSeekNumPy.from_pretrained(model_dir, **kwargs)


def quick_generate(model_dir: str, prompt: str, **kwargs) -> str:
    """Quick text generation without keeping model in memory.
    
    Args:
        model_dir: Directory containing model files.
        prompt: Input text prompt.
        **kwargs: Additional arguments for generation.
        
    Returns:
        Generated text string.
    """
    model = load_model(model_dir)
    return model.generate(prompt, **kwargs)