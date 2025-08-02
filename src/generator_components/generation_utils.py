#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generation Utilities Module

Utilities for text generation processes.
"""

import numpy as np
import time
from typing import List, Dict, Optional, Any

# Import from parent directory
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from tokenizer import DeepSeekTokenizer


class GenerationUtils:
    """Utilities for text generation processes."""
    
    @staticmethod
    def format_chat_messages(
        tokenizer: DeepSeekTokenizer,
        messages: List[Dict[str, str]]
    ) -> str:
        """Format chat messages using tokenizer's chat template.
        
        Args:
            tokenizer: DeepSeek tokenizer instance
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            Formatted prompt string
        """
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
    
    @staticmethod
    def prepare_input_ids(
        tokenizer: DeepSeekTokenizer,
        prompt: str,
        add_special_tokens: bool = True
    ) -> np.ndarray:
        """Prepare input IDs from prompt.
        
        Args:
            tokenizer: DeepSeek tokenizer instance
            prompt: Input prompt string
            add_special_tokens: Whether to add special tokens
            
        Returns:
            Input IDs as numpy array of shape (1, seq_len)
        """
        input_ids = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        return np.array([input_ids], dtype=np.int64)
    
    @staticmethod
    def extract_new_tokens(
        tokenizer: DeepSeekTokenizer,
        original_prompt: str,
        generated_sequence: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """Extract only the newly generated tokens from a sequence.
        
        Args:
            tokenizer: DeepSeek tokenizer instance
            original_prompt: Original input prompt
            generated_sequence: Complete generated token sequence
            skip_special_tokens: Whether to skip special tokens in decoding
            
        Returns:
            Newly generated text
        """
        # Get original prompt length
        original_length = len(tokenizer.encode(original_prompt, add_special_tokens=True))
        
        # Extract new tokens
        new_tokens = generated_sequence[original_length:]
        
        # Decode new tokens
        new_text = tokenizer.decode(new_tokens, skip_special_tokens=skip_special_tokens)
        
        return new_text
    
    @staticmethod
    def print_generation_progress(
        step: int,
        max_steps: int,
        start_time: float,
        generated_tokens: int,
        current_token: Optional[int] = None,
        update_interval: int = 5
    ):
        """Print generation progress information.
        
        Args:
            step: Current generation step (0-indexed)
            max_steps: Maximum number of steps
            start_time: Generation start time
            generated_tokens: Number of tokens generated so far
            current_token: Currently generated token ID
            update_interval: How often to print progress updates
        """
        if current_token is not None:
            print(f"🎲 Step {step + 1}: Sampled token: {current_token}")
        
        # Print progress update at intervals
        if (step + 1) % update_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = generated_tokens / elapsed if elapsed > 0 else 0
            print(f"   Generation progress: {step + 1}/{max_steps} ({tokens_per_sec:.1f} tokens/s)")
    
    @staticmethod
    def print_generation_summary(
        prompt: str,
        generated_text: str,
        generated_tokens: int,
        start_time: float,
        temperature: float,
        top_p: float,
        top_k: int
    ):
        """Print generation completion summary.
        
        Args:
            prompt: Original input prompt
            generated_text: Generated text
            generated_tokens: Number of tokens generated
            start_time: Generation start time
            temperature: Sampling temperature used
            top_p: Top-p value used
            top_k: Top-k value used
        """
        elapsed = time.time() - start_time
        tokens_per_sec = generated_tokens / elapsed if elapsed > 0 else 0
        
        print(f"\n✅ Generation completed!")
        print(f"   Generated tokens: {generated_tokens}")
        print(f"   Time elapsed: {elapsed:.2f}s")
        print(f"   Speed: {tokens_per_sec:.1f} tokens/s")
        print(f"   Temperature: {temperature}")
        print(f"   Top-p: {top_p}")
        print(f"   Top-k: {top_k}")
        print(f"   Prompt: {prompt}")
        print(f"   Generated: {generated_text}")
    
    @staticmethod
    def print_generation_start(
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        input_length: int
    ):
        """Print generation start information.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p value
            top_k: Top-k value
            input_length: Length of input in tokens
        """
        print(f"🎯 Starting text generation")
        print(f"   Input: {prompt}")
        print(f"   Max new tokens: {max_new_tokens}")
        print(f"   Temperature: {temperature}")
        print(f"   Top-p: {top_p}")
        print(f"   Top-k: {top_k}")
        print(f"   Input length: {input_length} tokens")
    
    @staticmethod
    def print_chat_start(
        formatted_prompt: str
    ):
        """Print chat generation start information.
        
        Args:
            formatted_prompt: Formatted chat prompt
        """
        print(f"💬 Starting chat generation")
        print(f"------------------------------")
        print(f"Formatted input:")
        print(formatted_prompt)
    
    @staticmethod
    def validate_generation_params(
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float
    ):
        """Validate generation parameters.
        
        Args:
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p value
            top_k: Top-k value
            repetition_penalty: Repetition penalty
            
        Raises:
            ValueError: If parameters are invalid
        """
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")